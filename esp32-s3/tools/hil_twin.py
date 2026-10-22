#!/usr/bin/env python3
"""Run the ESP32's policy against the MuJoCo digital twin, and diff its
observation against the Python one every tick.

The ESP32 is inserted as the policy: hardware/k2_policy_run.py keeps its
approach ramp, slew limiting, tilt cutoff and viewer, but the actions come from
the firmware. The observation ingredients are handed to the firmware, which
rebuilds the *stateful* channels itself -- last_action, the gait phase, and the
heading error -- so the diff exercises exactly the parts a port gets wrong.

Nothing on the robot is touched; the twin closes the loop.

    python esp32-s3/tools/hil_twin.py --port /dev/ttyACM0 --viewer
"""
import argparse, struct, sys, time
import numpy as np
import onnxruntime as ort
import serial

sys.path.insert(0, ".")
from hardware import k2_bus
from hardware.k2_policy_run import _read_metadata, run, _channel5_contract
from hardware.k2_stabilizer import UnsafeTiltError

N_IN, N_OUT = 35, 62
DEFAULT = np.array([-0.149, 0, 0, -0.611, 0.462, 0,
                    -0.153, 0, 0, -0.611, 0.458, 0], np.float32)


class Esp32Policy:
    """Quacks like an InferenceSession; forwards inference to the firmware."""

    def __init__(self, port, ref_session, baud=115200, marching=True):
        self.ser = serial.Serial(port, baud, timeout=1.0)
        self.ref = ref_session
        self.marching = marching
        self.obs_err = 0.0
        self.act_err = 0.0
        self.worst_channel = -1
        self.ticks = 0
        self.fails = 0
        self.trace = []
        # Reset the board: gaitAct, phase, lastAction and headingRef are static
        # state, and a warm board carries the previous session's values into
        # tick 0 -- which looks exactly like a port bug.
        self.ser.setRTS(True); self.ser.setDTR(False)
        time.sleep(0.05)
        self.ser.setDTR(True)
        time.sleep(2.5)
        self.ser.reset_input_buffer()

    # -- InferenceSession surface used by k2_policy_run --
    def get_inputs(self):     return self.ref.get_inputs()
    def get_outputs(self):    return self.ref.get_outputs()
    def get_modelmeta(self):  return self.ref.get_modelmeta()

    def run(self, _outputs, feed):
        obs = np.asarray(list(feed.values())[0], np.float32).ravel()

        # Recover the ingredients from the observation the Python built. The
        # firmware regenerates last_action / gait / heading from its OWN state,
        # so those channels are a real comparison rather than an echo.
        gyro, grav = obs[0:3], obs[3:6]
        q = obs[6:18] + DEFAULT
        qd = obs[18:30]
        # obs carries the GATED commands; recover the raw ones so the firmware
        # applies its own gate and the comparison actually covers it.
        m = float(obs[44])
        vx = obs[47] / m if m > 1e-6 else 0.0
        wz = obs[48] / m if m > 1e-6 else 0.0
        dh = obs[49] / (1.0 - m) if (1.0 - m) > 1e-6 else 0.0

        # Recover the heading the Python used from its own sin/cos channels,
        # plus the reference it had already integrated, so the firmware starts
        # from the same absolute yaw rather than a differently-sourced one.
        # Order matters: Python advances the reference with the GATED yaw
        # command (obs[48]) and only THEN reads the error against it. Doing it
        # the other way round leaves the reconstruction one tick stale, which
        # shows up as exactly yaw_rate*dt of heading error.
        self.head_ref = getattr(self, "head_ref", 0.0)
        self.head_ref += float(obs[48]) / 50.0
        self.head_ref = (self.head_ref + np.pi) % (2 * np.pi) - np.pi
        heading_err = float(np.arctan2(obs[42], obs[43]))
        heading_abs = heading_err + self.head_ref
        payload = np.concatenate([gyro, grav, q, qd,
                                  [vx, wz, dh, 1.0 if self.marching else 0.0,
                                   heading_abs]]).astype(np.float32)
        self.ser.write(b"K2" + payload.tobytes())
        self.ser.flush()

        hdr = self.ser.read(2)
        if hdr != b"K2":
            self.fails += 1
            return [self.ref.run(None, {"obs": obs[None]})[0]]   # fall back, keep the sim alive
        raw = self.ser.read(N_OUT * 4)
        if len(raw) != N_OUT * 4:
            self.fails += 1
            return [self.ref.run(None, {"obs": obs[None]})[0]]

        out = np.frombuffer(raw, np.float32)
        action, esp_obs = out[:12], out[12:]

        ref_action = self.ref.run(None, {"obs": obs[None]})[0].ravel()
        d = np.abs(esp_obs - obs)
        self.per_channel = np.maximum(getattr(self, "per_channel", np.zeros(50)), d)
        if d.max() > self.obs_err:
            self.obs_err = float(d.max())
            self.worst_channel = int(d.argmax())
        self.act_err = max(self.act_err, float(np.abs(action - ref_action).max()))
        if self.ticks < 60 or (self.ticks % 100 == 0 and self.ticks < 400):
            self.trace.append((self.ticks, obs[44], obs[45], obs[46],
                               esp_obs[44], esp_obs[45], esp_obs[46]))
        self.ticks += 1
        return [action.reshape(1, 12).astype(np.float32)]

    def report(self):
        names = (["gyro"]*3 + ["grav"]*3 + ["q_rel"]*12 + ["qd"]*12
                 + ["last_action"]*12 + ["heading"]*2 + ["gait"]*6)
        print(f"\n--- ESP32 vs Python over {self.ticks} ticks ---")
        print(f"  frame failures        : {self.fails}")
        print(f"  max |obs difference|  : {self.obs_err:.3e}", end="")
        if self.worst_channel >= 0:
            print(f"  (channel {self.worst_channel} = {names[self.worst_channel]})")
        else:
            print()
        print(f"  max |action difference|: {self.act_err:.3e}")
        groups = [("gyro",0,3),("grav",3,6),("q_rel",6,18),("qd",18,30),
                  ("last_action",30,42),("heading",42,44),("gait",44,50)]
        print("  per-group max |difference|:")
        for name, a, b in groups:
            m = float(self.per_channel[a:b].max())
            print(f"    {name:<12} {m:.3e}  {'ok' if m < 1e-4 else '<== DIVERGENT'}")
        print("  gait trace (py m,sin,cos | esp m,sin,cos):")
        for t, a, b, c, d, e, f in self.trace[:14] + self.trace[-6:]:
            print(f"    t{t:<4} {a:+.3f} {b:+.3f} {c:+.3f} | {d:+.3f} {e:+.3f} {f:+.3f}"
                  f"   {'' if abs(a-d)+abs(b-e)+abs(c-f) < 1e-5 else '<-- diff'}")
        # Every channel but heading is bit-exact. headingRef is an integrator run
        # in float32 on the MCU and float64 in Python, so its error grows with
        # run length (measured 8.7e-06 @4 s, 4.1e-05 @8 s, 1.6e-04 @16 s) while
        # the rest stay at 0.0. Budget it per-second rather than as a flat cap.
        head_budget = 2e-5 * max(1.0, self.ticks / 50.0)
        others = float(np.max(np.concatenate([self.per_channel[:42],
                                              self.per_channel[44:]])))
        ok = others < 1e-4 and self.per_channel[42:44].max() < head_budget
        verdict = "MATCH" if ok else "DIVERGENT"
        print(f"  verdict: {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--policy", default="policies/turn_v11_iter3400.onnx")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--mode", default="march")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--forward-speed", type=float, default=0.03)
    ap.add_argument("--yaw-rate", type=float, default=0.0)
    args = ap.parse_args()

    ref = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    (default_pos, action_scale, gait_freq_hz, _iter, _shift, gait_dim,
     heading_dim, gait_transition_time_s, vel_ramp, yaw_ramp,
     height_ramp) = _read_metadata(ref)
    md = ref.get_modelmeta().custom_metadata_map
    ch5, _ = _channel5_contract(ref)

    sess = Esp32Policy(args.port, ref, marching=args.mode in ('march', 'walk'))
    bus = k2_bus.SimBus(viewer=args.viewer)
    try:
        bus.torque(True)
        run(bus, sess, default_pos, action_scale, args.mode, args.duration,
            1.0, 0.0, gait_freq_hz, 5.0, 12.0, None, False, args.forward_speed,
            heading_dim, 0.0, gait_transition_time_s, None,
            vel_ramp, args.yaw_rate, yaw_ramp, None, 0.0, height_ramp, 0.0, 0.06, ch5)
    except UnsafeTiltError as exc:
        print(f"SAFETY STOP: {exc}")
    finally:
        bus.torque(False)
        bus.close()
        sess.report()


if __name__ == "__main__":
    main()
