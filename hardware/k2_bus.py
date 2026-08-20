"""The sim/real swap point for K2.

Both backends expose the same interface, and that interface speaks the real
robot's own language: raw STS3215 encoder counts. Everything above this file
-- calibration, count math, observation building, ONNX inference, safety
clamping -- is therefore identical code whether it drives MuJoCo or hardware,
which is what makes the sim a real digital twin rather than a lookalike.

    Bus.tick(dt)          advance one control period (sim: step physics,
                          real: sleep to hold the rate)
    Bus.read_pos_speed()  -> ({id: counts}, {id: counts/s})
    Bus.write_goals()     {id: goal counts}
    Bus.torque(on)

SimBus deliberately reproduces the parts of the servo that shape what the
policy sees: 12-bit encoder quantisation, the [40, 4055] goal clamp, and an
optional read latency. Fit `kp`, `kd` and `latency` from the system-ID run;
the defaults below are placeholders carried over from the single leg.
"""

from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path

import numpy as np

# MuJoCo is only needed by SimBus. On the robot (Pi) it is not installed and
# only SerialBus runs, so import it lazily rather than at module load.
try:
    import mujoco
except ImportError:
    mujoco = None

from . import k2_conventions as C

STALL_TORQUE = 2.94  # N.m

# Placeholders until phase A system ID measures them on the real servos.
DEFAULT_KP = 20.0
DEFAULT_KD = 0.5
# Measured on the 10-servo v3 chain: a sync read took 1.70 ms, well under one
# 20 ms control tick at 50 Hz, so the round trip costs less than a full tick of
# delay. The two v4 servos add ~0.3 ms, still comfortably within a tick. Left
# at 0 until system ID measures the servo's own response lag, which is the part
# that actually matters.
DEFAULT_LATENCY_TICKS = 0


class Bus(ABC):
    """Servo bus. Talks in raw encoder counts, exactly like the hardware."""

    ids: list[int]

    @abstractmethod
    def torque(self, on: bool) -> None: ...

    @abstractmethod
    def read_pos_speed(self) -> tuple[dict[int, int], dict[int, int]]: ...

    @abstractmethod
    def write_goals(self, id_to_raw: dict[int, int]) -> None: ...

    @abstractmethod
    def tick(self, dt: float) -> None: ...

    def latest_telemetry(self) -> dict[int, dict[str, int | float]] | None:
        """Most recent per-servo telemetry, or None when unsupported."""
        return None

    def base_position(self) -> np.ndarray | None:
        """World xyz when the backend has ground-truth position (simulation)."""
        return None

    def base_heading(self) -> float | None:
        """World yaw in radians when available (simulation only)."""
        return None

    def close(self) -> None: ...

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SerialBus(Bus):
    """The real STS3215 chain on a Waveshare adapter."""

    def __init__(self, port: str = "/dev/ttyAMA0", baud: int = 1_000_000,
                 ids: list[int] | None = None):
        import scservo_sdk as scs

        self.scs = scs
        self.ids = list(ids) if ids is not None else list(C.IDS)
        self.port = port
        self.baud = baud
        self.ph = scs.PortHandler(port)
        self.pk = scs.PacketHandler(0)
        # PortHandler.openPort() already configures its default 1 Mbps baud;
        # calling setBaudRate() immediately afterwards closes and reopens the
        # UART unnecessarily.  That second open can fail after a partial read.
        if not self.ph.setBaudRate(baud):
            raise RuntimeError(
                f"open/configure {port} failed. If it is a permission error: "
                f"sudo chmod 666 {port}  (it resets on every replug)")
        # One coherent block covers position through present current (56..70).
        # This adds dynamic voltage/load/current/temperature observability while
        # preserving one sync-read transaction per control tick.
        self._reader = scs.GroupSyncRead(self.ph, self.pk, C.ADDR_PRESENT_POS, 15)
        for i in self.ids:
            self._reader.addParam(i)
        self._next = None
        self._telemetry = None

    def torque(self, on: bool) -> None:
        target = 1 if on else 0
        pending = set(self.ids)
        last_errors = {}
        # A safety-stop happens while the bus and loaded servos are busiest.
        # A single unchecked write can be dropped, leaving every geared joint
        # actively holding the last gait target.  Retry and verify the actual
        # register state before returning.
        for _ in range(5):
            try:
                for sid in list(pending):
                    self.ph.clearPort()
                    result, error = self.pk.write1ByteTxRx(
                        self.ph, sid, C.ADDR_TORQUE, target
                    )
                    last_errors[sid] = (result, error)
                    time.sleep(0.002)
                time.sleep(0.01)
                for sid in list(pending):
                    self.ph.clearPort()
                    value, result, error = self.pk.read1ByteTxRx(
                        self.ph, sid, C.ADDR_TORQUE
                    )
                    last_errors[sid] = (result, error)
                    if (result == self.scs.COMM_SUCCESS and error == 0
                            and value == target):
                        pending.remove(sid)
            except Exception as exc:
                for sid in pending:
                    last_errors[sid] = exc
                # PySerial can leave the descriptor unusable after a partial
                # UART read. Reopen it before the next safety-write attempt.
                try:
                    self.ph.closePort()
                except Exception:
                    pass
                time.sleep(0.02)
                try:
                    self.ph.setBaudRate(self.baud)
                except Exception as reopen_exc:
                    for sid in pending:
                        last_errors[sid] = reopen_exc
            if not pending:
                return
        details_parts = []
        for sid in sorted(pending):
            failure = last_errors.get(sid, "no response")
            if isinstance(failure, tuple):
                failure = (f"{self.pk.getTxRxResult(failure[0])} "
                           f"{self.pk.getRxPacketError(failure[1])}")
            details_parts.append(f"id {sid}: {failure}")
        details = ", ".join(details_parts)
        raise RuntimeError(
            f"failed to verify torque={'on' if on else 'off'} for "
            f"IDs {sorted(pending)} ({details})"
        )

    def read_pos_speed(self):
        # A sync read can occasionally omit one response. Never pass a partial
        # dictionary upward: q_vector used to turn an omitted joint into zero,
        # which looked like a one-tick 20-degree encoder excursion. Retry once;
        # if the chain is still incomplete, fail so the caller's finally block
        # releases torque.
        missing_pos = missing_spd = []
        for attempt in range(5):
            self._reader.txRxPacket()
            pos, spd = {}, {}
            for i in self.ids:
                if self._reader.isAvailable(i, C.ADDR_PRESENT_POS, 2):
                    pos[i] = self._reader.getData(i, C.ADDR_PRESENT_POS, 2)
                if self._reader.isAvailable(i, C.ADDR_PRESENT_SPEED, 2):
                    spd[i] = C.decode_speed(
                        self._reader.getData(i, C.ADDR_PRESENT_SPEED, 2))
            missing_pos = [i for i in self.ids if i not in pos]
            missing_spd = [i for i in self.ids if i not in spd]
            if not missing_pos and not missing_spd:
                telemetry = {}
                for i in self.ids:
                    def get(address, size):
                        return int(self._reader.getData(i, address, size))
                    load_raw = get(60, 2)
                    temperature_raw = get(63, 1)
                    previous = self._telemetry.get(i) if self._telemetry else None
                    # A servo cannot change tens of degrees in 20 ms. Rare
                    # otherwise-valid packets contained one corrupt byte
                    # (78/95 C, immediately followed by a stable 36 C).
                    temperature_outlier = int(
                        previous is not None
                        and abs(temperature_raw - previous["temperature_c"]) > 10
                    )
                    temperature = (
                        previous["temperature_c"]
                        if temperature_outlier else temperature_raw
                    )
                    telemetry[i] = {
                        "load_raw": load_raw,
                        "load_signed": (
                            -(load_raw & 0x3ff)
                            if load_raw & 0x400 else load_raw & 0x3ff
                        ),
                        "voltage_v": get(62, 1) * 0.1,
                        "temperature_c": temperature,
                        "temperature_outlier": temperature_outlier,
                        "error_status": get(65, 1),
                        "moving": get(66, 1),
                        "current_raw": get(69, 2),
                    }
                self._telemetry = telemetry
                return pos, spd
            # Loaded servos can occasionally stretch one UART response. Give
            # the chain 1 ms to recover before retrying, still well inside the
            # 20 ms control period.
            time.sleep(0.001)
        raise RuntimeError(
            f"incomplete servo sync read after 5 retries: "
            f"missing position IDs {missing_pos}, speed IDs {missing_spd}")

    def latest_telemetry(self):
        return self._telemetry

    def write_goals(self, id_to_raw: dict[int, int]) -> None:
        gw = self.scs.GroupSyncWrite(self.ph, self.pk, C.ADDR_GOAL_POS, 2)
        for i, raw in id_to_raw.items():
            raw = int(max(C.COUNT_MIN, min(C.COUNT_MAX, raw)))
            gw.addParam(i, [self.scs.SCS_LOBYTE(raw), self.scs.SCS_HIBYTE(raw)])
        gw.txPacket()
        gw.clearParam()

    def tick(self, dt: float) -> None:
        """Hold the control rate against the wall clock."""
        now = time.perf_counter()
        if self._next is None:
            self._next = now
        self._next += dt
        slack = self._next - now
        if slack > 0:
            time.sleep(slack)
        else:
            self._next = now  # overran; resync rather than spiral

    def close(self) -> None:
        try:
            self.ph.closePort()
        except Exception:
            pass


class SimBus(Bus):
    """MuJoCo standing in for the servo chain, count-for-count."""

    def __init__(self, kp: float = DEFAULT_KP, kd: float = DEFAULT_KD,
                 latency_ticks: int = DEFAULT_LATENCY_TICKS,
                 viewer: bool = False, realtime: bool = True,
                 ids: list[int] | None = None, key_callback=None):
        if mujoco is None:
            raise RuntimeError("SimBus needs MuJoCo, which is not installed. "
                               "On the robot use SerialBus (--bus /dev/ttyACM0).")
        self.ids = list(ids) if ids is not None else list(C.IDS)
        self.model = _actuated_model(kp, kd)
        self.data = mujoco.MjData(self.model)
        self.calib = C.default_calibration()   # sim is perfectly calibrated
        self.realtime = realtime
        self._torque_on = True

        self._qadr, self._dadr, self._act = {}, {}, {}
        for joint, mjname in C.JOINT_TO_MJCF.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, mjname)
            self._qadr[joint] = self.model.jnt_qposadr[jid]
            self._dadr[joint] = self.model.jnt_dofadr[jid]
            self._act[joint] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, mjname)

        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self._delay = deque(maxlen=max(1, latency_ticks + 1))
        self._next = None

        self.viewer = None
        if viewer:
            # `import mujoco.viewer` here would rebind `mujoco` as a local.
            import mujoco.viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(
                self.model, self.data, key_callback=key_callback
            )

    # -- the servo's own view of itself, in counts ---------------------------
    def read_pos_speed(self):
        pos, spd = {}, {}
        for joint in C.SIM_ORDER:
            sid = C.JOINT_TO_ID[joint]
            q = float(self.data.qpos[self._qadr[joint]])
            qd = float(self.data.qvel[self._dadr[joint]])
            # Quantise to the 12-bit encoder, so the policy sees the same
            # resolution it will see on hardware.
            pos[sid] = C.q_to_raw(joint, q, self.calib)
            spd[sid] = int(round(qd * C.COUNTS_PER_RAD))
        self._delay.append((pos, spd))
        return self._delay[0]

    def write_goals(self, id_to_raw: dict[int, int]) -> None:
        if not self._torque_on:
            return
        id_to_joint = {v: k for k, v in C.JOINT_TO_ID.items()}
        for sid, raw in id_to_raw.items():
            joint = id_to_joint[sid]
            raw = int(max(C.COUNT_MIN, min(C.COUNT_MAX, raw)))
            self.data.ctrl[self._act[joint]] = C.raw_to_q(joint, raw, self.calib)

    def torque(self, on: bool) -> None:
        self._torque_on = on
        if not on:
            for joint in C.SIM_ORDER:
                self.model.actuator_gainprm[self._act[joint], 0] = 0.0
        else:
            mujoco.mj_forward(self.model, self.data)

    def tick(self, dt: float) -> None:
        n = max(1, int(round(dt / self.model.opt.timestep)))
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
        if self.realtime:
            # Hold wall-clock, the same way SerialBus does, so a sim run and a
            # hardware run of the same motion take the same time. Physics is
            # far faster than real time, so without this the sim just races.
            now = time.perf_counter()
            if self._next is None:
                self._next = now
            self._next += dt
            slack = self._next - now
            if slack > 0:
                time.sleep(slack)
            else:
                self._next = now

    # -- sim-only conveniences ---------------------------------------------
    def set_joints(self, q: np.ndarray) -> None:
        """Teleport the joints (for initialising a pose). Sim only."""
        for k, joint in enumerate(C.SIM_ORDER):
            self.data.qpos[self._qadr[joint]] = q[k]
            self.data.ctrl[self._act[joint]] = q[k]
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        """Reset the complete digital twin after a fall, keeping its viewer."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self._delay.clear()
        self._next = None
        self._torque_on = True
        mujoco.mj_forward(self.model, self.data)

    def base_height(self) -> float:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        return float(self.data.xpos[bid][2])

    def base_position(self) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        return self.data.xpos[bid].copy()

    def base_heading(self) -> float:
        # The free joint is first in qpos and stores quaternion wxyz.
        w, x, y, z = self.data.qpos[3:7]
        return float(math.atan2(2.0 * (w * z + x * y),
                                1.0 - 2.0 * (y * y + z * z)))

    def true_base_state(self):
        """(gyro, proj_grav_b) matching the walk env's actor observation.

        The env's base_ang_vel obs reads the imu_ang_vel GYRO sensor (IMU/site
        frame), not the base-frame angular velocity -- so return the gyro raw,
        which is exactly what the real LSM6DS3 gives with no transform.
        projected_gravity is in the k2_root frame (world-aligned).
        """
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR,
                                "imu_ang_vel")
        adr = self.model.sensor_adr[sid]
        gyro = self.data.sensordata[adr:adr + 3].astype(np.float32)
        rid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "k2_root")
        R = self.data.xmat[rid].reshape(3, 3)              # root -> world
        proj_grav_b = (R.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)
        return gyro, proj_grav_b

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()


def _actuated_model(kp: float, kd: float) -> mujoco.MjModel:
    """The physics MJCF plus one position servo per joint.

    The shipped MJCF has no actuators on purpose, so mjlab owns the actuator
    model during training. For direct control we add them here, torque-limited
    to the STS3215's real stall so the sim cannot do anything the robot can't.
    """
    tree = ET.parse(C.MJCF)
    mj = tree.getroot()
    act = ET.SubElement(mj, "actuator")
    for mjname in C.JOINT_TO_MJCF.values():
        ET.SubElement(act, "position", name=mjname, joint=mjname,
                      kp=str(kp), kv=str(kd),
                      forcerange=f"{-STALL_TORQUE} {STALL_TORQUE}")
    meshes = {p.name: p.read_bytes()
              for p in (C.ROOT / "meshes" / "K2").iterdir()
              if p.suffix.lower() in (".obj", ".stl", ".mtl")}
    return mujoco.MjModel.from_xml_string(
        ET.tostring(mj, encoding="unicode"), meshes)


def open_bus(spec: str, **kw) -> Bus:
    """'sim' -> SimBus, anything else -> SerialBus on that device path."""
    if spec == "sim":
        return SimBus(**{k: v for k, v in kw.items()
                         if k in ("kp", "kd", "latency_ticks", "viewer",
                                  "realtime", "ids", "key_callback")})
    return SerialBus(port=spec, **{k: v for k, v in kw.items()
                                   if k in ("baud", "ids")})
