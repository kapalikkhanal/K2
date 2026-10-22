#!/usr/bin/env python3
"""Compile the production policy/control headers and compare with the ONNX.

Run from any directory: python esp32-s3/tests/verify.py
Requires g++, numpy, onnx, onnxruntime. No serial device or servo movement.
"""
import ctypes
import argparse
import math
from pathlib import Path
import subprocess
import tempfile
import sys

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
ESP = ROOT / "esp32-s3"
sys.path.insert(0, str(ROOT))
from hardware.k2_command_profile import step_profile, RATE, JERK, STOP_SPEED, STOP_RATE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=ROOT / 'policies/last_policy.onnx')
    args = parser.parse_args()
    policy = args.policy.resolve()
    session = ort.InferenceSession(str(policy))
    md = session.get_modelmeta().custom_metadata_map
    default = np.array(md["default_joint_pos"].split(","), np.float32)
    rng = np.random.default_rng(3400)
    pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        libfile = temp / "policy.so"
        subprocess.run(["g++", "-std=c++11", "-O3", "-shared", "-fPIC",
                        "-I" + str(ESP / "tests"), "-I" + str(ESP / "src"),
                        str(ESP / "tests/native.cpp"), "-o", str(libfile)], check=True)
        lib = ctypes.CDLL(str(libfile))
        lib.policy_infer.argtypes = [pointer, pointer]
        lib.policy_obs.argtypes = [pointer, pointer]
        lib.regression_tests()
        print("PASS: stick activation, dead-man, centered/disconnected pad, clock rollover,")
        print("      joint-limit reversal, serial echo/checksum/error/timeout")
        for flash in (1, 0):
            lib.policy_init(flash)
            worst = 0
            for _ in range(256):
                obs = rng.uniform(-1, 1, 50).astype(np.float32)
                out = np.empty(12, np.float32)
                lib.policy_infer(obs, out)
                ref = session.run(None, {"obs": obs[None]})[0][0]
                worst = max(worst, float(np.max(np.abs(out - ref))))
            assert worst < 1e-4, worst
            print(f"PASS: {'flash' if flash else 'SRAM'} inference, 256 ONNX probes, max error {worst:.3g}")

        # Stateful command transitions, independent Python observation formula
        # used in hardware/k2_policy_run.py, including hold/turn/stop/restart.
        lib.policy_reset()
        m = np.float32(0)
        phase = head = 0.0
        value = np.zeros((1,3), np.float32)
        rate = np.zeros_like(value)
        phases = np.zeros(1, np.float32)
        march = np.zeros(1, np.float32)
        settling = np.zeros(1, bool)
        worst = 0
        for tick in range(1000):
            moving = 30 <= tick < 350 or 500 <= tick < 900
            vx, wz, dh, heading = 0.03, -0.1, (-.025 if tick < 600 else .025), 0.2 * math.sin(tick/100)
            ingredients = np.concatenate(([0.1, -0.2, 0.05, 0, 0, -1], default,
                                          np.zeros(12), [vx, wz, dh, moving, heading])).astype(np.float32)
            actual = np.empty(50, np.float32)
            lib.policy_obs(ingredients, actual)
            value, rate, phases, march, settling = step_profile(
                np, value, rate, phases, march, settling,
                np.array([[vx,dh,wz]],np.float32), np.array([moving]),
                .02, float(md['gait_frequency_hz']), float(md['gait_transition_time_s']),
                *[np.array(v,np.float32) for v in (RATE,JERK,STOP_SPEED,STOP_RATE)])
            m, phase = march[0], phases[0]
            head = (head + float(m) * value[0,2] * 0.02 + math.pi) % (2*math.pi) - math.pi
            expected = np.concatenate((ingredients[:6], np.zeros(36),
                                       [math.sin(heading-head), math.cos(heading-head),
                                        m, m*math.sin(phase), m*math.cos(phase),
                                        m*value[0,0], m*value[0,2], m*value[0,1]])).astype(np.float32)
            worst = max(worst, float(np.max(np.abs(actual-expected))))
        assert worst < 3e-4, worst
        print(f"PASS: 1000 observation ticks (hold/walk/turn/stop/restart), max error {worst:.3g}")

        # Verify checked-in calibration against the current Pi files and ONNX.
        config = temp / "k2_config.h"
        subprocess.run([sys.executable, str(ESP / "tools/export_k2_config.py"),
                        str(policy), str(config)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        assert config.read_text() == (ESP / "src/k2_config.h").read_text()
        print("PASS: generated mapping, limits, level correction and policy metadata match Pi files")


if __name__ == "__main__":
    main()
