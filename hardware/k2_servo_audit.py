#!/usr/bin/env python3
"""Read-only configuration and health audit for all K2 STS3215 servos.

This command never enables torque and never writes a servo register.  Run it
on the Pi before system identification so the digital twin can be based on the
actual position-loop settings rather than assumed defaults.

    python -m hardware.k2_servo_audit --port /dev/ttyAMA0
    python -m hardware.k2_servo_audit --port /dev/ttyAMA0 --json servo_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import k2_conventions as C
from .sts_common import open_port


REGISTERS = {
    "return_delay": (7, 1),
    "status_return_level": (8, 1),
    "min_angle_limit": (9, 2),
    "max_angle_limit": (11, 2),
    "max_temperature_c": (13, 1),
    "max_voltage_raw": (14, 1),
    "min_voltage_raw": (15, 1),
    "max_torque_raw": (16, 2),
    "position_p_gain": (21, 1),
    "position_d_gain": (22, 1),
    "position_i_gain": (23, 1),
    "cw_deadband": (26, 1),
    "ccw_deadband": (27, 1),
    "overload_current_raw": (28, 2),
    "angular_resolution": (30, 1),
    "homing_offset_raw": (31, 2),
    "mode": (33, 1),
    "protection_torque": (34, 1),
    "protection_time": (35, 1),
    "overload_torque": (36, 1),
    "torque_enabled": (40, 1),
    "acceleration": (41, 1),
    "goal_position": (42, 2),
    "goal_time": (44, 2),
    "goal_speed": (46, 2),
    "torque_limit_raw": (48, 2),
    "locked": (55, 1),
    "present_position": (56, 2),
    "present_speed_raw": (58, 2),
    "present_load_raw": (60, 2),
    "present_voltage_raw": (62, 1),
    "present_temperature_c": (63, 1),
    "hardware_error_status": (65, 1),
    "moving": (66, 1),
    "present_current_raw": (69, 2),
}


def _read(pk, ph, sid: int, address: int, size: int) -> int:
    for _ in range(3):
        ph.clearPort()
        if size == 1:
            value, result, error = pk.read1ByteTxRx(ph, sid, address)
        else:
            value, result, error = pk.read2ByteTxRx(ph, sid, address)
        if result == 0 and error == 0:
            return int(value)
        time.sleep(0.01)
    raise RuntimeError(
        f"id {sid} register {address}: {pk.getTxRxResult(result)} "
        f"{pk.getRxPacketError(error)}"
    )


def _derived(raw: dict[str, int]) -> dict[str, float | int]:
    load = raw["present_load_raw"]
    speed = raw["present_speed_raw"]
    return {
        "present_voltage_v": raw["present_voltage_raw"] * 0.1,
        "min_voltage_v": raw["min_voltage_raw"] * 0.1,
        "max_voltage_v": raw["max_voltage_raw"] * 0.1,
        "present_speed_steps_s": C.decode_speed(speed),
        "present_load_signed": -(load & 0x3ff) if load & 0x400 else load & 0x3ff,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyAMA0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--json", dest="json_path", help="also write full audit JSON")
    args = ap.parse_args(argv)

    ph, pk = open_port(args.port, args.baud)
    report = {
        "port": args.port,
        "baud": args.baud,
        "servos": {},
    }
    try:
        for sid in C.IDS:
            raw = {
                name: _read(pk, ph, sid, address, size)
                for name, (address, size) in REGISTERS.items()
            }
            values = {**raw, **_derived(raw)}
            report["servos"][str(sid)] = values
            print(
                f"id {sid:2d}  P/D/I={raw['position_p_gain']:3d}/"
                f"{raw['position_d_gain']:3d}/{raw['position_i_gain']:3d}  "
                f"mode={raw['mode']}  acc={raw['acceleration']:3d}  "
                f"speed={raw['goal_speed']:4d}  "
                f"enabled={raw['torque_enabled']}  "
                f"limit={raw['torque_limit_raw']:4d}  "
                f"V={values['present_voltage_v']:4.1f}  "
                f"T={raw['present_temperature_c']:2d}C  "
                f"load={values['present_load_signed']:+5d}  "
                f"err=0x{raw['hardware_error_status']:02x}"
            )
    finally:
        ph.closePort()

    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
