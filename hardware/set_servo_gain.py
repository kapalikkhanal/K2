#!/usr/bin/env python3
"""Set the STS3215 internal position-loop P (and D) gain on all servos.

The servos ship at P=32, which is softer than the sim's kp=20 (the squat run
measured 9.5 deg real tracking error vs 2.8 deg in sim). Under the dynamic
walking load the soft servos sag and the robot pitches forward. Raising P
stiffens them toward the trained stiffness.

    python -m hardware.set_servo_gain --p 48         # try 48, then 64 if needed
    python -m hardware.set_servo_gain --p 32 --d 32  # restore defaults

Raise gradually and watch for buzzing/oscillation (too-high P). P is in the
EPROM, so this unlocks, writes, and re-locks; do it rarely, not every run.
"""

from __future__ import annotations

import argparse
import sys
import time

import scservo_sdk as scs

from . import k2_conventions as C

ADDR_P, ADDR_D, ADDR_I = 21, 22, 23
ADDR_LOCK = 55


def _read_checked(pk, ph, sid, address, label):
    for _ in range(3):
        ph.clearPort()
        value, result, error = pk.read1ByteTxRx(ph, sid, address)
        if result == scs.COMM_SUCCESS and error == 0:
            return value
        time.sleep(0.03)
    raise RuntimeError(
        f"id {sid}: cannot read {label}: "
        f"{pk.getTxRxResult(result)} {pk.getRxPacketError(error)}"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--p", type=int, required=True, help="position P gain (0-254)")
    ap.add_argument("--d", type=int, default=None, help="position D gain")
    args = ap.parse_args(argv)
    if not 0 <= args.p <= 254:
        raise SystemExit("P must be 0-254")
    if args.d is not None and not 0 <= args.d <= 254:
        raise SystemExit("D must be 0-254")

    ph = scs.PortHandler(args.port)
    pk = scs.PacketHandler(0)
    if not (ph.openPort() and ph.setBaudRate(1_000_000)):
        raise SystemExit(f"cannot open {args.port}")

    try:
        for sid in C.IDS:
            # Writes may not return status at the servo's configured response
            # level. Verify from clean reads after the EEPROM is relocked.
            pk.write1ByteTxRx(ph, sid, ADDR_LOCK, 0)
            time.sleep(0.02)
            pk.write1ByteTxRx(ph, sid, ADDR_P, args.p)
            if args.d is not None:
                time.sleep(0.02)
                pk.write1ByteTxRx(ph, sid, ADDR_D, args.d)
            time.sleep(0.02)
            pk.write1ByteTxRx(ph, sid, ADDR_LOCK, 1)
            time.sleep(0.05)
            p = _read_checked(pk, ph, sid, ADDR_P, "P")
            d = _read_checked(pk, ph, sid, ADDR_D, "D")
            if p != args.p or (args.d is not None and d != args.d):
                raise RuntimeError(
                    f"id {sid}: verification failed: read P={p} D={d}"
                )
            print(f"  id {sid}: P={p} D={d} verified")
    finally:
        ph.closePort()
    print(f"verified P={args.p} on all servos" +
          (f", D={args.d}" if args.d is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
