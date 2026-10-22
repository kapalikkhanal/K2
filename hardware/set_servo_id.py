#!/usr/bin/env python3
"""Assign a servo's ID on the STS3215 bus -- used to give the two v4 ankle-roll
servos their IDs (11 right, 12 left) without disturbing the existing 1-10 chain.

The ID lives in the servo's EEPROM (register 5). Writing it means unlocking the
EEPROM (register 55 -> 0), writing the new ID, then locking it again. A factory
STS3215 ships as ID 1, so a brand-new servo must be addressed as 1.

    # See which IDs currently answer on the bus:
    python -m hardware.set_servo_id --port /dev/ttyACM0 --scan

    # Change one servo from its current ID to a new one:
    python -m hardware.set_servo_id --port /dev/ttyACM0 --from 1 --to 11

SAFETY: connect ONLY the servo you are renumbering (or be certain no other
servo already holds either the --from or the --to ID). If two servos share an
ID they both reply at once and the bus is corrupted. The tool refuses to write
--to if a *different* servo already answers on it.
"""

from __future__ import annotations

import argparse
import sys
import time

from . import k2_conventions as C


def _open(port: str, baud: int = 1_000_000):
    import scservo_sdk as scs

    ph = scs.PortHandler(port)
    pk = scs.PacketHandler(0)
    if not ph.openPort():
        raise SystemExit(
            f"open {port} failed. If it is a permission error:\n"
            f"  sudo chmod 666 {port}   (resets on every replug)")
    if not ph.setBaudRate(baud):
        raise SystemExit("setBaudRate failed")
    return scs, ph, pk


def scan(scs, ph, pk, lo: int = 0, hi: int = 20) -> list[int]:
    found = []
    for i in range(lo, hi + 1):
        _, comm, err = pk.ping(ph, i)
        if comm == scs.COMM_SUCCESS and err == 0:
            found.append(i)
    return found


def set_id(scs, ph, pk, old: int, new: int) -> None:
    # Refuse if the target ID is already taken by a *different* servo.
    _, comm, _ = pk.ping(ph, new)
    if comm == scs.COMM_SUCCESS and new != old:
        raise SystemExit(
            f"ID {new} is already in use on this bus. Disconnect that servo, "
            f"or pick a free ID. (Currently present: {scan(scs, ph, pk)})")
    _, comm, _ = pk.ping(ph, old)
    if comm != scs.COMM_SUCCESS:
        raise SystemExit(f"no servo answers on ID {old}. "
                         f"Present: {scan(scs, ph, pk)}")

    # Unlock EEPROM -> write ID -> lock EEPROM.
    pk.write1ByteTxRx(ph, old, C.ADDR_LOCK, 0)
    time.sleep(0.02)
    _, comm, err = pk.write1ByteTxRx(ph, old, C.ADDR_ID, new)
    time.sleep(0.02)
    if comm != scs.COMM_SUCCESS or err != 0:
        raise SystemExit(f"writing ID failed (comm={comm}, err={err})")
    # The servo now answers on the NEW id; lock its EEPROM there.
    pk.write1ByteTxRx(ph, new, C.ADDR_LOCK, 1)
    time.sleep(0.02)

    _, comm, _ = pk.ping(ph, new)
    if comm != scs.COMM_SUCCESS:
        raise SystemExit(f"servo did not come back on ID {new} after the write")
    print(f"OK: servo {old} -> {new}. Present now: {scan(scs, ph, pk)}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--scan", action="store_true", help="list IDs on the bus and exit")
    ap.add_argument("--from", dest="old", type=int, help="current servo ID")
    ap.add_argument("--to", dest="new", type=int, help="new servo ID")
    args = ap.parse_args(argv)

    scs, ph, pk = _open(args.port)
    try:
        if args.scan:
            print("IDs present:", scan(scs, ph, pk))
            # Name the ones we recognise from the K2 map.
            id_to_joint = {v: k for k, v in C.JOINT_TO_ID.items()}
            for i in scan(scs, ph, pk):
                print(f"  {i:2d} -> {id_to_joint.get(i, '(unassigned in K2 map)')}")
            return 0
        if args.old is None or args.new is None:
            ap.error("give --scan, or both --from and --to")
        set_id(scs, ph, pk, args.old, args.new)
    finally:
        ph.closePort()
    return 0


if __name__ == "__main__":
    sys.exit(main())
