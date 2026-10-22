#!/usr/bin/env python3
"""Record the joint positions the robot is CURRENTLY held in, as a pose file.

Read-only on purpose. Unlike `hardware/capture_home.py` this does NOT write
calibration.json -- that file defines where every joint's zero is, and
overwriting it to record a pose would silently redefine the whole robot.

Torque is left exactly as found: if the robot is standing under torque in the
pose being recorded, releasing it would drop the robot mid-capture.
"""
import argparse, datetime, json, time
import numpy as np
from hardware import k2_conventions as C
from hardware.k2_bus import SerialBus

ap = argparse.ArgumentParser()
ap.add_argument("--bus", default="/dev/ttyAMA0")
ap.add_argument("--samples", type=int, default=25)
ap.add_argument("--label", default="")
ap.add_argument("--out", default="pose.json")
a = ap.parse_args()

bus = SerialBus(a.bus)                    # constructor does not touch torque
rows = []
try:
    for _ in range(a.samples):
        pos, _ = bus.read_pos_speed()
        rows.append([pos[C.JOINT_TO_ID[j]] for j in C.SIM_ORDER])
        time.sleep(0.02)
finally:
    bus.close()

v = np.asarray(rows, int)
raw = np.rint(np.median(v, axis=0)).astype(int)
calib = C.load_calibration()
if calib is None:
    raise SystemExit("no calibration.json; raw counts alone are not a pose")
q = np.array([C.raw_to_q(j, int(raw[k]), calib) for k, j in enumerate(C.SIM_ORDER)])

print(f"{'joint':16} {'id':>3} {'raw':>5} {'spread':>7} {'deg':>9}   limits(deg)")
for k, j in enumerate(C.SIM_ORDER):
    lo, hi = C.LIMITS[j]
    flag = "" if lo <= q[k] <= hi else "   <-- OUTSIDE LIMIT"
    print(f"{j:16} {C.JOINT_TO_ID[j]:3d} {raw[k]:5d} "
          f"{int(v[:,k].max()-v[:,k].min()):7d} {np.degrees(q[k]):+9.2f}   "
          f"{np.degrees(lo):+6.0f}..{np.degrees(hi):+6.0f}{flag}")

out = {
    "captured": datetime.datetime.now().isoformat(timespec="seconds"),
    "label": a.label,
    "raw": {j: int(raw[k]) for k, j in enumerate(C.SIM_ORDER)},
    "q_rad": {j: float(q[k]) for k, j in enumerate(C.SIM_ORDER)},
    "q_deg": {j: float(np.degrees(q[k])) for k, j in enumerate(C.SIM_ORDER)},
    "spread_counts": {j: int(v[:, k].max() - v[:, k].min())
                      for k, j in enumerate(C.SIM_ORDER)},
}
with open(a.out, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nsaved {a.out}")
