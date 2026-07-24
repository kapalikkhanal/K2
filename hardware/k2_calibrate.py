#!/usr/bin/env python3
"""Calibration GUI for K2 (12 STS3215 servos).

Captures the mapping between servo counts and MJCF joint angles:

    q_sim = sign * (raw - home_raw) * 2*pi/4096

Writes hardware/calibration.json next to this file -- which is the only path
k2_ctrl reads. A calibration saved under Robot_v2_single_leg is NOT picked up.

Everything is taken from k2_conventions, which reads the MJCF: joint limits,
nudge directions and the descriptions used to confirm each sign. Nothing is
hardcoded here, so this cannot drift out of step with the model the way the
single-leg tool did.

    DISPLAY=:1 python -m hardware.k2_calibrate

Procedure: connect -> LIMP -> hand-pose the robot dead straight -> Capture
HOME -> Nudge each joint and Flip any whose motion does not match the
description -> Save -> Go to CROUCH to validate.
"""

from __future__ import annotations

import glob
import sys
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox

import numpy as np

from . import k2_conventions as C
from .k2_bus import SerialBus

NUDGE_RAD = 0.20          # ~11.5 deg, big enough to see, small enough to be safe
WRAP_MARGIN = 400         # counts from 0/4095 before we warn about the wrap


class CalibApp:
    def __init__(self, root):
        self.root = root
        root.title("K2 - Calibration (12 servos)")
        root.geometry("1180x900")
        self.bus = None
        self.calib = C.load_calibration() or C.default_calibration()
        self.crouch = None

        default = tkfont.nametofont("TkDefaultFont")
        default.configure(size=12)
        root.option_add("*Font", default)
        self.mono = tkfont.Font(family="TkFixedFont", size=12)

        self._build_toolbar()
        self._build_table()
        self._build_log()
        self.log("Connect, then: LIMP + hand-pose straight -> Capture HOME "
                 "-> Nudge/Flip each sign -> Save.")
        self.log(f"Calibration will be written to {C.CALIB_PATH}")
        self._poll()

    # ---------------- layout ----------------
    def _build_toolbar(self):
        bar = tk.Frame(self.root)
        bar.pack(fill="x", padx=4, pady=4)
        tk.Label(bar, text="Port").pack(side="left")
        ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
        self.port = tk.StringVar(value=ports[0] if ports else "/dev/ttyACM0")
        tk.OptionMenu(bar, self.port, *(ports or ["/dev/ttyACM0"])).pack(side="left")
        for text, cmd in (("Connect", self.connect),
                          ("Disconnect", self.disconnect),
                          ("LIMP (torque off)", self.limp),
                          ("Hold HOME", self.hold_home),
                          ("Set MIDDLE (2048)", self.set_middle)):
            tk.Button(bar, text=text, command=cmd).pack(side="left", padx=2)

        bar2 = tk.Frame(self.root)
        bar2.pack(fill="x", padx=4)
        for text, cmd in (("1  Capture HOME (straight, LIMP)", self.capture_home),
                          ("2  Save calibration", self.save),
                          ("3  Go to CROUCH (validate)", self.go_crouch)):
            tk.Button(bar2, text=text, command=cmd).pack(side="left", padx=2)

    def _build_table(self):
        tk.Label(self.root, text="Joints (right leg, then left leg)",
                 anchor="w").pack(fill="x", padx=6, pady=(8, 0))
        grid = tk.Frame(self.root)
        grid.pack(fill="x", padx=6)
        heads = ("joint", "id", "raw", "q (rad)",
                 "a NUDGE should make it...", "test", "sign", "")
        for c, h in enumerate(heads):
            tk.Label(grid, text=h, font=self.mono, anchor="w").grid(
                row=0, column=c, sticky="w", padx=3)

        self.rows = {}
        for r, j in enumerate(C.SIM_ORDER, start=1):
            lo, hi = C.LIMITS[j]
            raw_v, q_v, sign_v = tk.StringVar(), tk.StringVar(), tk.StringVar()
            tk.Label(grid, text=j, font=self.mono, anchor="w").grid(
                row=r, column=0, sticky="w", padx=3)
            tk.Label(grid, text=str(C.JOINT_TO_ID[j]), font=self.mono).grid(
                row=r, column=1, padx=3)
            tk.Label(grid, textvariable=raw_v, font=self.mono, width=6).grid(
                row=r, column=2, padx=3)
            tk.Label(grid, textvariable=q_v, font=self.mono, width=8).grid(
                row=r, column=3, padx=3)
            arrow = "-" if C.NUDGE_DIR[j] < 0 else "+"
            tk.Label(grid, text=f"{C.POS_DESC[j]}", font=self.mono,
                     anchor="w").grid(row=r, column=4, sticky="w", padx=3)
            tk.Button(grid, text=f"Nudge {arrow}",
                      command=lambda j=j: self.nudge(j)).grid(row=r, column=5, padx=3)
            tk.Label(grid, textvariable=sign_v, font=self.mono, width=4).grid(
                row=r, column=6, padx=3)
            tk.Button(grid, text="Flip",
                      command=lambda j=j: self.flip(j)).grid(row=r, column=7, padx=3)
            self.rows[j] = (raw_v, q_v, sign_v)
            sign_v.set(f"{self.calib['sign'][j]:+d}")
            _ = (lo, hi)

    def _build_log(self):
        tk.Label(self.root, text="Log", anchor="w").pack(fill="x", padx=6,
                                                         pady=(8, 0))
        self.logbox = tk.Text(self.root, height=12, font=self.mono)
        self.logbox.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    def log(self, msg):
        self.logbox.insert("end", msg + "\n")
        self.logbox.see("end")

    # ---------------- bus ----------------
    def connect(self):
        if self.bus:
            return
        try:
            self.bus = SerialBus(self.port.get())
        except Exception as e:
            messagebox.showerror("connect failed", str(e))
            return
        pos, _ = self.bus.read_pos_speed()
        self.log(f"connected {self.port.get()} ({len(pos)}/12 servos)")
        if len(pos) < 12:
            missing = [i for i in C.IDS if i not in pos]
            self.log(f"  WARNING: no reply from servo IDs {missing}")

    def disconnect(self):
        if self.bus:
            self.bus.close()
            self.bus = None
            self.log("disconnected")

    def limp(self):
        if self.bus:
            self.bus.torque(False)
            self.log("LIMP - hand-pose the robot now.")

    def hold_home(self):
        if not self.bus:
            return
        self.bus.torque(True)
        self.bus.write_goals(C.goal_counts(np.zeros(len(C.SIM_ORDER)), self.calib))
        self.log("holding HOME (all q = 0)")

    def set_middle(self):
        """Drive every servo to 2048 so a bracket can be remounted straight."""
        if not self.bus:
            return
        if not messagebox.askokcancel(
                "Set MIDDLE", "Drive ALL servos to 2048?\n\n"
                "Only do this with the linkages DISCONNECTED - it is the "
                "re-homing step, not a pose."):
            return
        self.bus.torque(True)
        self.bus.write_goals({i: 2048 for i in C.IDS})
        self.log("all servos -> 2048 (remount brackets at the straight pose, "
                 "then Capture HOME)")

    # ---------------- calibration ----------------
    def capture_home(self):
        if not self.bus:
            return
        pos, _ = self.bus.read_pos_speed()
        if len(pos) < 12:
            messagebox.showerror("capture failed",
                                 f"only {len(pos)}/12 servos replied")
            return
        for j in C.SIM_ORDER:
            self.calib["home_raw"][j] = int(pos[C.JOINT_TO_ID[j]])
        self.log(f"captured HOME: {self.calib['home_raw']}")

        # A home near 0/4095 means the working range crosses the single-turn
        # wrap; the servo would take the long way round. Re-home instead.
        risky = [j for j in C.SIM_ORDER
                 if not (WRAP_MARGIN < self.calib["home_raw"][j] < 4096 - WRAP_MARGIN)]
        if risky:
            self.log(f"  WARNING: {risky} sit near the encoder wrap. "
                     f"Re-home them (Set MIDDLE, remount straight).")
        self._check_reach()

    def _check_reach(self):
        """Warn if a joint's full sim range would run past the encoder ends."""
        for j in C.SIM_ORDER:
            lo, hi = C.LIMITS[j]
            ends = [self.calib["home_raw"][j]
                    + self.calib["sign"][j] * q * C.COUNTS_PER_RAD for q in (lo, hi)]
            if min(ends) < C.COUNT_MIN or max(ends) > C.COUNT_MAX:
                self.log(f"  note: {j} range needs counts "
                         f"[{min(ends):.0f},{max(ends):.0f}], outside "
                         f"[{C.COUNT_MIN},{C.COUNT_MAX}] - it will saturate.")

    def nudge(self, j):
        if not self.bus:
            return
        q = C.clamp_q(j, C.NUDGE_DIR[j] * NUDGE_RAD)
        self.bus.torque(True)
        self.bus.write_goals({C.JOINT_TO_ID[j]: C.q_to_raw(j, q, self.calib)})
        self.log(f"nudge {j} to q={q:+.2f} rad -> should be: {C.POS_DESC[j]}")
        self.log("   if it did the OPPOSITE, press Flip.")

    def flip(self, j):
        self.calib["sign"][j] *= -1
        self.rows[j][2].set(f"{self.calib['sign'][j]:+d}")
        self.log(f"{j} sign -> {self.calib['sign'][j]:+d}")

    def save(self):
        C.save_calibration(self.calib)
        self.log(f"saved {C.CALIB_PATH}")

    def go_crouch(self):
        """Validate against a squat pose solved from the model itself."""
        if not self.bus:
            return
        if self.crouch is None:
            from .k2_motion import SquatIK
            self.log("solving crouch pose from the MJCF...")
            self.crouch, _ = SquatIK().solve(35.0)
        self.bus.torque(True)
        self.bus.write_goals(C.goal_counts(self.crouch, self.calib))
        self.log("CROUCH (35 deg knee). Both knees should bend the SAME way, "
                 "heels down, robot squatting evenly.")

    # ---------------- polling ----------------
    def _poll(self):
        if self.bus:
            try:
                pos, _ = self.bus.read_pos_speed()
                for j in C.SIM_ORDER:
                    sid = C.JOINT_TO_ID[j]
                    if sid in pos:
                        raw_v, q_v, _ = self.rows[j]
                        raw_v.set(str(pos[sid]))
                        q_v.set(f"{C.raw_to_q(j, pos[sid], self.calib):+.3f}")
            except Exception as e:
                self.log(f"read error: {e}")
        self.root.after(100, self._poll)


def main():
    root = tk.Tk()
    app = CalibApp(root)
    try:
        root.mainloop()
    finally:
        if app.bus:
            app.bus.torque(False)
            app.bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
