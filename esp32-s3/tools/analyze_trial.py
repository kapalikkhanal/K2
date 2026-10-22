#!/usr/bin/env python3
"""Summarise a decoded K2 flight log for the paper.

usage: analyze_trial.py <decoded_dir>
"""
import csv, json, math, sys, collections
from pathlib import Path

JOINTS = ['hip_pitch_R','hip_roll_R','hip_yaw_R','knee_R','ankle_pitch_R','ankle_roll_R',
          'hip_pitch_L','hip_roll_L','hip_yaw_L','knee_L','ankle_pitch_L','ankle_roll_L']
SERVO_ID = [5,4,3,2,1,0,6,7,8,9,10,11]
TICK_US = 20000

d = Path(sys.argv[1])
rows = list(csv.DictReader(open(d/'control.csv')))
events = json.load(open(d/'events.json')) if (d/'events.json').exists() else []

def f(row, key):
    try:
        v = float(row[key])
        return v if math.isfinite(v) else None
    except (KeyError, ValueError, TypeError):
        return None

def pct(vals, p):
    v = sorted(vals)
    return v[min(len(v)-1, max(0, round(p*(len(v)-1))))] if v else float('nan')

def line(label, vals, unit='', scale=1.0):
    v = [x*scale for x in vals if x is not None]
    if not v:
        print(f'  {label:<26s} (no data)'); return
    print(f'  {label:<26s} p50 {pct(v,.5):8.2f}  p95 {pct(v,.95):8.2f}  max {max(v):8.2f} {unit}')

print(f'\n{"="*72}\nFILE  {d}\nRECORDS  {len(rows)} control\n{"="*72}')

# ---- sessions -------------------------------------------------------------
by_sess = collections.defaultdict(list)
for r in rows: by_sess[r['session']].append(r)
print('\n## Sessions')
total_span = 0.0
for s, rs in sorted(by_sess.items(), key=lambda kv: float(kv[1][0]['time_s'])):
    t = [f(r,'time_s') for r in rs]; t = [x for x in t if x is not None]
    span = max(t)-min(t) if t else 0; total_span += span
    # Event (kind 1) records consume sequence numbers too, so a control-only
    # view shows a false gap wherever an event sits. Merge both streams.
    seqs = sorted({int(r['sequence']) for r in rs} |
                  {int(e['sequence']) for e in events if str(e.get('session')) == s})
    gaps = sum(1 for a, b in zip(seqs, seqs[1:]) if b != a + 1)
    drop = max(int(r['dropped_records']) for r in rs)
    print(f'  {s:>12s}  {len(rs):>6d} rec  {span:7.1f} s  seq_gaps={gaps}  dropped={drop}')
print(f'  {"TOTAL":>12s}  {len(rows):>6d} rec  {total_span:7.1f} s  = {total_span/60:.2f} min')

# ---- data completeness ----------------------------------------------------
print('\n## Data completeness (finite values)')
for grp in ['obs_','action_','goal_rad_','q_rad_','qd_rad_s_','load_signed_raw_','current_raw_']:
    cols = [c for c in rows[0] if c.startswith(grp)]
    if not cols: continue
    tot = len(cols)*len(rows)
    fin = sum(1 for r in rows for c in cols if f(r,c) is not None)
    flag = '' if fin else '   <-- NOT LOGGED'
    print(f'  {grp:<20s} {fin:>8d}/{tot:<8d} ({100*fin/tot:5.1f}%){flag}')
print(f'  servos.csv present: {(d/"servos.csv").exists()}')

# ---- timing ---------------------------------------------------------------
print(f'\n## Timing (budget {TICK_US} us/tick)')
for key, label in [('interval_us','tick interval'),('read_us','servo read'),
                   ('infer_us','policy inference'),('work_us','total work')]:
    line(label, [f(r,key) for r in rows], 'us')
over = sum(1 for r in rows if (f(r,'interval_us') or 0) > 21000)
maxdl = max((int(r['missed_deadlines']) for r in rows), default=0)
maxmiss = max((int(r['missing_ticks']) for r in rows), default=0)
print(f'  intervals >21 ms: {over} / {len(rows)} ({100*over/len(rows):.2f}%)')
print(f'  missed_deadlines (max counter): {maxdl}   missing_ticks (max): {maxmiss}')

# ---- feedback reliability -------------------------------------------------
print('\n## Per-joint feedback dropouts (valid_mask)')
miss = collections.Counter(); err = collections.Counter()
for r in rows:
    vm = int(r['valid_mask']); em = int(r['error_mask'])
    for j in range(12):
        if not (vm >> j) & 1: miss[j] += 1
        if (em >> j) & 1: err[j] += 1
any_miss = False
for j in range(12):
    if miss[j] or err[j]:
        any_miss = True
        print(f'  [{j:>2d}] id{SERVO_ID[j]:<3d} {JOINTS[j]:<15s} no-reply {miss[j]:>5d} ({100*miss[j]/len(rows):5.2f}%)  servo_err {err[j]}')
if not any_miss: print('  all 12 joints answered every tick')

# ---- tracking error -------------------------------------------------------
print('\n## Joint tracking error |goal - measured| (deg)')
for j in range(12):
    v = [abs(x) for x in (f(r,f'goal_minus_measured_deg_{j}') for r in rows) if x is not None]
    if not v: continue
    rms = math.sqrt(sum(x*x for x in v)/len(v))
    print(f'  [{j:>2d}] {JOINTS[j]:<15s} rms {rms:6.2f}  p95 {pct(v,.95):6.2f}  max {max(v):6.2f}')

# ---- attitude -------------------------------------------------------------
print('\n## Attitude')
for key, label in [('tilt_deg','tilt'),('roll_deg','roll |.|'),('pitch_deg','pitch |.|')]:
    vals = [f(r,key) for r in rows]
    vals = [abs(x) for x in vals if x is not None] if key != 'tilt_deg' else [x for x in vals if x is not None]
    line(label, vals, 'deg')

# ---- commands -------------------------------------------------------------
print('\n## Commanded motion (s, at 50 Hz)')
modes = collections.Counter()
for r in rows:
    vx, vy, wz = f(r,'target_vx_mps'), f(r,'target_vy_mps'), f(r,'target_wz_rad_s')
    if None in (vx,vy,wz): continue
    tags = []
    if vx >  0.005: tags.append('forward')
    if vx < -0.005: tags.append('backward')
    if abs(vy) > 0.003: tags.append('lateral')
    if abs(wz) > 0.02: tags.append('turn')
    modes['hold' if not tags else '+'.join(tags)] += 1
for m, n in modes.most_common():
    print(f'  {m:<22s} {n/50:7.1f} s  ({n} ticks)')

flags = collections.Counter()
for r in rows:
    fl = int(r['flags'])
    for bit, name in [(1,'enabled'),(4,'skipped/paused'),(8,'motion requested'),
                      (16,'controller linked'),(32,'dead-man'),(64,'policy goal applied')]:
        if fl & bit: flags[name] += 1
print('\n## Flag occupancy')
for name, n in flags.most_common():
    print(f'  {name:<22s} {n:>6d} ticks ({100*n/len(rows):5.1f}%)')

# ---- events ---------------------------------------------------------------
print('\n## Events (non-identity)')
skip = ('joint index=','imu gyro_bias','firmware','build','limits','default pose','action scale')
shown = 0
for e in events:
    t = e.get('text','')
    if any(t.startswith(s) or s in t[:20] for s in skip): continue
    print(f'  [{e.get("time_us",0)/1e6:8.2f} s] {t[:110]}')
    shown += 1
    if shown > 40: print('  ...'); break
if not shown: print('  (none)')
print()
