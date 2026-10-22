#!/usr/bin/env python3
"""Retrieve K2 flash logs without resetting/arming; decode to CSV and JSON.

python esp32-s3/tools/flight_logs.py status --port PORT
python esp32-s3/tools/flight_logs.py download --port PORT --out run.bin
python esp32-s3/tools/flight_logs.py decode run.bin --out-dir analysis
python esp32-s3/tools/flight_logs.py erase --port PORT --confirm-erase
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time
import zlib

SIZE = 512
HEADER = struct.Struct('<IHHIIQIIIIIHHIIII')
assert HEADER.size == 64
MAGIC = 0x324C324B
# v2 stored q/qd in payload slots 86..109. v3 reconstructs those from the
# observation block and spends the slots on per-joint load/current instead.
SUPPORTED = (2, 3)
JOINT_EVENT = re.compile(r'joint index=(\d+)\b.*?\bdefault_rad=(-?[\d.]+(?:[eE][-+]?\d+)?)')


def read_records(data):
    records, invalid = [], []
    for offset in range(0, len(data), SIZE):
        block = data[offset:offset+SIZE]
        if len(block) != SIZE:
            invalid.append({'offset': offset, 'reason': 'truncated'})
            continue
        if block == b'\xff'*SIZE:
            continue
        h = HEADER.unpack_from(block)
        if h[0] != MAGIC or h[1] not in SUPPORTED or zlib.crc32(block[:508]) != struct.unpack_from('<I', block, 508)[0]:
            invalid.append({'offset': offset, 'reason': 'magic/version/CRC'})
            continue
        record = dict(zip(('magic', 'version', 'kind', 'session', 'sequence', 'time_us',
                           'flags', 'interval_us', 'read_us', 'infer_us', 'work_us',
                           'valid_mask', 'error_mask', 'dropped_records', 'missed_deadlines',
                           'missing_ticks', 'pad_age_ms'), h))
        record['offset'] = offset
        record['values'] = struct.unpack_from('<111f', block, 64)
        if record['kind'] == 1:
            record['text'] = block[64:508].split(b'\0', 1)[0].decode('utf-8', errors='replace')
        records.append(record)
    return records, invalid


def stats(values):
    v = sorted(float(x) for x in values if math.isfinite(x))
    if not v:
        return None
    return {'n': len(v), 'min': v[0], 'max': v[-1], 'mean': sum(v)/len(v),
            'rms': math.sqrt(sum(x*x for x in v)/len(v)),
            'p95': v[round(.95*(len(v)-1))]}


def columns(r, pose=None):
    v = r['values']
    row = {k: x for k, x in r.items() if k not in ('values', 'text', 'magic')}
    row['time_s'] = r['time_us']/1e6
    if r['kind'] == 2:
        for i, x in enumerate(v[:50]): row[f'obs_{i}'] = x
        for start, name in ((50, 'action'), (62, 'goal_rad')):
            for j in range(12): row[f'{name}_{j}'] = v[start+j]
        if r['version'] >= 3:
            # obs[6..17] is q minus the default pose and obs[18..29] is qd, filled
            # identically whether or not the tick ran the policy, so nothing is
            # lost. The default pose comes from this session's boot events.
            for j in range(12):
                row[f'q_rad_{j}'] = v[6+j] + pose[j] if pose and j in pose else math.nan
                row[f'qd_rad_s_{j}'] = v[18+j]
            for start, name in ((86, 'load_signed_raw'), (98, 'current_raw')):
                for j in range(12): row[f'{name}_{j}'] = v[start+j]
        else:
            for start, name in ((86, 'q_rad'), (98, 'qd_rad_s')):
                for j in range(12): row[f'{name}_{j}'] = v[start+j]
        for start, name in ((74, 'acc_chip_g'), (77, 'gyro_chip_rad_s')):
            for i, axis in enumerate('xyz'): row[f'{name}_{axis}'] = v[start+i]
        row.update(target_vx_mps=v[80], target_vy_mps=v[81], target_wz_rad_s=v[82],
                   tilt_deg=math.degrees(v[83]), heading_est_rad=v[84],
                   heading_ref_rad=v[85], phase_rad=v[110])
        row['roll_deg'] = math.degrees(math.atan2(-v[4], -v[5]))
        row['pitch_deg'] = math.degrees(math.asin(max(-1, min(1, v[3])))) if math.isfinite(v[3]) else math.nan
        row['measured_yaw_rate_rad_s'] = -v[2]  # site -> root frame
        for j in range(12):
            row[f'goal_minus_measured_deg_{j}'] = math.degrees(v[62+j]-row[f'q_rad_{j}'])
    elif r['kind'] == 3:
        for start, name in ((0, 'voltage_v'), (12, 'temperature_c'), (24, 'load_signed_raw'),
                            (36, 'current_raw'), (48, 'status'), (60, 'age_ms'), (72, 'servo_id')):
            for j in range(12): row[f'{name}_{j}'] = v[start+j]
    return row


def write_csv(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def decode(source, out_dir):
    data = source.read_bytes()
    records, invalid = read_records(data)
    out_dir.mkdir(parents=True, exist_ok=False)
    events = [{k: v for k, v in r.items() if k != 'values'} for r in records if r['kind'] == 1]
    # v3 control records carry q as an offset from the default pose, which each
    # session writes into its own boot events. A download that starts mid-run can
    # miss them; fall back to the file's pose only when every session agrees.
    poses = {}
    for r in records:
        if r['kind'] != 1: continue
        found = JOINT_EVENT.search(r['text'])
        if found: poses.setdefault(r['session'], {})[int(found.group(1))] = float(found.group(2))
    complete = {s: d for s, d in poses.items() if len(d) == 12}
    distinct = {tuple(sorted(d.items())) for d in complete.values()}
    shared = next(iter(complete.values())) if len(distinct) == 1 else None
    pose_source = {}
    def pose_for(session):
        if session in complete:
            pose_source[session] = 'session boot events'; return complete[session]
        pose_source[session] = 'other sessions in this file' if shared else None
        return shared
    control = [columns(r, pose_for(r['session'])) for r in records if r['kind'] == 2]
    telemetry = [columns(r) for r in records if r['kind'] == 3]
    write_csv(out_dir/'control.csv', control)
    write_csv(out_dir/'servos.csv', telemetry)
    (out_dir/'events.json').write_text(json.dumps(events, indent=2))
    sessions = {}
    for session in sorted({r['session'] for r in records}):
        rows = [r for r in control if r['session'] == session]
        # Only inferred walking/marching frames with all joints fresh.
        moving = [r for r in rows if (r['flags'] & 0x49) == 0x49 and
                  not r['flags'] & 6 and r['valid_mask'] == 0xfff]
        telem = [r for r in telemetry if r['session'] == session]
        sequences = [r['sequence'] for r in records if r['session'] == session]
        modes = {}
        for name, predicate in {
            'all_moving': lambda r: True,
            'forward': lambda r: r['target_vx_mps'] > .001 and abs(r['target_vy_mps']) < .001,
            'backward': lambda r: r['target_vx_mps'] < -.001 and abs(r['target_vy_mps']) < .001,
            'left': lambda r: r['target_vy_mps'] > .001,
            'right': lambda r: r['target_vy_mps'] < -.001,
            'turn': lambda r: abs(r['target_wz_rad_s']) > .001,
        }.items():
            samples = [r for r in moving if predicate(r)]
            modes[name] = {key: stats(r[key] for r in samples) for key in
                           ('tilt_deg', 'roll_deg', 'pitch_deg', 'measured_yaw_rate_rad_s')}
            modes[name]['goal_minus_measured_deg'] = {
                str(j): stats(r[f'goal_minus_measured_deg_{j}'] for r in samples) for j in range(12)}
        loaded = [r for r in moving if 'load_signed_raw_0' in r]
        sessions[str(session)] = {
            'record_versions': sorted({r['version'] for r in records if r['session'] == session}),
            'default_pose_source': pose_source.get(session),
            'control_records': len(rows), 'moving_records': len(moving),
            'recorded_span_s': ((rows[-1]['time_s']-rows[0]['time_s']) if rows else 0),
            'sequence_gaps': sum(max(0, b-a-1) for a, b in zip(sequences, sequences[1:])),
            'max_dropped_counter': max((r['dropped_records'] for r in records if r['session']==session), default=0),
            'max_missing_ticks': max((r['missing_ticks'] for r in rows), default=0),
            'max_missed_deadlines': max((r['missed_deadlines'] for r in rows), default=0),
            'timing_us': {key: stats(r[key] for r in rows) for key in
                          ('interval_us', 'read_us', 'infer_us', 'work_us')},
            'modes': modes,
            'servo_telemetry': {str(j): {key: stats(r[f'{key}_{j}'] for r in telem
                if r['valid_mask'] & (1 << j)) for key in
                ('voltage_v', 'temperature_c', 'load_signed_raw', 'current_raw')} for j in range(12)},
            # v3 only: load/current sampled with every logged control tick rather
            # than once a second, which is what resolves them against gait phase.
            'control_rate_servo': ({str(j): {key: stats(r[f'{key}_{j}'] for r in loaded
                if r['valid_mask'] & (1 << j)) for key in ('load_signed_raw', 'current_raw')}
                for j in range(12)} if loaded else None),
        }
    summary = {'source': str(source), 'sha256': hashlib.sha256(data).hexdigest(),
               'decoded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'valid_records': len(records), 'invalid_records': invalid, 'sessions': sessions,
               'limitations': [
                   'Current firmware logs every 50 Hz control tick; older runs used 25 Hz. Check boot log_hz and record timestamps. work_us excludes record CRC/enqueue overhead; interval_us includes scheduling delays.',
                   'Targets and emitted velocity commands are not measured translation. No ground-truth speed/distance, contact force, calibrated torque, power, or cost of transport.',
                   'Current and load are raw servo values, not amperes or Nm; temperature is unfiltered.',
                   'Servo telemetry depends on firmware: the current 4-byte position/velocity reader leaves load/current NaN and emits no voltage/temperature telemetry. Older extended-reader captures may contain these fields.',
                   'v3 reconstructs q from the observation block and the recorded default pose; q is NaN for any session whose boot events are missing from the download.',
                   'Goal-minus-measured position includes actuator dynamics and sample/command timing; not an isolated actuator accuracy test.',
                   'Roll/pitch are IMU estimates; yaw is integrated and can drift. Modes can overlap (e.g. forward and turning).',
                   'No wall clock on robot. Session uptime requires external trial/video synchronization.',
                   'Abrupt power loss can lose queued records (up to 64) and the in-progress write. CRC rejects torn writes.',
               ]}
    (out_dir/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f'{len(records)} valid records, {len(invalid)} invalid; saved {out_dir}')
    return summary


def connect(port):
    import serial
    # Set line levels BEFORE open: avoid resetting the ESP32 just to read logs.
    stream = serial.Serial(port=None, baudrate=115200, timeout=.5, write_timeout=2)
    stream.dtr = False; stream.rts = False; stream.port = port; stream.open()
    return stream


def serial_command(port, operation, output=None):
    with connect(port) as stream:
        deadline = time.monotonic()+30
        status = None
        while time.monotonic() < deadline:
            stream.write(b'L')
            until = time.monotonic()+1
            while time.monotonic() < until:
                line = stream.readline().decode(errors='replace').strip()
                if line.startswith('LOG status '):
                    status = line; break
                if line.startswith('LOG denied:'): raise RuntimeError(line)
            if status: break
        if status is None: raise RuntimeError('No logger response. Check port, firmware and torque-off state.')
        print(status)
        if operation == 'status': return
        if operation == 'erase':
            stream.write(b'E')
            deadline = time.monotonic()+120
            while time.monotonic() < deadline:
                line = stream.readline().decode(errors='replace').strip()
                if line == 'LOG erased': print(line); return
                if line.startswith(('LOG denied', 'LOG erase failed')): raise RuntimeError(line)
            raise RuntimeError('Erase completion not received')
        if output is None: raise ValueError('download needs --out')
        # Exclusive creation prevents overwriting the only copy of a trial.
        with output.open('xb') as target:
            started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            stream.write(b'D'); expected = None; count = 0
            deadline = time.monotonic()+600
            while time.monotonic() < deadline:
                line = stream.readline().strip()
                if line.startswith(b'LOG BEGIN '):
                    expected = int(line.split()[-1]); continue
                if line == b'LOG END':
                    if expected is None or count != expected:
                        raise RuntimeError(f'Incomplete download: {count}/{expected}; partial file retained')
                    target.flush()
                    # The robot has no wall clock: uptime microseconds and a random
                    # session id are all a record carries. This sidecar is the only
                    # absolute time reference the pipeline ever produces.
                    meta = output.with_name(output.name+'.meta.json')
                    meta.write_text(json.dumps({
                        'file': output.name, 'bytes': count,
                        'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                        'port': port, 'started_utc': started,
                        'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        'logger_status': status,
                    }, indent=2))
                    print(f'Saved {count} bytes to {output} (flash retained)')
                    print(f'Wrote {meta}')
                    return
                if line.startswith((b'LOG denied', b'LOG unavailable')): raise RuntimeError(line.decode())
                if expected is not None and re.fullmatch(rb'[0-9a-f]{1024}', line):
                    block = bytes.fromhex(line.decode());target.write(block);count += len(block)
            raise RuntimeError('Download timed out; partial file retained, flash untouched')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    for operation in ('status', 'download', 'erase'):
        p = sub.add_parser(operation);p.add_argument('--port', required=True)
        if operation == 'download': p.add_argument('--out', type=Path, required=True)
        if operation == 'erase': p.add_argument('--confirm-erase', action='store_true')
    p = sub.add_parser('decode');p.add_argument('source', type=Path);p.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.operation == 'decode': decode(args.source, args.out_dir)
    else:
        if args.operation == 'erase' and not args.confirm_erase:
            parser.error('erase requires --confirm-erase; download first to preserve existing sessions')
        serial_command(args.port, args.operation, getattr(args, 'out', None))


if __name__ == '__main__': main()
