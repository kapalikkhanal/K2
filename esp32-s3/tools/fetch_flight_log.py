#!/usr/bin/env python3
"""Download the k2log partition, waiting for the board to reach READY first.

Same boot-anchoring as reset_flight_log.py: opening the port resets the XIAO,
every boot runs a 5 s crouch with torque ON, and the logger refuses D/E during
it. Download never erases. Writes a .meta.json sidecar with host UTC.

usage: fetch_flight_log.py <out.bin> [port]
"""
import hashlib, json, re, sys, time
import serial

out = sys.argv[1] if len(sys.argv) > 1 else 'trial.bin'
PORT = sys.argv[2] if len(sys.argv) > 2 else '/dev/ttyACM0'
CROUCH, READY = 'approach: easing into the default pose', 'READY: boot crouch complete'

stream = serial.Serial(port=None, baudrate=115200, timeout=0.5, write_timeout=5)
stream.dtr = False; stream.rts = False; stream.port = PORT
stream.open()

def lines(seconds):
    end = time.monotonic() + seconds
    got = []
    while time.monotonic() < end:
        raw = stream.readline()
        if not raw: continue
        t = raw.decode('utf-8', 'replace').rstrip()
        if t: got.append(t); print(f'  | {t[:120]}', flush=True)
    return got

def wait_for(needle, seconds, label):
    print(f'== waiting for {label} ==', flush=True)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if any(needle in l for l in lines(1.0)):
            print(f'== got {label} ==', flush=True); return True
    return False

print('== flushing stale output ==', flush=True)
lines(2.0); stream.reset_input_buffer()
if not wait_for(CROUCH, 90, "this boot's crouch"): sys.exit(2)
if not wait_for(READY, 60, 'READY (torque off)'): sys.exit(3)
time.sleep(1.5); stream.reset_input_buffer()

status = None
for _ in range(8):
    stream.write(b'L')
    for l in lines(1.0):
        if l.startswith('LOG status '): status = l
    if status: break
if not status: print('!! no status'); sys.exit(4)
if (m := re.search(r'active=(\d)', status)) is None or m.group(1) != '0':
    print(f'!! torque may be on: {status}'); sys.exit(5)

print('== DOWNLOADING (flash is retained, never erased) ==', flush=True)
started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
stream.write(b'D')
expected, count, buf = None, 0, bytearray()
hexline = re.compile(rb'[0-9a-f]{1024}')
end = time.monotonic() + 900
done = False
while time.monotonic() < end and not done:
    raw = stream.readline()
    if not raw: continue
    line = raw.strip()
    if line.startswith(b'LOG BEGIN '):
        expected = int(line.split()[-1]); print(f'  expecting {expected} bytes', flush=True); continue
    if line == b'LOG END':
        done = True; break
    if line.startswith((b'LOG denied', b'LOG unavailable')):
        print('!! ' + line.decode()); sys.exit(6)
    if expected is not None and hexline.fullmatch(line):
        buf += bytes.fromhex(line.decode()); count += 512
        if count % (512*500) == 0: print(f'  {count}/{expected} bytes', flush=True)

stream.close()
if not done or expected is None:
    print(f'!! incomplete: {count}/{expected}'); sys.exit(7)
if count != expected:
    print(f'!! size mismatch {count} != {expected}'); sys.exit(8)
with open(out, 'xb') as f: f.write(buf)
meta = out + '.meta.json'
with open(meta, 'w') as f:
    json.dump({'file': out, 'bytes': count, 'sha256': hashlib.sha256(buf).hexdigest(),
               'port': PORT, 'started_utc': started,
               'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'logger_status': status}, f, indent=2)
print(f'Saved {count} bytes to {out} (flash retained)\nWrote {meta}')
