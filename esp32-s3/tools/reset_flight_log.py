#!/usr/bin/env python3
"""Erase the k2log partition on the XIAO ESP32-S3.

Opening the port resets the board (rst:0x15 USB_UART_CHIP_RESET), and every
boot runs a 5 s crouch with torque ON, during which the logger refuses D/E.
So: anchor on THIS boot's crouch line, wait for the READY banner that follows
it, and only then erase. Stale buffered text from the previous boot is ignored.
"""
import re, sys, time
import serial

PORT = '/dev/ttyACM0'
CROUCH = 'approach: easing into the default pose'
READY = 'READY: boot crouch complete'

stream = serial.Serial(port=None, baudrate=115200, timeout=0.5, write_timeout=5)
stream.dtr = False; stream.rts = False; stream.port = PORT
stream.open()

def pump(seconds):
    end = time.monotonic() + seconds
    got = []
    while time.monotonic() < end:
        raw = stream.readline()
        if not raw:
            continue
        line = raw.decode('utf-8', 'replace').rstrip()
        if line:
            got.append(line); print(f'  | {line}', flush=True)
    return got

def wait_for(needle, seconds, label):
    print(f'== waiting for {label} (<= {seconds} s) ==', flush=True)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if any(needle in l for l in pump(1.0)):
            print(f'== got {label} ==', flush=True)
            return True
    return False

# Discard anything buffered from the previous boot before anchoring.
print('== flushing stale output ==', flush=True)
pump(2.0)
stream.reset_input_buffer()

# Anchor on THIS boot: the crouch must start before we trust any READY banner.
if not wait_for(CROUCH, 90, "this boot's crouch"):
    print('!! never saw the boot crouch; NOT erasing', flush=True)
    stream.close(); sys.exit(2)
if not wait_for(READY, 60, 'READY (torque verified off)'):
    print('!! crouch never completed; NOT erasing', flush=True)
    stream.close(); sys.exit(3)

# Let the arm-wait loop settle, then confirm torque is really off.
time.sleep(1.5)
stream.reset_input_buffer()
print('== confirming logger status ==', flush=True)
status = None
for _ in range(8):
    stream.write(b'L')
    for line in pump(1.0):
        if line.startswith('LOG status '):
            status = line
    if status:
        break
if not status:
    print('!! no status response; NOT erasing', flush=True)
    stream.close(); sys.exit(4)
if (m := re.search(r'active=(\d)', status)) is None or m.group(1) != '0':
    print(f'!! torque may be on; NOT erasing -> {status}', flush=True)
    stream.close(); sys.exit(5)

print('== ERASING k2log (6 MB, may take ~60 s) ==', flush=True)
stream.write(b'E')
result = None
end = time.monotonic() + 200
while time.monotonic() < end and not result:
    for line in pump(2.0):
        if line == 'LOG erased':
            result = 'erased'
        elif line.startswith(('LOG denied', 'LOG erase failed', 'LOG unavailable')):
            result = line
print(f'== erase result: {result} ==', flush=True)

time.sleep(2.0)
stream.reset_input_buffer()
print('== post-erase status ==', flush=True)
for _ in range(8):
    stream.write(b'L')
    hit = [l for l in pump(1.0) if l.startswith('LOG status ')]
    if hit:
        print('FINAL: ' + hit[-1], flush=True)
        break
stream.close()
sys.exit(0 if result == 'erased' else 6)
