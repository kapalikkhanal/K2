#!/usr/bin/env python3
"""env:robot_rates only: set control rate, CPU clock and inference mode, then
confirm the boot banner reports them.

Opening the port resets the board, and every boot runs the 5 s crouch with
torque ON, so the robot must be powered and standing/supported. Clock and mode
changes reboot the board once more each.

usage: set_config.py --hz 50|25|10 --mhz 240|160|80 --mode N|F
  N = deployed placement + scalar loop (today's firmware behaviour)
  F = weights that miss SRAM go to PSRAM + ESP-DSP dot products
"""
import argparse, re, sys, time
import serial

ap = argparse.ArgumentParser()
ap.add_argument('--hz', choices=('50', '25', '10'), required=True)
ap.add_argument('--mhz', choices=('240', '160', '80'), required=True)
ap.add_argument('--mode', choices=('N', 'F'), required=True)
ap.add_argument('--port', default='/dev/ttyACM0')
a = ap.parse_args()
want_mode = 'psram+esp-dsp' if a.mode == 'F' else 'deployed-scalar'

def open_port():
    for _ in range(60):  # the USB device re-enumerates after esp_restart()
        try:
            p = serial.Serial(port=None, baudrate=115200, timeout=0.3)
            p.dtr = False; p.rts = False; p.port = a.port; p.open()
            return p
        except serial.SerialException:
            time.sleep(0.5)
    sys.exit('port did not come back')

s = open_port()

def readline():
    global s
    try:
        return s.readline().decode('utf-8', 'replace').rstrip()
    except serial.SerialException:
        s.close(); time.sleep(1.0); s = open_port()
        return ''

def until_ready(seconds=120):
    """Return the RUNTIME banner of the boot that reaches READY."""
    end, banner = time.time() + seconds, None
    while time.time() < end:
        l = readline()
        if not l: continue
        if l.startswith('RUNTIME build:'): banner = l; print(l, flush=True)
        elif any(k in l for k in ('self-test', 'servos:', 'sync read', 'SAFETY', 'READY', 'BOOT CONFIG', 'RATE')):
            print('  ' + l, flush=True)
        if 'SAFETY STOP' in l: sys.exit('board faulted (servo power off?)')
        if l.startswith('READY'): return banner
    sys.exit('never reached READY')

banner = until_ready()
for _ in range(3):
    m = re.search(r'control (\d+) Hz, CPU (\d+) MHz, inference (\S+)', banner or '')
    if not m: sys.exit('not a robot_rates firmware')
    hz, mhz, mode = m.groups()
    if mhz != a.mhz:
        time.sleep(0.5); s.write({'240': b'4', '160': b'6', '80': b'8'}[a.mhz]); banner = until_ready(); continue
    if mode != want_mode:
        time.sleep(0.5); s.write(a.mode.encode()); banner = until_ready(); continue
    if hz != a.hz:
        time.sleep(0.5); s.write({'50': b'5', '25': b'2', '10': b'1'}[a.hz])
        end = time.time() + 5
        while time.time() < end:
            l = readline()
            if 'RATE set' in l: print('  ' + l); break
    break
else:
    sys.exit('configuration did not converge')
time.sleep(0.3); s.write(b'L')
end = time.time() + 3
while time.time() < end:
    l = readline()
    if l.startswith('LOG status'): print('  ' + l); break
print(f'CONFIGURED: {a.hz} Hz, {a.mhz} MHz, {want_mode}. Arm with LT/RT + A.')
s.close()
