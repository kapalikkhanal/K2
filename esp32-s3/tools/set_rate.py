"""env:robot_rates only: wait for READY, set the control rate (50/25/10), report status.

Opening the port resets the board (boot crouch, torque ON). usage: set_rate.py 50|25|10
"""
import serial, sys, time
hz = sys.argv[1]; key = {'50': b'5', '25': b'2', '10': b'1'}[hz]
s = serial.Serial(port=None, baudrate=115200, timeout=0.3); s.dtr = False; s.rts = False
s.port = '/dev/ttyACM0'; s.open()
end = time.time() + 90; state = 'boot'
while time.time() < end:
    l = s.readline().decode('utf-8', 'replace').rstrip()
    if not l: continue
    if any(k in l for k in ('RUNTIME RATE', 'RATE set', 'LOG status', 'SAFETY', 'self-test', 'servos:', 'READY')): print(l, flush=True)
    if 'SAFETY STOP' in l: sys.exit(2)
    if 'READY' in l and state == 'boot':
        time.sleep(0.5); s.write(key); state = 'sent'
    elif 'RATE set' in l:
        time.sleep(0.3); s.write(b'L'); state = 'done'
    elif state == 'done' and l.startswith('LOG status'):
        break
s.close()
