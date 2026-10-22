# K2 on Seeed XIAO ESP32S3 Sense

The `robot` firmware runs the policy compiled into `src/policy_weights.h`: 50 observations,
256–128–64 hidden units and 12 actions. Channel 5 is lateral velocity.
Both staged lateral and `smooth_full_v1` ONNX policies are supported. The C++
command smoothing matches the Python `smooth_full_v1` profile, including
deceleration, double-support stopping and lateral leading-foot selection.
Height commands and standing rotation are not part of this policy.
For staged lateral policies this adds deployment smoothing; training used
different command transitions, so evaluate starts, stops and direction changes
on the robot. The exporter preserves ONNX metadata and checks the input layout.
The maintainer has tested this firmware on the physical K2, as well as the
Python deployment on Raspberry Pi 5. These demonstrations do not establish
performance on every surface or independently assembled robot.

## Wiring (unchanged peripherals)

| Peripheral | XIAO pin |
| --- | --- |
| LSM6DS3 SDA | D4 / GPIO5 |
| LSM6DS3 SCL | D5 / GPIO6 |
| Servo driver RX | D6 / GPIO43 (ESP TX) |
| Servo driver TX | D7 / GPIO44 (ESP RX) |

Share ground. The Waveshare driver must select TTL, not USB. The servo UART is
1 Mbps; servo IDs, offsets, signs and limits are generated from the Pi's files.
The IMU is detected at 0x6A or 0x6B. Its mounting orientation must stay the same.

## Test from a phone

1. Power the driver and ESP32 with the robot supported. Keep it still while the
   gyro calibrates. Boot checks the IMU, all 12 servo replies and policy output.
2. Connect the phone to Wi-Fi **K2-Robot**, password **k2robot50**. Accept staying
   on this network without internet. Open **http://192.168.4.1**.
3. Boot automatically approaches the default crouch over five seconds, then
   verifies torque off. At **READY**, press **Enable torque / stand** to start
   recording and policy control. Keep the robot supported during startup.
4. Hold a direction button to move. Release it to request hold. Browser commands
   expire after 500 ms without a refresh. Closing/hiding the page or losing Wi-Fi
   therefore requests a smooth stop; the deceleration takes additional time.
5. **Release torque** latches a stop. Support the robot before using it.
   Use **Reboot robot** to recover from a stop or fall cutoff without opening
   the enclosure. Reconnect to K2-Robot if necessary; reboot repeats the automatic
   crouch and torque release before waiting for policy enable. Keep the robot still during gyro calibration.

Joystick deflection scales continuously through the deadzone to the policy's
trained limits: forward/backward ±0.05 m/s, lateral ±0.03 m/s, and turning
±0.30 rad/s. Full deflection reaches the limit in either direction.

Web buttons start at forward/backward 0.02 m/s, lateral 0.015 m/s, and turning
0.10 rad/s. Open **Speed settings** to change these three speeds up to the
trained limits. Settings apply only to web buttons, and reset on reboot.
Both input sources retain the trained acceleration/jerk limits, so reaching
maximum speed is gradual. Centered/no controls request learned HOLD, never a latched
march. This prevents stale commanded motion; actual standstill stability still
depends on the policy, calibration and contact conditions.

## Xbox BLE controller

The existing paired Xbox BLE address is retained in `policy_robot.cpp`.
For a different controller, set `PAD_ADDRESS` in that file to your controller's
BLE address before building; the `blescan` diagnostic can help identify it.
With centered sticks, press a trigger plus **A** to enable torque independently
of the web page and start the logger/policy. The boot crouch is already complete. The web
enable button or serial `g` are alternatives. Hold LT or RT as a dead-man.
Left stick Y walks, left stick X strafes, right stick X turns. A marches only
while pressed; B requests hold. Centered sticks with A released request hold.
Releasing triggers or disconnecting requests hold. BLE HID reports can be quiet
while a stick is held steady, so link state—not notification age—detects loss.
Press a trigger plus **X** for one second to reboot from running, ready, or
fault state. X alone does not reboot. Release the chord before using it again;
holding it across boot does not cause a reboot loop. Torque release is checked
before reboot; if a servo will not confirm release, the firmware remains stopped.
Back or serial `x` releases torque and latches a fault.

## Switching between web and gamepad

Both inputs stay available. The latest deliberate command takes ownership;
commands are never added together. Connecting an idle gamepad or refreshing a
held web command does not steal ownership. Move a stick or press another web
direction to switch. Releasing the active input or losing its connection requests
hold, without resuming an older command from the other source. The status card
shows Web, Gamepad or Hold, plus the BLE connection state. If new commands arrive
in the same control tick, the gamepad command wins that tie.

## Build, upload and verify

The selected staged lateral checkpoint is currently
`logs/rsl_rl/k2_lateral_v1/2026-09-15_08-31-17/last_policy.onnx`.
Its headers have been generated. To regenerate and verify this specific policy:

```bash
conda activate unitree_sim_env
policy=logs/rsl_rl/k2_lateral_v1/2026-09-15_08-31-17/last_policy.onnx
python esp32-s3/tools/export_policy.py "$policy" esp32-s3/src/policy_weights.h
python esp32-s3/tools/export_k2_config.py "$policy" esp32-s3/src/k2_config.h
python esp32-s3/tests/verify.py --policy "$policy"
~/.platformio/penv/bin/pio run -d esp32-s3 -e robot
~/.platformio/penv/bin/pio run -d esp32-s3 -e robot -t upload --upload-port /dev/ttyACM0
```

This checkpoint's yaw limit is ±0.25 rad/s. Forward speed remains ±0.05 m/s;
lateral speed follows the backup and policy limit of ±0.03 m/s. Command acceleration
and jerk limits, gradual stopping and the joint slew limit remain enabled.
Use the actual USB port if it differs from `/dev/ttyACM0`.

To select the generic `policies/last_policy.onnx` instead, from the repository root:

```bash
python esp32-s3/tools/export_policy.py policies/last_policy.onnx esp32-s3/src/policy_weights.h
python esp32-s3/tools/export_k2_config.py policies/last_policy.onnx esp32-s3/src/k2_config.h
python esp32-s3/tests/verify.py
~/.platformio/penv/bin/pio run -d esp32-s3 -e robot
~/.platformio/penv/bin/pio run -d esp32-s3 -e robot -t upload --upload-port /dev/ttyACM0
~/.platformio/penv/bin/pio device monitor --port /dev/ttyACM0
```

The native tests compare C++ inference against ONNX Runtime, observation/profile
transitions against Python, web command expiry, gamepad behavior, servo protocol
errors and generated calibration. They do not measure ESP32 timing or prove
real-robot gait quality. Serial telemetry reports inference, feedback and loop
times. Tilt over 24 degrees, invalid feedback or missed control deadlines stop
the controller. HTTP and BLE handling run outside the servo-control loop.

If upload fails, hold BOOT, tap RESET, release BOOT, then retry. Native USB may
re-enumerate under a different `/dev/ttyACM*` port. The camera and microphone
are unused. Older diagnostic environments describe historical firmware;
use the `robot` environment for the current policy.

## Restored deployment

The runtime source and PlatformIO settings were restored from an earlier
known-good deployment. This restores the original 4-byte
position/velocity group read with retries, fault handling, 14° tilt cutoff,
web controls. Transient-stop recovery was subsequently restored as described below.
The flash logger is now integrated again using its dedicated partition, while
retaining the working 4-byte servo reads. See [field logging](docs/field_logging.md)
for recording limits, initialization and download commands.

The currently exported lateral policy weights are retained. Its generated metadata
uses a 0.8 s gait transition and ±0.25 rad/s yaw limit, versus the backup
policy's 1.0 s and ±0.30 rad/s. Calibration, observation construction and
command smoothing match the backup. No training config was changed.


### Restored transient-stop recovery

The backup runtime now includes the previously tested feedback recovery guard
and partial-reply parser. It keeps the compact 4-byte position/velocity read,
with one bounded attempt per policy tick. A missed reply holds the last goals
and skips inference for that tick; fresh feedback automatically resumes it.
Five consecutive incomplete ticks or a joint stale for over 150 ms latch a
pause: release controls after feedback recovers to resume HOLD. Timing overruns
also pause with torque retained. Serial reports latched pauses. Feedback absent
for over 2 seconds or a loop gap over 2 seconds faults. The 14° tilt cutoff,
operator stops, policy, calibration, smoothing and speed limits are unchanged.
Flight logging and its larger telemetry reads remain disabled.
