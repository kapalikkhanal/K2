# ESP32-S3 paper trial logging

## Startup and controls

Keep K2 supported and still during gyro calibration. After the existing IMU,
servo and inference self-checks, boot **automatically moves to the default crouch
in five seconds**, commands the exact final pose, and verifies torque disabled
on all 12 servos. It then waits with no policy control and no flash writes.
An unpowered pose can sag; support the robot as needed.

With centered sticks, release and then press **LT/RT + A** to start the logger
and policy. Browser enable and serial `g` remain equivalent start controls.
Arming reads the actual joint positions, writes those goals before enabling
torque, and hands over through the existing policy slew limiter. It does not
repeat the five-second approach. A alone does not arm; holding A after arming
retains the existing march command. Back, browser stop, or serial `x` stops and
verifies torque release. Reboot repeats the automatic boot crouch.

The working policy weights, observation/control math, 14° tilt cutoff, servo
profiles, and 4-byte position/velocity group-read payload/timeouts are retained.
Logger storage and startup changes are confined to the deployment firmware.

## Build and initialize

```bash
pio run -d esp32-s3 -e robot
python esp32-s3/tools/archive_trial.py --out trial-build.zip
pio run -d esp32-s3 -e robot -t upload --upload-port PORT
python esp32-s3/tools/flight_logs.py status --port PORT
```

The USB-only partition table reserves 0x210000..0x7effff for logs and removes
OTA's second application slot. Normal application uploads preserve this data;
whole-chip erase, old partition layouts and filesystem uploads may destroy it.
Unknown existing partition contents are never automatically erased. If needed,
with torque off, save the existing contents and explicitly initialize:

```bash
python esp32-s3/tools/flight_logs.py download --port PORT --out previous.bin
python esp32-s3/tools/flight_logs.py erase --port PORT --confirm-erase
```

`ready=1`, `holding=1`, `full=0`, `io_error=0` indicates ready for arming.
Initialization while waiting restages the boot identity. If already faulted,
reboot after initialization. Firmware refuses to start a trial without ready
storage. Downloads and erases are refused whenever torque may remain enabled.
No flashing or hardware motion is performed by the host tests.

## Recording

Boot identity is staged in RAM; flash recording starts only at operator enable.
Pre-arm faults discard staged records. Each armed session includes:

- Firmware MD5, build/reset reason, sample rates, action scale, gait settings,
  safety thresholds, gyro calibration, level matrix, joint IDs/home/signs,
  default pose and limits. Archive the exact sources and binary before trials.
- Every **50 Hz** control tick: complete 50-D policy observation, 12 actions,
  12 bounded/slewed joint goals, measured joint positions and velocities
  (reconstructed from observations/default pose), raw chip-frame acceleration
  and gyro, target commands, estimated tilt/heading, heading reference and phase.
- Actual tick interval, read/inference/work durations, missing-feedback and
  deadline counters, per-joint valid/error masks, controller age, state flags,
  transitions, arming, faults, torque-release verification and deliberate reboot.
  `work_us` ends before record CRC/enqueue; the following `interval_us` captures
  their scheduling impact. Deadline counts include intervals over 21 ms and
  inference work reaching the 20 ms budget; both may count in one tick.
- Paused/skipped ticks are recorded, with unused policy fields and unavailable
  joint samples NaN. Flags: 1 enabled, 2 approach (historical), 4 skipped/paused,
  8 motion requested, 16 controller linked, 32 dead-man, 64 policy goal applied.

**Current/load/voltage/temperature are not collected by this firmware.** The
working 4-byte servo payload contains position/velocity only. v3 load/current
slots remain NaN and no servo telemetry records are emitted. Old extended-read
captures still decode, including v2 and v3 data. Do not interpret empty fields
as zero or use them to infer torque, energy, or power.

512-byte CRC-protected records enter a 64-record RAM queue and a separate flash
writer. Flash writes can still stall the MCU; measure actual board timing in a
supported trial. Capacity is nominally **four minutes** at 50 Hz, less with
state events and existing runs. Plan **three minutes total per empty partition**
and check the browser's remaining seconds and dropped count. Remaining seconds
are an upper bound based on control records alone. Full/failed storage stops
recording without overwriting old data or stopping the robot; the browser shows
NOT RECORDING. Stop the trial if recording fails.

After stopping, wait at least three seconds before removing power. Abrupt power
loss can lose the queued tail (64 records, about 1.28 seconds of control data)
and tear the in-progress record. CRC detects torn writes; sequence/drop counters
expose queue losses. Timing snapshots on faulting ticks may be absent; the fault
event records the reason and torque-release result.

## Retrieve and analyze

Reconnect USB without reflashing. With torque off:

```bash
python esp32-s3/tools/flight_logs.py download --port PORT --out trial.bin
python esp32-s3/tools/flight_logs.py decode trial.bin --out-dir trial-analysis
```

Download never erases. Keep `trial.bin`, its `.meta.json` sidecar (host UTC,
port, byte count, SHA-256), `trial-build.zip`, analysis, and trial notes/video.
The decoder writes `control.csv`, `events.json`, `summary.json`, and `servos.csv`
only when servo telemetry exists. Summaries report timing, feedback/drop counts,
corruption, joint tracking errors and motion segments. The ESP32 records uptime,
not wall time; host download time is not the trial start time.

If firmware cannot boot, bootloader `read_flash 0x210000 0x5e0000 trial.bin`
can recover the partition using esptool with motor power off. Retrieve before
repairing/reflashing. No OTA or automatic formatting is used.

## Paper measurements

These records support joint tracking, estimated torso sway, gait timing,
command transitions, feedback reliability and inference/control timing. 50 Hz
snapshots cannot resolve frequencies at or above 25 Hz; actuator error includes
sample/command timing and dynamics. Integrated yaw can drift. Motion summaries
exclude skipped/incomplete-feedback ticks and modes can overlap.

For ground-truth speed/distance and falls, record uninterrupted video with a
visible distance scale, trial surface, robot mass/configuration, battery state,
command sequence, assistance and repeated successes/failures. Synchronize with a
visible command transition. External power measurement is required for energy
or cost-of-transport claims. Commanded velocity is not measured translation;
the robot has no ground-truth translation, foot-force or calibrated power sensor.
