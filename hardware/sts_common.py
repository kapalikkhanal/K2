"""Shared helpers for Feetech STS3215 bus servos (Waveshare adapter).

STS/SMS series -> PacketHandler protocol_end = 0. Default bus baud = 1 Mbps.
Register map (STS3215):
  5  ID            6  Baud        40 TorqueEnable   55 Lock(0=unlock,1=lock)
  42 GoalPos(2B)   56 PresentPos(2B)
"""
import scservo_sdk as scs

ADDR_ID = 5
ADDR_BAUD = 6
ADDR_TORQUE = 40
ADDR_GOAL_POS = 42
ADDR_LOCK = 55
ADDR_PRESENT_POS = 56

BAUDS = [1_000_000, 500_000, 115_200, 57_600]  # try these when scanning

# STS3215 joint -> bus ID map for the full Robot_v2 biped.
# Right leg is FOOT-UP (ankle=1 .. hip_pitch=5); left leg mirrors it HIP-DOWN
# (hip_pitch=6 .. ankle=10), so both feet sit at the extremes (R-ankle=1, L-ankle=10).
LEG_IDS = {
  # right leg (foot-up)
  "ankle_R": 1,
  "knee_R": 2,
  "hip_yaw_R": 3,
  "hip_roll_R": 4,
  "hip_pitch_R": 5,
  # left leg (mirror: hip-down, ankle_L = 10)
  "hip_pitch_L": 6,
  "hip_roll_L": 7,
  "hip_yaw_L": 8,
  "knee_L": 9,
  "ankle_L": 10,
}


def open_port(port: str, baud: int = 1_000_000):
  ph = scs.PortHandler(port)
  pk = scs.PacketHandler(0)  # STS/SMS little-endian
  # setBaudRate opens the port. Avoid openPort() followed by a redundant close
  # and reopen, which can fail after a partial UART response.
  if not ph.setBaudRate(baud):
    raise RuntimeError(f"failed to open {port} at baud {baud}")
  return ph, pk


def scan_ids(ph, pk, id_range=range(0, 21)):
  """Return list of (id, model) that respond to ping on the current baud."""
  found = []
  for i in id_range:
    model, comm, err = pk.ping(ph, i)
    if comm == scs.COMM_SUCCESS:
      found.append((i, model))
  return found


def unlock(ph, pk, sid):
  pk.write1ByteTxRx(ph, sid, ADDR_LOCK, 0)


def lock(ph, pk, sid):
  pk.write1ByteTxRx(ph, sid, ADDR_LOCK, 1)


def set_id(ph, pk, old_id, new_id):
  unlock(ph, pk, old_id)
  pk.write1ByteTxRx(ph, old_id, ADDR_ID, new_id)
  lock(ph, pk, new_id)  # servo now answers on new_id


def read_pos(ph, pk, sid):
  val, comm, err = pk.read2ByteTxRx(ph, sid, ADDR_PRESENT_POS)
  return val if comm == scs.COMM_SUCCESS else None
