#!/usr/bin/env bash
# Pair an Xbox Wireless Controller to this Pi for joystick.py.
#
# Xbox controllers will not bind their HID profile while Bluetooth ERTM is
# enabled: BlueZ reports Connected but Bonded: no, no /dev/input/event* node
# is created, and joystick.py's find_pad() sees nothing.  Disabling ERTM is
# therefore a prerequisite for pairing, not a tweak -- and it must be
# persisted, or the next reboot silently breaks the controller again.
#
#   sudo scripts/pair_xbox.sh            # fix ERTM, then pair interactively
#   sudo scripts/pair_xbox.sh --ertm-only  # just make the ERTM fix persistent
set -euo pipefail

MODPROBE_CONF=/etc/modprobe.d/xbox-controller.conf
ERTM_PARAM=/sys/module/bluetooth/parameters/disable_ertm

if [[ $EUID -ne 0 ]]; then
  echo "needs root: sudo $0 $*" >&2
  exit 1
fi

# 1. ERTM off now (applies to new connections) and across reboots.
if [[ "$(cat "$ERTM_PARAM")" != "Y" ]]; then
  echo "Y" > "$ERTM_PARAM"
  echo "[+] ERTM disabled for this boot"
else
  echo "[=] ERTM already disabled"
fi

if ! grep -qs "disable_ertm" "$MODPROBE_CONF"; then
  echo "options bluetooth disable_ertm=1" > "$MODPROBE_CONF"
  echo "[+] persisted to $MODPROBE_CONF"
else
  echo "[=] already persisted in $MODPROBE_CONF"
fi

[[ "${1:-}" == "--ertm-only" ]] && exit 0

# 2. Drop any half-bonded entry.  A controller that is Trusted but not Bonded
#    auto-connects and then fails to expose a gamepad, which looks like a dead
#    controller rather than a stale pairing.
mapfile -t STALE < <(bluetoothctl devices | grep -i "xbox" | awk '{print $2}')
for mac in "${STALE[@]:-}"; do
  echo "[+] removing stale entry $mac"
  bluetoothctl remove "$mac" >/dev/null 2>&1 || true
done

# 3. Pair.  The controller must be in pairing mode: hold the Xbox button until
#    it lights, then hold the small sync button on the back until it flashes
#    rapidly (about 3 s).  A slow blink means connected-but-not-pairing.
echo
echo ">>> Put the controller in pairing mode now (Xbox button, then hold sync"
echo ">>> on the back until the light flashes FAST), then press Enter."
read -r

bluetoothctl --timeout 20 scan on >/dev/null 2>&1 &
sleep 12

MAC="$(bluetoothctl devices | grep -i "xbox" | awk '{print $2}' | head -1)"
if [[ -z "$MAC" ]]; then
  echo "no Xbox controller found in scan. Is it flashing fast?" >&2
  exit 1
fi
echo "[+] found $MAC"

bluetoothctl pair "$MAC"
bluetoothctl trust "$MAC"
bluetoothctl connect "$MAC"

sleep 2
echo
if bluetoothctl info "$MAC" | grep -q "Bonded: yes"; then
  echo "[+] bonded"
else
  echo "[!] still not bonded -- power-cycle the controller and rerun" >&2
fi
ls /dev/input/event* 2>/dev/null
for f in /dev/input/event*; do
  printf '  %s: %s\n' "$f" "$(cat "/sys/class/input/$(basename "$f")/device/name" 2>/dev/null)"
done
