"""Minimal repository-owned LSM6DS3 I2C driver for K2.

Only the accelerometer and gyroscope path needed by the policy is implemented.
``read()`` returns acceleration in g and angular velocity in degrees/second,
matching the interface historically supplied by ``visualize_imu.LSM6DS3``.
"""

from __future__ import annotations

import time

import numpy as np
from smbus2 import SMBus


class LSM6DS3:
    WHO_AM_I = 0x0F
    WHO_AM_I_VALUE = 0x69
    CTRL1_XL = 0x10
    CTRL2_G = 0x11
    CTRL3_C = 0x12
    OUTX_L_G = 0x22

    # Configuration below: 104 Hz, accelerometer +/-2 g, gyro +/-245 dps.
    ACCEL_G_PER_LSB = 0.000061
    GYRO_DPS_PER_LSB = 0.00875

    def __init__(self, bus_number: int = 1, address: int | None = None):
        self.bus = SMBus(bus_number)
        self.address = address if address is not None else self._detect_address()
        # Start from a known register state.  The Pi and sensor may survive a
        # software restart without a power cycle, so inheriting an old ODR,
        # full-scale, or interface configuration is not safe.
        self.bus.write_byte_data(self.address, self.CTRL3_C, 0x01)  # SW_RESET
        deadline = time.monotonic() + 0.2
        while self.bus.read_byte_data(self.address, self.CTRL3_C) & 0x01:
            if time.monotonic() >= deadline:
                self.close()
                raise RuntimeError("LSM6DS3 software reset timed out")
            time.sleep(0.002)
        self.bus.write_byte_data(self.address, self.CTRL3_C, 0x44)  # BDU + auto-inc
        self.bus.write_byte_data(self.address, self.CTRL1_XL, 0x40)
        self.bus.write_byte_data(self.address, self.CTRL2_G, 0x40)
        # Discard the power-up/ODR transition interval before bias capture.
        time.sleep(0.12)

    def _detect_address(self) -> int:
        failures = []
        for address in (0x6A, 0x6B):
            try:
                who = self.bus.read_byte_data(address, self.WHO_AM_I)
            except OSError as exc:
                failures.append(f"0x{address:02x}: {exc}")
                continue
            if who == self.WHO_AM_I_VALUE:
                return address
            failures.append(f"0x{address:02x}: WHO_AM_I=0x{who:02x}")
        self.bus.close()
        raise RuntimeError("LSM6DS3 not found (" + "; ".join(failures) + ")")

    @staticmethod
    def _int16(lo: int, hi: int) -> int:
        value = lo | (hi << 8)
        return value - 65536 if value & 0x8000 else value

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        # Gyro XYZ occupies 0x22..0x27 and accel XYZ immediately follows at
        # 0x28..0x2d, so one block transaction gives a coherent sample.
        raw = self.bus.read_i2c_block_data(self.address, self.OUTX_L_G, 12)
        values = np.array(
            [self._int16(raw[i], raw[i + 1]) for i in range(0, 12, 2)],
            dtype=np.float64,
        )
        gyro_dps = values[:3] * self.GYRO_DPS_PER_LSB
        accel_g = values[3:] * self.ACCEL_G_PER_LSB
        return accel_g, gyro_dps

    def close(self) -> None:
        self.bus.close()
