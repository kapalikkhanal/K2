"""Shared NumPy/Torch command dynamics for the full K2 policy.

No robot I/O or torch import: usable on the Pi and in batched RL training.
All inputs are arrays; xp is numpy or torch. Rates are physical command rates,
not joint/servo gains. Hold is reached only after deceleration and double support.
"""
import math

RATE = (0.04, 0.035, 0.25)  # vx, vy, wz: m/s², m/s², rad/s²
JERK = (0.12, 0.105, 0.75)
STOP_SPEED = (0.001, 0.001, 0.008)
STOP_RATE = (0.005, 0.005, 0.04)
VERSION = "smooth_full_v1"


def step_profile(xp, value, rate, phase, march, settling, target, moving,
                 dt, frequency, transition, limits, jerks, stop_speed, stop_rate):
  requested = xp.where(moving[..., None], target, target * 0)
  # Damped second-order response with explicit speed-change and jerk bounds.
  acceleration = xp.clip(36 * (requested - value) - 12 * rate, -jerks, jerks)
  rate = xp.clip(rate + acceleration * dt, -limits, limits)
  value = value + rate * dt
  stopped = (abs(value) <= stop_speed).all(axis=-1) & (abs(rate) <= stop_rate).all(axis=-1)
  double_support = abs(xp.sin(phase)) < 0.30
  settling = (~moving) & (settling | (stopped & double_support))
  active = moving | (~settling & (march > 0))
  # Select the leading foot only on departure from a completed hold.
  initial = xp.where(requested[..., 1] < 0, phase * 0 + math.pi, phase * 0)
  phase = xp.where(moving & (march <= 0), initial, phase)
  delta = march * 0 + dt / transition
  march = xp.clip(march + xp.where(active, delta, -delta), 0.0, 1.0)
  advance = active & (march >= 1 - 1e-6)
  phase = xp.where(advance, (phase + 2 * math.pi * frequency * dt) % (2 * math.pi), phase)
  phase = xp.where(march <= 0, phase * 0, phase)
  return value, rate, phase, march, settling


def step_height(xp, value, rate, target, dt):
  """Standing-height profile shared by the staged trainer and Python runner."""
  acceleration = xp.clip(36 * (target - value) - 12 * rate, -0.04, 0.04)
  rate = xp.clip(rate + acceleration * dt, -0.012, 0.012)
  value = value + rate * dt
  return value, rate
