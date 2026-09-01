#!/usr/bin/env python3
"""Isolated PCA9685 channel-4 (shoulder) diagnostic.

Sweeps ONLY channel 4 across a wide, still-safe pulse range, holding each
step long enough to see/hear a response. Also prints the register readback
after every write so you can confirm (independent of whether the servo
physically moves) that the PCA9685 itself is accepting and holding the
command — if the readback always matches what was sent, the fault is
downstream of the chip (wiring / power / the servo itself), not the code
or the I2C bus.
"""

import sys
import time
from smbus2 import SMBus

I2C_BUS = 7
PCA9685_ADDR = 0x40
CHANNEL = int(sys.argv[1]) if len(sys.argv) > 1 else 4

MODE1 = 0x00
MODE1_AI = 0x20
PRESCALE = 0xFE
LED0_ON_L = 0x06

# Wider than servo_config.py's normal range, still within safe electrical
# limits for a standard analog servo (~500-2500us @ 50Hz => ~102-512 counts).
PULSE_MIN = 110
PULSE_MAX = 500
STEP = 40
HOLD_S = 1.2


def set_pwm(bus, channel, on, off):
    reg = LED0_ON_L + 4 * channel
    bus.write_i2c_block_data(
        PCA9685_ADDR, reg,
        [on & 0xFF, (on >> 8) & 0xFF, off & 0xFF, (off >> 8) & 0xFF])


def read_pwm(bus, channel):
    reg = LED0_ON_L + 4 * channel
    on = (bus.read_byte_data(PCA9685_ADDR, reg + 1) << 8) | \
        bus.read_byte_data(PCA9685_ADDR, reg)
    off = (bus.read_byte_data(PCA9685_ADDR, reg + 3) << 8) | \
        bus.read_byte_data(PCA9685_ADDR, reg + 2)
    return on, off


def main():
    bus = SMBus(I2C_BUS)
    bus.write_byte_data(PCA9685_ADDR, MODE1, MODE1_AI)
    time.sleep(0.005)

    print(f"Sweeping ONLY channel {CHANNEL} from {PULSE_MIN} to {PULSE_MAX} "
          f"counts (~{PULSE_MIN*4.883:.0f}-{PULSE_MAX*4.883:.0f}us), "
          f"{HOLD_S}s per step. Watch/listen to the shoulder servo now.")

    steps = list(range(PULSE_MIN, PULSE_MAX + 1, STEP))
    steps += steps[-2::-1]  # sweep back down too

    for pulse in steps:
        set_pwm(bus, CHANNEL, 0, pulse)
        on, off = read_pwm(bus, CHANNEL)
        us = pulse * 4.883
        ok = "OK" if off == pulse and on == 0 else "MISMATCH"
        print(f"  commanded pulse={pulse:4d} (~{us:6.0f}us)  "
              f"readback on={on} off={off}  [{ok}]")
        time.sleep(HOLD_S)

    print("Done. If every readback said OK but the servo never moved, "
          "hummed, or twitched at any point in that sweep, the PCA9685 "
          "chip and I2C link are confirmed fine — the fault is downstream: "
          "check the channel-4 signal/power wiring at the board header, "
          "or swap the shoulder servo onto a known-good channel (e.g. the "
          "wrist's channel 2, temporarily) to tell servo vs. board/wiring "
          "apart.")


if __name__ == '__main__':
    main()
