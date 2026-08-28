#!/usr/bin/env python3
"""Direct PCA9685 test sweeping each arm servo through safe joint ranges."""

import math
import time
from smbus2 import SMBus

I2C_BUS = 7
PCA9685_ADDR = 0x40

# Registers
MODE1 = 0x00
MODE1_AI = 0x20  # auto-increment: required for multi-byte block writes to advance
                  # past ON_L, otherwise every byte in set_pwm's block write lands
                  # on the same register and OFF_L/OFF_H (the actual pulse) never updates
PRESCALE = 0xFE
LED0_ON_L = 0x06

# Joint definitions from servo_config
JOINTS = [
    {'name': 'Base',     'channel': 0, 'min': 20, 'max': 180, 'home': 107, 'range': 270},
    {'name': 'Shoulder', 'channel': 3, 'min': 10, 'max': 110, 'home': 125, 'range': 180},
    {'name': 'Elbow',    'channel': 1, 'min': 10, 'max': 170, 'home': 180, 'range': 180},
    {'name': 'Wrist',    'channel': 2, 'min': 10, 'max': 170, 'home': 45,  'range': 180},
]

class PCA9685Driver:
    def __init__(self, bus_num=I2C_BUS, address=PCA9685_ADDR):
        self.bus = SMBus(bus_num)
        self.address = address
        self.init_pwm(50)  # 50 Hz standard servo frequency

    def init_pwm(self, freq_hz=50):
        self.bus.write_byte_data(self.address, MODE1, MODE1_AI)
        time.sleep(0.005)
        prescaleval = int(math.floor(25000000.0 / (4096.0 * freq_hz) - 1.0 + 0.5))
        oldmode = self.bus.read_byte_data(self.address, MODE1)
        newmode = (oldmode & 0x7F) | 0x10  # Sleep mode
        self.bus.write_byte_data(self.address, MODE1, newmode)
        self.bus.write_byte_data(self.address, PRESCALE, prescaleval)
        self.bus.write_byte_data(self.address, MODE1, oldmode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, MODE1, oldmode | 0x80)

    def set_pwm(self, channel, on, off):
        base_reg = LED0_ON_L + 4 * channel
        data = [on & 0xFF, (on >> 8) & 0xFF, off & 0xFF, (off >> 8) & 0xFF]
        self.bus.write_i2c_block_data(self.address, base_reg, data)

    def set_angle(self, joint, angle):
        angle = max(joint['min'], min(joint['max'], angle))
        pulse_len = 150 + int((angle / joint['range']) * (600 - 150))
        self.set_pwm(joint['channel'], 0, pulse_len)


def run_servo_test():
    pca = PCA9685Driver()
    print("Testing PCA9685 Arm Servos...")

    # 1. Move all to home position
    print("1. Moving all joints to HOME positions...")
    for j in JOINTS:
        pca.set_angle(j, j['home'])
        time.sleep(0.3)
    time.sleep(1.0)

    # 2. Test each servo sequentially
    for j in JOINTS:
        print(f"2. Testing {j['name']} (Channel {j['channel']})...")
        for angle in range(int(j['min']), int(j['max']), 5):
            pca.set_angle(j, angle)
            time.sleep(0.03)
        for angle in range(int(j['max']), int(j['min']), -5):
            pca.set_angle(j, angle)
            time.sleep(0.03)
        pca.set_angle(j, j['home'])
        time.sleep(0.5)

    print("Servo test completed successfully!")

if __name__ == '__main__':
    run_servo_test()