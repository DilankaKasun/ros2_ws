import smbus2
import time
import threading


class PCA9685:
    MODE1 = 0x00
    MODE1_RESTART = 0x80
    MODE1_SLEEP = 0x10
    MODE1_AI = 0x20
    MODE2 = 0x01
    MODE2_OUTDRV = 0x04
    PRESCALE = 0xFE
    LED0_ON_L = 0x06
    ALL_LED_ON_L = 0xFA
    ALL_LED_OFF_L = 0xFC

    OSC_CLOCK = 25000000.0
    PWM_RESOLUTION = 4096
    MAX_RETRIES = 3

    def __init__(self, bus: int = 7, address: int = 0x40, freq: float = 50.0):
        self._bus = bus
        self._address = address
        self._lock = threading.Lock()
        self._error_count = 0

        self._i2c = smbus2.SMBus(bus)

        self._write_byte(self.MODE2, self.MODE2_OUTDRV)
        self._write_byte(self.MODE1, self.MODE1_AI)

        time.sleep(0.005)

        mode1 = self._read_byte(self.MODE1)
        mode1 &= ~self.MODE1_SLEEP
        self._write_byte(self.MODE1, mode1)

        time.sleep(0.005)

        self.set_frequency(freq)

    def _read_byte(self, reg):
        for attempt in range(self.MAX_RETRIES):
            try:
                with self._lock:
                    return self._i2c.read_byte_data(self._address, reg)
            except OSError:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(0.001)
        self._error_count += 1
        raise OSError(f'I2C read failed after {self.MAX_RETRIES} retries, reg=0x{reg:02X}')

    def _write_byte(self, reg, value):
        for attempt in range(self.MAX_RETRIES):
            try:
                with self._lock:
                    self._i2c.write_byte_data(self._address, reg, value)
                return
            except OSError:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(0.001)
        self._error_count += 1
        raise OSError(f'I2C write failed after {self.MAX_RETRIES} retries, reg=0x{reg:02X}')

    def set_frequency(self, freq_hz):
        prescale = int((self.OSC_CLOCK / (self.PWM_RESOLUTION * freq_hz)) + 0.5) - 1
        prescale = max(3, min(prescale, 255))

        old_mode = self._read_byte(self.MODE1)
        self._write_byte(self.MODE1, (old_mode & 0x7F) | self.MODE1_SLEEP)

        self._write_byte(self.PRESCALE, prescale)

        wake_mode = old_mode & ~self.MODE1_SLEEP
        self._write_byte(self.MODE1, wake_mode)
        time.sleep(0.005)
        self._write_byte(self.MODE1, wake_mode | self.MODE1_RESTART)

    def set_pwm(self, channel, on, off):
        on = max(0, min(int(on), self.PWM_RESOLUTION - 1))
        off = max(0, min(int(off), self.PWM_RESOLUTION - 1))
        reg = self.LED0_ON_L + 4 * channel

        buf = [
            on & 0xFF,
            (on >> 8) & 0x0F,
            off & 0xFF,
            (off >> 8) & 0x0F,
        ]

        for attempt in range(self.MAX_RETRIES):
            try:
                with self._lock:
                    self._i2c.write_i2c_block_data(
                        self._address, reg, buf)
                return
            except OSError:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(0.001)
        self._error_count += 1
        raise OSError(
            f'I2C write failed after {self.MAX_RETRIES} retries, '
            f'channel={channel}')

    def get_pwm(self, channel):
        """Read back the ON/OFF pulse counts currently set on `channel`."""
        reg = self.LED0_ON_L + 4 * channel
        for attempt in range(self.MAX_RETRIES):
            try:
                with self._lock:
                    on_l = self._i2c.read_byte_data(self._address, reg)
                    on_h = self._i2c.read_byte_data(self._address, reg + 1)
                    off_l = self._i2c.read_byte_data(self._address, reg + 2)
                    off_h = self._i2c.read_byte_data(self._address, reg + 3)
                return (on_h << 8) | on_l, (off_h << 8) | off_l
            except OSError:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(0.001)
        self._error_count += 1
        raise OSError(
            f'I2C read failed after {self.MAX_RETRIES} retries, '
            f'channel={channel}')

    def set_pulse(self, channel, pulse_12bit):
        pulse_12bit = max(0, min(int(pulse_12bit), self.PWM_RESOLUTION - 1))
        self.set_pwm(channel, 0, pulse_12bit)

    def set_angle(self, channel, angle_deg,
                  pulse_min, pulse_max, servo_range):
        """Drive `channel` to `angle_deg`.

        `servo_range` is the servo's full mechanical travel, not the joint
        limit — passing a narrower joint limit here stretches the pulse span
        over too small an arc and the joint overshoots proportionally.
        """
        servo_range = float(servo_range)
        if servo_range <= 0:
            return
        angle_deg = max(0.0, min(float(angle_deg), servo_range))
        pulse = int(
            pulse_min + (angle_deg / servo_range) * (pulse_max - pulse_min)
        )
        pulse = max(0, min(pulse, self.PWM_RESOLUTION - 1))
        self.set_pwm(channel, 0, pulse)

    def set_all_angles(self, angles, servo_configs):
        for angle, cfg in zip(angles, servo_configs):
            self.set_angle(
                cfg['channel'], angle,
                cfg['pulse_min'], cfg['pulse_max'], cfg['servo_range']
            )

    def disable(self):
        for attempt in range(self.MAX_RETRIES):
            try:
                old_mode = self._read_byte(self.MODE1)
                self._write_byte(self.MODE1, old_mode | self.MODE1_SLEEP)
                return
            except OSError:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(0.001)

    def enable(self):
        for attempt in range(self.MAX_RETRIES):
            try:
                old_mode = self._read_byte(self.MODE1)
                wake_mode = old_mode & ~self.MODE1_SLEEP
                self._write_byte(self.MODE1, wake_mode)
                time.sleep(0.005)
                self._write_byte(self.MODE1, wake_mode | self.MODE1_RESTART)
                return
            except OSError:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(0.001)
