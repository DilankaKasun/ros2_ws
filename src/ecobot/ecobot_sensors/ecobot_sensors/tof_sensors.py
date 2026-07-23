import json
import time
from collections import deque
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import urllib.request
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False


class TofSensors(Node):
    """Read VL53L0X sensors from ESP32 via UART or HTTP (WiFi)."""

    def __init__(self):
        super().__init__('tof_sensors')

        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('serial_baud', 115200)
        self.declare_parameter('http_url', '')
        self.declare_parameter('http_interval', 0.2)
        self.declare_parameter('publish_topic', '/ecobot/tof_ranges')

        self._port = str(self.get_parameter('serial_port').value)
        self._baud = int(self.get_parameter('serial_baud').value)
        self._url = str(self.get_parameter('http_url').value).strip()
        self._interval = float(self.get_parameter('http_interval').value)
        self._topic = str(self.get_parameter('publish_topic').value)

        self.pub = self.create_publisher(String, self._topic, 10)
        self.ser = None
        self._buf = ''
        self._mode = None
        self._buf_s1 = deque(maxlen=5)
        self._buf_s2 = deque(maxlen=5)

        if self._url and HTTP_AVAILABLE:
            self._mode = 'http'
            self.get_logger().info(f'HTTP mode: {self._url}')
        elif SERIAL_AVAILABLE:
            try:
                self.ser = serial.Serial(
                    self._port, self._baud, timeout=0.05)
                self._mode = 'uart'
                self.get_logger().info(
                    f'UART mode: {self._port} @ {self._baud}')
            except Exception as e:
                self.get_logger().error(f'{self._port}: {e}')
        else:
            self.get_logger().error('No serial or HTTP available')
        self.timer = self.create_timer(
            self._interval if self._mode == 'http' else 0.05,
            self._loop)

    def _loop(self):
        if self._mode == 'uart' and self.ser:
            self._read_uart()
        elif self._mode == 'http':
            self._poll_http()

    def _read_uart(self):
        try:
            raw = self.ser.read(self.ser.in_waiting or 1)
            self._buf += raw.decode('latin-1', errors='ignore')
            while '\n' in self._buf:
                line, self._buf = self._buf.split('\n', 1)
                line = line.strip(chr(0))
                self._parse(line)
        except Exception:
            pass

    def _poll_http(self):
        try:
            req = urllib.request.urlopen(self._url, timeout=1)
            data = req.read().decode('utf-8')
            req.close()
            self._parse(data)
        except Exception:
            pass

    def _parse(self, text):
        idx = text.find('{')
        if idx < 0:
            return
        try:
            data = json.loads(text[idx:])
            s1 = data.get('sensor1')
            s2 = data.get('sensor2')
        except Exception:
            return
        # filter invalid readings and add to rolling buffer
        def _valid(v):
            return v is not None and v > 0 and v < 8000
        # sensor1=right, sensor2=left (swap order)
        if _valid(s2):
            self._buf_s1.append(s2)
        if _valid(s1):
            self._buf_s2.append(s1)
        # publish median of buffer (smoothed)
        def _median(buf):
            if not buf:
                return None
            s = sorted(buf)
            return s[len(s) // 2]
        mm = [_median(self._buf_s1), _median(self._buf_s2)]
        m_vals = [round(v / 1000.0, 3) if v is not None else None
                  for v in mm]
        count = sum(1 for v in mm if v is not None)
        payload = {
            'ranges_mm': mm,
            'ranges_m': m_vals,
            'count': count,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)

    def destroy_node(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TofSensors()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
