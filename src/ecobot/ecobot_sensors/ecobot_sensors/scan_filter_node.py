#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ScanFilterNode(Node):
    """Sanitize /scan for Nav2 (AMCL + costmaps).

    depthimage_to_laserscan can emit NaN / out-of-range / negative ranges on
    invalid depth pixels. AMCL has repeatedly segfaulted (exit code -11) in
    live runs with this data. This node rewrites each message in place:
      - NaN / non-finite  -> inf  (no return = no obstacle; the costmap
        obstacle layer is configured with inf_is_valid: false)
      - range below range_min -> inf (too close to trust, treat as clear)
      - range above range_max -> range_max (free space beyond usable range)
    Publishes to /scan_filtered; Nav2 configs subscribe to that topic."""

    def __init__(self):
        super().__init__('scan_filter')
        self._sub = self.create_subscription(
            LaserScan, '/scan', self._scan_cb, 10)
        self._pub = self.create_publisher(
            LaserScan, '/scan_filtered', 10)
        self.get_logger().info('scan_filter started (/scan -> /scan_filtered)')

    def _scan_cb(self, msg: LaserScan):
        try:
            ranges = np.asarray(msg.ranges, dtype=np.float32)
        except Exception:
            return
        if ranges.size == 0:
            self._pub.publish(msg)
            return

        rmin = msg.range_min
        rmax = msg.range_max
        if rmin <= 0.0 or not math.isfinite(rmin):
            rmin = 0.0
        if not math.isfinite(rmax) or rmax <= 0.0:
            rmax = 100.0

        finite = np.isfinite(ranges)
        # Not finite (NaN / +/-inf) -> inf (no obstacle)
        ranges = np.where(finite, ranges, np.inf)
        # Below usable minimum -> inf (too close / noise, treat as clear)
        ranges = np.where(ranges < rmin, np.inf, ranges)
        # Above usable maximum -> clamp to max (free space)
        ranges = np.where(ranges > rmax, rmax, ranges)

        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = ranges.tolist()
        out.intensities = list(msg.intensities) if msg.intensities else []
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ScanFilterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()