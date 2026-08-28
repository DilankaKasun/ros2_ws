#!/usr/bin/env python3
"""Save the current map from slam_toolbox to the maps directory."""

import os
import sys
import rclpy
from rclpy.node import Node
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
import cv2
import numpy as np


def save_map_to_pgm(map_msg, filepath):
    h, w = map_msg.info.height, map_msg.info.width
    data = np.array(map_msg.data, dtype=np.int8).reshape((h, w))
    data = np.clip(data, 0, 100).astype(np.uint8)
    data = 255 - (data * 2.55).astype(np.uint8)
    cv2.imwrite(filepath, data)
    print(f'map saved: {filepath} ({w}x{h})')


def main():
    rclpy.init()
    node = Node('save_map')

    maps_dir = os.path.join(
        os.path.dirname(__file__), '..', 'maps')
    os.makedirs(maps_dir, exist_ok=True)

    client = node.create_client(GetMap, '/map_server/map')
    if not client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error('map_server /map service not available')
        rclpy.shutdown()
        return 1

    future = client.call_async(GetMap.Request())
    rclpy.spin_until_future_complete(node, future)
    if future.result() is None:
        node.get_logger().error('failed to get map')
        rclpy.shutdown()
        return 1

    map_msg = future.result().map
    base = os.path.join(maps_dir, 'default_map')
    save_map_to_pgm(map_msg, base + '.pgm')

    with open(base + '.yaml', 'w') as f:
        f.write(f"""image: default_map.pgm
resolution: {map_msg.info.resolution}
origin: [{map_msg.info.origin.position.x}, {map_msg.info.origin.position.y}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
""")
    print(f'map metadata: {base}.yaml')
    print(f'resolution: {map_msg.info.resolution}')
    print(f'origin: ({map_msg.info.origin.position.x}, {map_msg.info.origin.position.y})')
    print(f'size: {map_msg.info.width}x{map_msg.info.height}')

    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
