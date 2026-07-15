import os
import threading
from socketserver import TCPServer
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from http.server import HTTPServer, SimpleHTTPRequestHandler


TCPServer.allow_reuse_address = True


class DashboardServer(Node):
    def __init__(self):
        super().__init__('dashboard_server')

        self.declare_parameter('port', 8080)
        try:
            www_dir = os.path.join(
                get_package_share_directory('ecobot_dashboard'), 'www')
        except Exception:
            www_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'www')
        self.declare_parameter('directory', www_dir)

        port = self.get_parameter('port').value
        directory = self.get_parameter('directory').value
        os.makedirs(directory, exist_ok=True)

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)

        self.server = HTTPServer(('', port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.get_logger().info(f'dashboard server on http://0.0.0.0:{port}')

    def destroy_node(self):
        self.server.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DashboardServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
