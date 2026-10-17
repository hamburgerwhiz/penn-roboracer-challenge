#!/usr/bin/env python3
"""Wall-following node for the AutoDRIVE RoboRacer simulator."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


# Topics (from the challenge sheet / config.py)
LIDAR_TOPIC = '/autodrive/roboracer_1/lidar'
THROTTLE_TOPIC = '/autodrive/roboracer_1/throttle_command'
STEERING_TOPIC = '/autodrive/roboracer_1/steering_command'

# Scan geometry: 1080 beams, -135 deg to +135 deg, 0.25 deg step.
# 0 deg is straight ahead, +90 deg is straight left. Range ~0.06 m to 10 m.
# Commands: throttle and steering in [-1, 1]; +throttle = forward,
# +1 steering = full lock left (~0.52 rad).


class WallFollowNode(Node):

    def __init__(self):
        super().__init__('wall_follow_node')
        qos = QoSProfile(depth=1)

        self.scan_sub = self.create_subscription(
            LaserScan, LIDAR_TOPIC, self.scan_callback, qos)
        self.throttle_pub = self.create_publisher(Float32, THROTTLE_TOPIC, qos)
        self.steering_pub = self.create_publisher(Float32, STEERING_TOPIC, qos)

        self.get_logger().info('wall_follow_node started')

    def scan_callback(self, scan: LaserScan):
        # TODO: your wall-following logic goes here.
        pass

    def publish_command(self, throttle: float, steering: float):
        """Clamp to [-1, 1] and publish throttle and steering."""
        self.throttle_pub.publish(Float32(data=float(max(-1.0, min(1.0, throttle)))))
        self.steering_pub.publish(Float32(data=float(max(-1.0, min(1.0, steering)))))


def main(args=None):
    rclpy.init(args=args, signal_handler_options=rclpy.SignalHandlerOptions.NO)    
    node = WallFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_command(0.0, 0.0)  # stop the car on exit
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
