#!/usr/bin/env python3
"""Wall-following node for the AutoDRIVE RoboRacer simulator (first version).

The idea: We try to decipher the "shape" of the upcoming wall on
each side of the car and figure out how sharply the track is about to bend.
That is done in tangent_follower.py (see there for more info).

This first version only does the steering. Preliminary version has
constant (slow) speed
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

try:
    from autodrive_roboracer import tangent_follower   # inside the ROS package
except ImportError:                                    # running the file on its own (tests)
    import tangent_follower


# Topics (from the challenge sheet / config.py)
LIDAR_TOPIC = '/autodrive/roboracer_1/lidar'
THROTTLE_TOPIC = '/autodrive/roboracer_1/throttle_command'
STEERING_TOPIC = '/autodrive/roboracer_1/steering_command'

# Scan geometry: 1080 beams, -135 deg to +135 deg, 0.25 deg step.
# 0 deg is straight ahead, +90 deg is straight left. Range ~0.06 m to 10 m.
# Commands: throttle and steering in [-1, 1]; +throttle = forward,
# +1 steering = full lock left (~0.52 rad).

MAX_RANGE = 10.0            # m, LiDAR max range
CRUISE_THROTTLE = 0.10      # constant throttle. Speed is roughly 24 * throttle (0.10 -> 2.4 m/s)


def clean_ranges(ranges):
    """Turn the input into numpy array"""
    r = np.array(ranges, dtype=float)
    r = np.nan_to_num(r, nan=0.0, posinf=MAX_RANGE, neginf=0.0)
    return np.clip(r, 0.0, MAX_RANGE)


def compute_command(ranges, angle_min, angle_increment):
    """Input: raw scan in. 
       Output: throttle, steering, debug info
    """
    r = clean_ranges(ranges)
    angles = angle_min + np.arange(len(r)) * angle_increment

    steering, summary = tangent_follower.tangent_steering(r, angles)
    steering = max(-1.0, min(1.0, steering))

    info = {'target_speed': CRUISE_THROTTLE * 24.0, 'corner': abs(steering), 'summary': summary}
    return CRUISE_THROTTLE, steering, info


class WallFollowNode(Node):

    def __init__(self):
        super().__init__('wall_follow_node')
        qos = QoSProfile(depth=1)

        self.scan_sub = self.create_subscription(
            LaserScan, LIDAR_TOPIC, self.scan_callback, qos)
        self.throttle_pub = self.create_publisher(Float32, THROTTLE_TOPIC, qos)
        self.steering_pub = self.create_publisher(Float32, STEERING_TOPIC, qos)

        self.scan_count = 0
        self.get_logger().info('wall_follow_node started, steering = tangent follower')

    def scan_callback(self, scan: LaserScan):
        throttle, steering, info = compute_command(
            scan.ranges, scan.angle_min, scan.angle_increment)
        self.publish_command(throttle, steering)

        # Log about twice a second (scans arrive ~85 times a second).
        self.scan_count += 1
        if self.scan_count % 40 == 0:
            self.get_logger().info(
                f"{info['summary']}  steer {steering:+.2f}  throttle {throttle:.2f}")

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
