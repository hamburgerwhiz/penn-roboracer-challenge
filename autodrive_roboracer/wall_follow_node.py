#!/usr/bin/env python3
"""Wall-following node for the AutoDRIVE RoboRacer simulator.

The idea: We try to decipher the "shape" of the upcoming wall on
each side of the car and figure out how sharply the track is about to bend.
That is done in tangent_follower.py (see there for more info).

This sixth version adds a preventative mechanism: if the curve about to be driven runs
into a wall, then drive the nearest clear curve instead. This is motivated by the 
bottom right of the curve, which is a common place to hit the wall on a corner exit. 
For the life of me I can't figure out why the tangent follower wants to drive into that wall,
so I'll slap a bandaid on the symptom for the time being...
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Int32

try:
    from autodrive_roboracer import tangent_follower   # inside the ROS package
except ImportError:                                    # running the file on its own (tests)
    import tangent_follower


# Topics (from the challenge sheet / config.py)
LIDAR_TOPIC = '/autodrive/roboracer_1/lidar'
THROTTLE_TOPIC = '/autodrive/roboracer_1/throttle_command'
STEERING_TOPIC = '/autodrive/roboracer_1/steering_command'
ODOM_TOPIC = '/autodrive/roboracer_1/odom'
LAST_LAP_TOPIC = '/autodrive/roboracer_1/last_lap_time'
BEST_LAP_TOPIC = '/autodrive/roboracer_1/best_lap_time'
COLLISION_TOPIC = '/autodrive/roboracer_1/collision_count'

# Scan geometry: 1080 beams, -135 deg to +135 deg, 0.25 deg step.
# 0 deg is straight ahead, +90 deg is straight left. Range ~0.06 m to 10 m.
# Commands: throttle and steering in [-1, 1]; +throttle = forward,
# +1 steering = full lock left (~0.52 rad).

MAX_RANGE = 10.0            # m, LiDAR max range

SIDE_HARD = 0.30            # m, wall this close beside us: no steering toward it at all
SIDE_SOFT = 0.45            # m, wall farther than this: steer freely. In between, steering
                            # toward that wall is eased off smoothly (no sudden snap).
SIDE_MIN_ANGLE = math.radians(70)     # "beside the car" = 70..135 deg on each side
SIDE_MAX_ANGLE = math.radians(135)

MAX_STEER_RAD = 0.52        # rad, steering command +1 = full lock left

# Speed. From the first test run, throttle acts like a speed setting:
# speed (m/s) is roughly SPEED_PER_THROTTLE * throttle (0.12 -> 3.0 m/s).
SPEED_PER_THROTTLE = 24.0
MAX_SPEED = 7.0             # m/s, top speed on straights
CORNER_SPEED = 1.8          # m/s, speed at full steering lock
MIN_SPEED = 1.0             # m/s, never go slower than this
BRAKE_DECEL = 4.0           # m/s^2, how hard we plan to brake (smaller = brake earlier)
KP_SPEED = 0.04             # extra throttle per m/s of speed shortfall (0 = off). This is what
                            # makes the car accelerate hard when it is slower than it wants to be.
ACCEL_LIMIT = 6.0           # m/s^2, how fast the TARGET speed may rise (braking is instant).
                            # Stops the target from flickering up and down and jerking the throttle.
                            # Smaller = smoother but slower out of corners.

# Corner memory: after a sharp turn, keep driving at corner speed for a moment
# instead of flooring it the instant the steering relaxes (matters in S-bends).
CORNER_RELEASE = 3.0        # per second. How fast the remembered sharpness (0..1) fades.
                            # 3.0 = from full lock to "straight" in about 0.33 s. Smaller = more cautious.

THROTTLE_MAX = 0.60         # never command more throttle than this
THROTTLE_MIN = 0.0          # never command less (negative might mean reverse; untested)
ACTIVE_BRAKE = 0.4          # max NEGATIVE throttle used to brake when we're too fast (0 = coast
                            # only, as before). Only used above BRAKE_MIN_SPEED, so it can never
                            # make the car reverse.
BRAKE_GAIN = 0.3            # negative throttle per m/s we are over the target speed
BRAKE_MIN_SPEED = 1.5       # m/s, never brake actively below this
STOP_MARGIN = 1.0           # m, plan to be slowed down this far before the wall ahead
CORRIDOR_HALF_WIDTH = 0.18  # m, "straight ahead" = a strip about as wide as the car itself.
                            # Wider than the car and walls beside us count as "ahead".
ARC_BRAKE = 1.0             # 0 = brake for the wall STRAIGHT ahead (old behaviour).
                            # 1 = brake for the first wall along the CURVED path the car is
                            #     actually steering (much faster out of corners: on a corner exit
                            #     the nose points at the outside wall, but the car is turning
                            #     away from it). In between = a blend.
LAT_ACCEL = 8.0             # m/s^2, sideways grip we trust in a turn. Corner speed limit is
                            # v = sqrt(LAT_ACCEL / curvature): wide loops fast, tight turns slow.
                            # Car slides into walls in turns -> lower (6). Too timid -> raise (10).
                            # 0 = off.
TURN_LOOKAHEAD = 2.0        # m. Anticipates TIGHTENING corners: finds the gentlest turn that
                            # still has this much clear road, and slows to the speed that turn
                            # allows (sqrt(LAT_ACCEL / curvature)). Bigger = slows earlier. 0 = off.
TURN_HALF_WIDTH = 0.25      # m, clearance used for that check (a bit wider than the car)
ARC_GUARD = 1.0             # 1 = steering guard: if the curve we're about to drive hits a wall
                            # within GUARD_MIN + GUARD_TIME * speed meters, steer the nearest
                            # curve that is clear instead (catches both turning too late and
                            # over-turning past an apex). 0 = off.
GUARD_MIN = 0.6             # m
GUARD_TIME = 0.3            # s
GUARD_HALF_WIDTH = 0.22     # m, clearance the guard demands on each side of the path
LIDAR_X = 0.27              # m, LiDAR sits this far ahead of the rear axle (turning centre)
WHEELBASE = 0.33            # m

# Steering smoothing (the scan arrives ~85 times a second, so a little filtering is invisible).
STEER_SMOOTH = 0.4          # 0 = no filtering, 0.9 = very heavy. Fraction of the old value kept per scan.
STEER_RATE = 20.0           # max steering change per second (a full-lock swing, 2.0, takes 2/20 = 0.1 s)


def clean_ranges(ranges):
    """Turn the input into numpy array"""
    r = np.array(ranges, dtype=float)
    r = np.nan_to_num(r, nan=0.0, posinf=MAX_RANGE, neginf=0.0)
    return np.clip(r, 0.0, MAX_RANGE)


def arc_free_distance(r, angles, steer, half_width=None):
    """How far the car can drive along the circle it is steering (steer in [-1, 1],
    held constant) before a wall point comes within CORRIDOR_HALF_WIDTH of its path.
    Measured as distance travelled along the arc, in meters."""
    hw = CORRIDOR_HALF_WIDTH if half_width is None else half_width
    kappa = math.tan(steer * MAX_STEER_RAD) / WHEELBASE     # 1/m, + = left
    valid = r > 0.06
    x, y = r * np.cos(angles), r * np.sin(angles)
    if abs(kappa) < 0.02:                                  # basically straight
        hit = valid & (x > 0) & (np.abs(y) < hw)
        return float(x[hit].min()) if hit.any() else MAX_RANGE
    R = 1.0 / kappa                                        # signed turn radius (rear axle)
    cx, cy = -LIDAR_X, R                                   # turning centre, LiDAR frame
    R_l = math.hypot(LIDAR_X, R)                           # radius the LiDAR point travels on
    dx, dy = x - cx, y - cy
    d = np.hypot(dx, dy)
    near_path = valid & (np.abs(d - R_l) < hw)
    # Angle swept around the centre from the LiDAR's position to each point,
    # in the direction the car is turning.
    a0 = math.atan2(0.0 - cy, 0.0 - cx)
    sweep = (np.arctan2(dy, dx) - a0) * math.copysign(1.0, kappa)
    sweep = np.mod(sweep, 2 * math.pi)
    max_sweep = min(math.pi, MAX_RANGE / R_l)              # look half a circle ahead at most
    hit = near_path & (sweep > 0) & (sweep < max_sweep)
    if not hit.any():
        return max_sweep * R_l
    return float((sweep[hit] * R_l).min())


TURN_FAN = np.linspace(-1.0, 1.0, 21)    # steering values checked by the turn-ahead limit
GUARD_FAN = np.linspace(-1.0, 1.0, 21)   # steering values the guard may choose from


def side_factor(min_dist):
    """0 when a wall is very close beside us, 1 when it is comfortably far."""
    return max(0.0, min(1.0, (min_dist - SIDE_HARD) / (SIDE_SOFT - SIDE_HARD)))


def compute_command(ranges, angle_min, angle_increment, speed=None,
                    prev_steer=None, prev_target=None, dt=0.012, prev_corner=None):
    """Raw scan in, (throttle, steering, debug info) out.
    `speed`       current speed in m/s (None = no speed feedback)
    `prev_steer`  steering we sent last scan (None = no smoothing)
    `prev_target` target speed from last scan (None = no ramp limit)
    `dt`          seconds since the last scan
    `prev_corner` remembered turn sharpness from last scan"""
    r = clean_ranges(ranges)
    angles = angle_min + np.arange(len(r)) * angle_increment

    # How far is the nearest wall straight ahead, inside a strip as wide as the car?
    # (Used for braking below.)
    fwd_mask = np.abs(angles) < math.radians(45)
    xs = r[fwd_mask] * np.cos(angles[fwd_mask])
    ys = r[fwd_mask] * np.sin(angles[fwd_mask])
    in_strip = (np.abs(ys) < CORRIDOR_HALF_WIDTH) & (r[fwd_mask] > 0.0)
    fwd_dist = float(xs[in_strip].min()) if in_strip.any() else MAX_RANGE

    # The tangent follower (see tangent_follower.py) decides the steering.
    steering, summary = tangent_follower.tangent_steering(r, angles)
    steering = max(-1.0, min(1.0, steering))

    # Steering guard: is the curve we're about to drive clear for long enough?
    if ARC_GUARD > 0:
        need = GUARD_MIN + GUARD_TIME * (speed or 0.0)
        if arc_free_distance(r, angles, steering, GUARD_HALF_WIDTH) < need:
            free = np.array([arc_free_distance(r, angles, s_, GUARD_HALF_WIDTH) for s_ in GUARD_FAN])
            ok = free >= need
            if ok.any():       # the clear curve closest to what the algorithm wanted
                pick = GUARD_FAN[ok][np.argmin(np.abs(GUARD_FAN[ok] - steering))]
            else:              # nothing is clear: take the one with the most room
                pick = GUARD_FAN[int(np.argmax(free))]
            summary += f'  GUARD {steering:+.2f}->{pick:+.2f}'
            steering = float(pick)

    # How sharp the turn really is, before any easing or smoothing. Speed uses this.
    demand = max(-1.0, min(1.0, steering))

    # Side safety: ease off steering toward a wall that is close beside us.
    # (Smooth transition instead of snapping the wheels straight.)
    left = r[(angles >= SIDE_MIN_ANGLE) & (angles <= SIDE_MAX_ANGLE)]
    right = r[(angles <= -SIDE_MIN_ANGLE) & (angles >= -SIDE_MAX_ANGLE)]
    blocked = ''
    if steering > 0 and left.size:
        f = side_factor(float(left.min()))
        if f < 1.0:
            steering, blocked = steering * f, 'left'
    elif steering < 0 and right.size:
        f = side_factor(float(right.min()))
        if f < 1.0:
            steering, blocked = steering * f, 'right'
    steering = max(-1.0, min(1.0, steering))

    # Smooth the steering: blend with last scan's value, and limit how fast it can change.
    if prev_steer is not None:
        smoothed = prev_steer + (steering - prev_steer) * (1.0 - STEER_SMOOTH)
        step = max(-STEER_RATE * dt, min(STEER_RATE * dt, smoothed - prev_steer))
        steering = prev_steer + step

    # Speed limit 1 (braking): pick a speed we can brake from in time before the
    # wall straight ahead (fwd_dist, measured above): v = sqrt(2 * decel * distance).
    # With ARC_BRAKE, also measure along the curves the car is about to drive: any
    # steering between what the wheels do now and what the algorithm asks for. If
    # one of those curves is clearer than straight ahead (typical on a corner exit,
    # nose pointing at the outside wall while turning away from it), brake for that.
    # Never less room than straight ahead, so this is never MORE cautious than before.
    if ARC_BRAKE > 0:
        lo, hi = sorted((steering, demand))
        arc_dist = max(arc_free_distance(r, angles, s)
                       for s in np.linspace(lo, hi, 5))
        arc_dist = max(arc_dist, fwd_dist)
        brake_dist = fwd_dist + ARC_BRAKE * (arc_dist - fwd_dist)
    else:
        arc_dist = brake_dist = fwd_dist
    v_brake = math.sqrt(2.0 * BRAKE_DECEL * max(0.0, brake_dist - STOP_MARGIN))

    # Speed limit 2 (cornering): the sharper the turn, the slower we go.
    # Uses the real turn sharpness (not the eased steering), and remembers a
    # recent sharp turn for a moment so the car doesn't surge mid-corner.
    corner = abs(demand)
    if prev_corner is not None:
        corner = max(corner, prev_corner - CORNER_RELEASE * dt)
    v_corner = MAX_SPEED + (CORNER_SPEED - MAX_SPEED) * corner
    # Grip limit from the actual curvature of the turn: v^2 * curvature <= LAT_ACCEL.
    # (Uses the sharper of the turn asked for and the turn the wheels are making.)
    kappa = math.tan(max(corner, abs(steering)) * MAX_STEER_RAD) / WHEELBASE
    if LAT_ACCEL > 0 and kappa > 1e-3:
        v_corner = min(v_corner, math.sqrt(LAT_ACCEL / kappa))

    # Speed limit 3 (turn ahead): of all the curves the car could steer, which is the
    # gentlest one with TURN_LOOKAHEAD of clear road? The car will need at least that
    # much turn soon, so don't arrive faster than that turn allows.
    v_turn = MAX_SPEED
    if TURN_LOOKAHEAD > 0 and LAT_ACCEL > 0:
        need_k = math.tan(MAX_STEER_RAD) / WHEELBASE       # nothing clear: full lock
        for s_ in TURN_FAN:
            k_ = abs(math.tan(s_ * MAX_STEER_RAD) / WHEELBASE)
            if k_ < need_k and arc_free_distance(r, angles, s_, TURN_HALF_WIDTH) >= TURN_LOOKAHEAD:
                need_k = k_
        if need_k > 1e-3:
            v_turn = math.sqrt(LAT_ACCEL / need_k)
    v_corner = min(v_corner, v_turn)

    target_speed = max(MIN_SPEED, min(MAX_SPEED, v_brake, v_corner))
    # Which limit is holding the speed down (shown in the log).
    limit = ('top speed' if target_speed >= MAX_SPEED - 1e-6 else
             'wall ahead' if v_brake <= v_corner else
             'turn ahead' if v_turn <= v_corner + 1e-9 and v_turn < MAX_SPEED else 'cornering')
    # The target may drop instantly (braking) but can only rise at ACCEL_LIMIT.
    if prev_target is not None and target_speed > prev_target + ACCEL_LIMIT * dt:
        target_speed = prev_target + ACCEL_LIMIT * dt
        limit = 'accel ramp'
    # Throttle = the throttle that should hold the target speed, plus a push
    # proportional to how far we are below it (or a pull-back if we are above it).
    throttle = target_speed / SPEED_PER_THROTTLE
    if speed is not None:
        throttle += KP_SPEED * (target_speed - speed)
    throttle = max(THROTTLE_MIN, min(THROTTLE_MAX, throttle))

    # Active braking: coasting alone sheds speed too slowly before tight corners.
    over = (speed - target_speed) if speed is not None else 0.0
    if ACTIVE_BRAKE > 0 and over > 0.5 and speed > BRAKE_MIN_SPEED:
        throttle = -min(ACTIVE_BRAKE, BRAKE_GAIN * (over - 0.2))
        limit += ', braking'

    info = {
        'fwd_dist': fwd_dist,
        'arc_dist': arc_dist,
        'limit': limit,
        'target_speed': target_speed,
        'corner': corner,
        'blocked': blocked,
        'summary': summary,
    }
    return throttle, steering, info


class WallFollowNode(Node):

    def __init__(self):
        super().__init__('wall_follow_node')
        qos = QoSProfile(depth=1)

        self.scan_sub = self.create_subscription(
            LaserScan, LIDAR_TOPIC, self.scan_callback, qos)
        self.throttle_pub = self.create_publisher(Float32, THROTTLE_TOPIC, qos)
        self.steering_pub = self.create_publisher(Float32, STEERING_TOPIC, qos)

        # Extra inputs, only used for logging: speed, lap times, wall contacts.
        self.speed = 0.0
        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, qos)
        self.create_subscription(Float32, LAST_LAP_TOPIC, self.last_lap_callback, qos)
        self.create_subscription(Float32, BEST_LAP_TOPIC, self.best_lap_callback, qos)
        self.create_subscription(Int32, COLLISION_TOPIC, self.collision_callback, qos)
        self.last_lap = None
        self.best_lap = None
        self.collisions = None
        # Lap bookkeeping, so different versions can be compared fairly.
        self.hit_this_lap = False
        self.clean_laps = []        # times of laps with no wall contact
        self.clean_streak = 0       # clean laps in a row (the challenge needs 3)

        # State carried from one scan to the next (for smoothing).
        self.prev_steer = None
        self.prev_target = None
        self.prev_time = None
        self.prev_corner = None

        self.scan_count = 0
        self.get_logger().info('wall_follow_node started, steering = tangent follower')

    def scan_callback(self, scan: LaserScan):
        now = time.monotonic()
        dt = 0.012 if self.prev_time is None else max(0.004, min(0.05, now - self.prev_time))
        self.prev_time = now

        throttle, steering, info = compute_command(
            scan.ranges, scan.angle_min, scan.angle_increment, self.speed,
            self.prev_steer, self.prev_target, dt, self.prev_corner)
        self.prev_corner = info['corner']
        self.prev_steer = steering
        self.prev_target = info['target_speed']
        self.publish_command(throttle, steering)

        # Log about twice a second (scans arrive ~85 times a second).
        self.scan_count += 1
        if self.scan_count % 40 == 0:
            self.get_logger().info(
                f"{info['summary']}  "
                f"ahead {info['fwd_dist']:4.1f} m (path {info['arc_dist']:4.1f})  "
                f"steer {steering:+.2f}  throttle {throttle:.2f}  "
                f"want {info['target_speed']:3.1f} / have {self.speed:3.1f} m/s "
                f"[{info['limit']}]"
                + (f"  (blocked {info['blocked']})" if info['blocked'] else ''))

    def odom_callback(self, msg: Odometry):
        v = msg.twist.twist.linear
        self.speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def last_lap_callback(self, msg: Float32):
        if self.last_lap is None:
            self.last_lap = msg.data
            return
        if msg.data > 0 and msg.data != self.last_lap:
            self.last_lap = msg.data
            if self.hit_this_lap:
                self.clean_streak = 0
                verdict = 'HIT A WALL (does not count)'
            else:
                self.clean_laps.append(msg.data)
                self.clean_streak += 1
                verdict = 'clean'
            self.hit_this_lap = False
            n = len(self.clean_laps)
            stats = ''
            if n:
                # Median of the last 5 clean laps: a typical lap time that one odd lap
                # (like a slow first lap from a standing start) can't distort.
                recent = sorted(self.clean_laps[-5:])
                typical = recent[len(recent) // 2] if len(recent) % 2 else \
                    0.5 * (recent[len(recent) // 2 - 1] + recent[len(recent) // 2])
                stats = (f' | clean laps: {n}, typical (median of last 5) {typical:.2f} s,'
                         f' best {min(self.clean_laps):.2f} s')
            self.get_logger().info(
                f'LAP {msg.data:.2f} s ({verdict}) | clean in a row: {self.clean_streak}{stats}')

    def best_lap_callback(self, msg: Float32):
        if msg.data > 0 and msg.data != self.best_lap:
            self.best_lap = msg.data
            self.get_logger().info(f'NEW BEST LAP: {msg.data:.2f} s')

    def collision_callback(self, msg: Int32):
        if self.collisions is not None and msg.data > self.collisions:
            self.hit_this_lap = True
            self.get_logger().warn(f'COLLISION! total = {msg.data}')
        self.collisions = msg.data

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
