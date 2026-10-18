# Penn RoboRacer Technical Challenge

Wall-following ROS 2 node for the AutoDRIVE RoboRacer simulator.

- Video: https://youtu.be/LZqekUvg5M8
- Fastest lap: 6.89 s (the recorded laps were 6.89 s, 7.31 s and 7.13 s)

## How it works

**Steering (`tangent_follower.py`).**

I devised my own algorithm for deciding the steering path. Here is a general summary of how it works:

If the path was completely straight, life would be easy; we could simply allow the car to go straight forward. Thus, we seek to calculate the deviation of the path ahead with what a straight path *would* look like as the basis for our steering. We handle the left and right sides separately, then combine the two answers.

**1. Expected wall**

For each side, we fit a straight line through the wall points directly beside the car (beams 75–105°). That line is what the wall would look like if the track stayed straight. Fitting a line, rather than assuming the wall is parallel to the car, means this "expected" wall stays roughly constant regardless of the orientation of the car.

**2. Deviation**

For every lidar point on a given side, we measure how far its hit point is from that straight-line wall, sideways (perpendicular to the wall). This quantifies how much the path deviates from a straight-line path for every lidar beam angle.

- If the deviation is positive, then the wall is farther than expected and curves "away" from us.
- If the deviation is negative, then the wall is closer than expected and closes "inwards".

**3. First significant deviation**

It would be unwise to adjust to every deviation from a straight-line path that we identify. Fluctutions in the wall path comparable to "random noise" should not spur constant adjustments in the car's path.

Instead, we find the first "significant" deviation in front of the car on the given side (i.e., what lidar beam in front of the car has a distance that is first to differ significantly from our expected straight-line pathing?) Here, a simple threshold deviation is used as the basis for signifiance. This is the first run of 4 beams in a row (DEV_COUNT) with the same sign and a deviation above 0.25 m (DEV_THRESHOLD). It marks the first curve ahead that the car must adjust to.

**4. Turn needed**

From this first significant deviation, we need to decide how to steer the car.

If the deviation is positive (wall bows outwards), then the last beam that still matches the expected wall grazes the inside corner of the curve. We aim at that point, setting an incremental angle shift in order to ensure that the car is aligned to that portion of the curve by the time it reaches there. We also shift 0.5 m (CORNER_MARGIN) off the wall so the car doesn’t clip it.

If the deviation is negative (wall closes inwards), we measure the direction of the wall just past the first deviation (12 beams, SEGMENT_BEAMS) and line up with it. The angle we need to incrementally reach is calculated via some basic trig.

If there is no significant deviation, then the wall is straight, so we line up with it over 2 m (ALIGN_LOOKAHEAD).

In each case we know the heading change needed and the distance available. Heading change divided by distance is the curvature, and steering angle = arctan(curvature × wheelbase). The angle is divided by the maximum steering angle (0.52 rad) and clamped to ±1.

**5. Combining sides**

We repeat the steps 1-4 to ensure we have an angle of turn needed to adjust to both the left and right wall of the path. To give the car a single uniform angle of turn, we combine the walls' respective angles as follows:

- Both ≈ 0: average the two.
- One ≈ 0: follow the other side.
- Same direction: follow the inside wall.
- Opposite directions: add them.

**6. Wall push**

Controlling the heading alone can’t prevent a slow drift into a wall on a straight path. We need a checker to ensure we are not within 0.5 m (PUSH_DIST) of the wall-- if we are, then we push the curvature away from it, in proportion to how far inside that distance the wall is (WALL_PUSH).

**Speed and safety (`wall_follow_node.py`).**

I coded the implementation of the algorithm alongside Claude. I directed it and reviewed the legitimacy of its code at every step. The final stage of fine-tuning parameters (e.g. speed) was automated via Claude.

Here are some notes on the implementation below:

- Top speed is 7.0 m/s on straights. Speed drops with steering angle and is capped by a grip limit.
- It plans braking in advance for walls ahead, using the curved path the car will actually drive.
- A steering guard checks the curve about to be driven. If it runs into a wall, it steers the nearest clear curve instead.
- Steering is smoothed and rate-limited, and the car eases away from side walls.
- Lap times and wall contacts are logged.

## Run

```bash
# Terminal 1: bridge to the simulator
ros2 run autodrive_roboracer autodrive_bridge

# Terminal 2: reset the car in the simulator, then start the driver
ros2 run autodrive_roboracer wall_follow_node
```

Reset the car right before starting the node. The simulator's lap timer includes setup time, so the first lap is inflated otherwise.