#!/usr/bin/env python3
"""
Summary of my approach (updated manually for reader's ease of understanding):

For EACH side of the car (left and right) separately:

  1. We first imagine the expected wall ("e"). This can be visualized by 
     fiting a straight line through the wall points directly parallel
     beside the car (i.e. beams 75..105 deg on that side). That line is what 
     the wall would look like if the track stayed "straight". Fitting a line (rather
     than assuming the wall is parallel to the car) means a car that is angled
     to the wall doesn't see a fake curve.

  2. Calculate the deviation of reality wall vs expectation wall ("r - e"). 
     Walk the beams from 90 deg (beside the car) toward straight ahead. 
     For each beam, measure how far its hit point is from the expected line, SIDEWAYS 
    (perpendicular to the wall), in meters. Measuring sideways keeps one threshold 
    valid at every angle.
         dev > 0 : wall is farther than expected ("falls away", r > e)
         dev < 0 : wall is closer than expected ("closes in", r < e)

  3. First significant deviation = DEV_COUNT beams in a row with |dev| above
     DEV_THRESHOLD and the same sign. This decides what the first significant
     curve or deviation in the path ahead that the car must adjust to.

  4. Figuring out the turn needed:
       Case 1 (veers away, r > e): the track bends outwards. The last
         beam that still matched the expected wall grazes the inside corner,
         so its direction is the tangent to approach. Aim at that corner point,
         shifted some arb. "CORNER_MARGIN" var off the wall so we don't clip it.
       Case 2 (closes in, r < e): the wall turns in front of us. Measure the
         direction of the wall just past the first deviation (a short segment
         of SEGMENT_BEAMS beams) and line up with it.
       Case 3 (no deviation): the wall is straight. Continue as normal.
     In each case we know (heading change needed) and (distance until we
     must have it). heading / distance = turn per meter = CURVATURE, and
     curvature maps straight to a steering angle: angle = atan(curvature * L),
     L = wheelbase. So no speed is needed to convert it.

Combine the two sides (positive = turn LEFT, the car's convention):
  * both ~0                     -> nearly straight (average of the two)
  * same direction, or one is ~0 -> follow the INSIDE wall's answer
                                    (shorter path), or the non-zero one
  * opposite directions         -> track widening/narrowing: add them

"""

import math

import numpy as np

WHEELBASE = 0.33            # m, from the bridge's wheel positions
MAX_STEER_RAD = 0.52        # rad, steering command +1 = full lock left
MAX_RANGE = 10.0            # m

# ---------------------------------------------------------------------------
# Tuning knobs (all live-tunable: ros2 param set /wall_follow_node t_<name> <value>)
# ---------------------------------------------------------------------------
FIT_MIN_DEG = 75.0          # beams used to fit the nearby wall, on each side
FIT_MAX_DEG = 105.0
SCAN_MIN_DEG = 5.0          # look for deviations from 90 deg down to this angle
DEV_THRESHOLD = 0.25        # m, sideways deviation that counts as "significant"
DEV_COUNT = 4               # this many beams in a row (ignores single odd readings)
SEGMENT_BEAMS = 12          # case B: beams used to measure the new wall direction
CORNER_MARGIN = 0.50        # m, case A: aim this far off the inside corner
ALIGN_LOOKAHEAD = 2.0       # m, case C: line up with a straight wall over this distance
ZERO_CURV = 0.05            # 1/m, smaller curvatures count as "zero" when combining
# ---------------------------------------------------------------------------

def analyse_side(r, angles, side):
    """Look at one side of the car. side = +1 for left, -1 for right.
    Returns a dict with the curvature this side asks for, or None if this
    side has no wall to work with."""
    deg = np.degrees(angles) * side            # positive = on this side
    xs = r * np.cos(angles)
    ys = r * np.sin(angles)
    valid = (r > 0.06) & (r < MAX_RANGE - 0.01)

    # 1. Expected wall. i.e. straightline through the points beside the car.
    fit = valid & (deg >= FIT_MIN_DEG) & (deg <= FIT_MAX_DEG)
    if fit.sum() < 10:
        return None
    b, a = np.polyfit(xs[fit], ys[fit], 1)
    norm = math.sqrt(1.0 + b * b)
    psi = math.atan(b)  # wall direction vs car heading
    nx, ny = side * math.sin(psi), -side * math.cos(psi)

    # 2. Walk from 90 deg toward straight ahead.
    order = np.where((deg <= 90.0) & (deg >= SCAN_MIN_DEG))[0]
    order = order[np.argsort(-deg[order])]     # 90 deg first
    ro, ao = r[order], angles[order]
    px, py = xs[order], ys[order]

    dev = side * (py - (a + b * px)) / norm    # + falls away, - closes in
    # Beams with no return: only a "falls away" if the expected wall was in range.
    c, s = np.cos(ao), np.sin(ao)
    denom = s - b * c
    with np.errstate(divide='ignore', invalid='ignore'):
        expected = np.where(np.abs(denom) > 1e-9, a / denom, np.inf)
    no_return = ro >= MAX_RANGE - 0.01
    expected_in_range = (expected > 0) & (expected < MAX_RANGE - 0.5)
    dev = np.where(no_return, np.where(expected_in_range, 10.0, 0.0), dev)
    dev = np.where(ro <= 0.06, 0.0, dev)       # junk reading: ignore
    # Only beams that could reach THIS side's expected wall mean anything. Near
    # straight ahead (or when the car is angled), beams pass the line's range or
    # hit the OTHER side's wall, which is not a deviation of this wall.
    dev = np.where(expected_in_range | no_return, dev, 0.0)
    dev = np.where(~no_return & (side * py < 0), 0.0, dev)

    # 3. First run of DEV_COUNT same-sign significant deviations.
    sig = np.sign(dev) * (np.abs(dev) > DEV_THRESHOLD)
    count = int(DEV_COUNT)
    k_found = None
    for k in range(len(sig) - count + 1):
        if sig[k] != 0 and np.all(sig[k:k + count] == sig[k]):
            k_found = k
            break

    if k_found is None:
        # Case C: straight wall. Line up with it.
        kappa = psi / ALIGN_LOOKAHEAD
        case, aim = 'C', None
    elif sig[k_found] > 0:
        # Case A: falls away. Corner = last beam that still matched.
        j = max(k_found - 1, 0)
        if valid[order[j]]:
            cx, cy = px[j], py[j]
        else:                                  # use the expected wall point instead
            t = expected[j] if np.isfinite(expected[j]) else 1.0
            cx, cy = t * c[j], t * s[j]
        ax, ay = cx + CORNER_MARGIN * nx, cy + CORNER_MARGIN * ny
        heading = math.atan2(ay, ax)
        dist = max(0.3, math.hypot(ax, ay))
        kappa = heading / dist
        case, aim = 'A', (ax, ay)
    else:
        # Case B: closes in. Direction of the wall just past the bend.
        seg = [m for m in range(k_found, min(len(order), k_found + int(SEGMENT_BEAMS)))
               if valid[order[m]]]
        if len(seg) >= 2:
            dx, dy = px[seg[-1]] - px[seg[0]], py[seg[-1]] - py[seg[0]]
            heading = math.atan2(dy, dx)
            heading = max(-math.pi / 2, min(math.pi / 2, heading))
        else:
            heading = psi
        dist = max(0.3, math.hypot(px[k_found], py[k_found]))
        kappa = heading / dist
        case, aim = 'B', (px[k_found], py[k_found])

    # Closest wall point on this side (for the push-away safety).
    near = valid & (deg >= 45.0) & (deg <= 135.0)
    min_dist = float(np.min(np.abs(ys[near]))) if near.any() else MAX_RANGE

    return {'kappa': float(kappa), 'case': case, 'psi': psi,
            'wall_dist': abs(a) / norm, 'min_dist': min_dist, 'aim': aim}


def combine(left, right):
    """Positive = turn left."""
    kl = left['kappa'] if left else 0.0
    kr = right['kappa'] if right else 0.0
    zl, zr = abs(kl) < ZERO_CURV, abs(kr) < ZERO_CURV

    if zl and zr:
        return 0.5 * (kl + kr), 'straight'
    if zl or zr:
        return (kr if zl else kl), ('right side' if zl else 'left side')
    if (kl > 0) == (kr > 0):
        # Same direction: follow the inside wall (left wall for a left turn).
        return (kl, 'inside=left') if kl > 0 else (kr, 'inside=right')
    return kl + kr, 'sum'


def tangent_steering(r, angles):
    """Return (steering demand in [-1, 1], one-line summary for the log)."""
    left = analyse_side(r, angles, +1)
    right = analyse_side(r, angles, -1)
    kappa, rule = combine(left, right)

    steer_angle = math.atan(kappa * WHEELBASE)
    demand = max(-1.0, min(1.0, steer_angle / MAX_STEER_RAD))

    def fmt(name, info):
        if not info:
            return f'{name}: none     '
        return f"{name}:{info['case']} {info['kappa']:+5.2f}"
    summary = f"{fmt('L', left)}  {fmt('R', right)}  rule {rule:<12s} k {kappa:+5.2f}/m"
    return demand, summary
