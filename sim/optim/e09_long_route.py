"""Generated long ID-composite routes for E09-C rolling evaluation."""

from __future__ import annotations

import math

from common import TrajectoryPoint
from model.trajectory import (
    apply_trapezoidal_speed_profile,
    generate_clothoid_turn,
    generate_double_lane_change,
    generate_lane_change,
    generate_straight,
)


def _append_transformed(target: list[TrajectoryPoint], segment,
                        direction: float = 1.0) -> None:
    if not target:
        target.extend(segment)
        return
    origin = segment[0]
    anchor = target[-1]
    heading_offset = anchor.theta - direction * origin.theta
    cos_h = math.cos(heading_offset)
    sin_h = math.sin(heading_offset)
    for point in segment[1:]:
        local_x = point.x - origin.x
        local_y = direction * (point.y - origin.y)
        target.append(TrajectoryPoint(
            x=anchor.x + cos_h * local_x - sin_h * local_y,
            y=anchor.y + sin_h * local_x + cos_h * local_y,
            theta=heading_offset + direction * point.theta,
            kappa=direction * point.kappa,
            v=point.v,
            a=point.a,
            s=0.0,
            t=0.0,
        ))


def _reparameterize(points: list[TrajectoryPoint], speed: float):
    output = []
    distance = 0.0
    for index, point in enumerate(points):
        if index:
            previous = points[index - 1]
            distance += math.hypot(
                point.x - previous.x, point.y - previous.y)
        output.append(TrajectoryPoint(
            x=point.x, y=point.y, theta=point.theta, kappa=point.kappa,
            v=speed, a=0.0, s=distance, t=distance / speed,
        ))
    return output


def generate_e09_id_composite(speed: float = 25.0 / 3.6,
                              turn_direction: str = 'left'):
    """Compose familiar primitives without geometry or speed discontinuities."""
    if turn_direction not in {'left', 'right'}:
        raise ValueError(f'unknown turn_direction: {turn_direction}')
    direction = 1.0 if turn_direction == 'left' else -1.0
    points: list[TrajectoryPoint] = []
    segments = [
        (generate_straight(length=100.0, speed=speed), 1.0),
        (generate_lane_change(
            lane_width=3.5, change_length=120.0, speed=speed), direction),
        (generate_straight(length=80.0, speed=speed), 1.0),
        (generate_clothoid_turn(
            radius=60.0, turn_angle=math.pi / 2,
            speed=speed, lead_in=60.0, lead_out=60.0), direction),
        (generate_double_lane_change(
            lane_width=3.5, change_length=140.0, speed=speed), direction),
        (generate_straight(length=100.0, speed=speed), 1.0),
    ]
    for segment, mirror in segments:
        _append_transformed(points, segment, direction=mirror)
    base = _reparameterize(points, speed=speed)
    return apply_trapezoidal_speed_profile(
        base, v_base=speed, delta_ratio=0.20, accel_rate=0.30)
