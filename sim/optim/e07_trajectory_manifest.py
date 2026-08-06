"""E07 trajectory manifest utilities.

The manifest keeps the E07 adapter dataset separate from the standard 48
benchmark registry.  A spec is a plain dictionary with:
    key, type, family, split, speed_kph, params
"""

from __future__ import annotations

import math
import os
from collections import Counter
from copy import deepcopy

import yaml

from common import TrajectoryPoint
from model.trajectory import (
    SPEED_BANDS_KPH,
    TRAJECTORY_TYPES,
    _SPEED_PARAMS,
    _build_trajectory,
    apply_curvature_speed_profile,
    apply_trapezoidal_speed_profile,
    expand_trajectories,
    generate_clothoid_turn,
    generate_combined,
    generate_circle,
    generate_double_lane_change,
    generate_lane_change,
    generate_s_curve,
    generate_stop_and_go,
    generate_straight,
    generate_uturn,
)


DEFAULT_E07_MANIFEST = os.path.join(
    'results', 'e07_trajectories', 'e07_param228_manifest.yaml')
DEFAULT_E07_V2_CANDIDATE_MANIFEST = os.path.join(
    'results', 'e07_trajectories', 'e07_v2_candidate_manifest.yaml')
DEFAULT_E07_V2_TRAIN_MANIFEST = os.path.join(
    'results', 'e07_trajectories', 'e07_v2_train_manifest.yaml')
EXTRA_STANDARD_SPEEDS_KPH = [10, 15, 30, 40]


def _spec(key: str, typ: str, family: str, speed_kph: float,
          params: dict | None = None, split: str = 'train',
          gate_role: str | None = None) -> dict:
    spec = {
        'key': key,
        'type': typ,
        'family': family,
        'split': split,
        'speed_kph': float(speed_kph),
        'params': params or {},
    }
    if gate_role is not None:
        spec['gate_role'] = gate_role
    return spec


def _interp_speed_params(speed_kph: float) -> dict:
    if int(speed_kph) in _SPEED_PARAMS:
        return deepcopy(_SPEED_PARAMS[int(speed_kph)])

    bands = sorted(_SPEED_PARAMS)
    if speed_kph <= bands[0]:
        return deepcopy(_SPEED_PARAMS[bands[0]])
    if speed_kph >= bands[-1]:
        return deepcopy(_SPEED_PARAMS[bands[-1]])

    lo = max(b for b in bands if b < speed_kph)
    hi = min(b for b in bands if b > speed_kph)
    ratio = (speed_kph - lo) / (hi - lo)
    params = {}
    for name, lo_value in _SPEED_PARAMS[lo].items():
        hi_value = _SPEED_PARAMS[hi][name]
        params[name] = float(lo_value + ratio * (hi_value - lo_value))
    return params


def _standard_family_trajectory(family: str, speed_kph: float):
    if int(speed_kph) in SPEED_BANDS_KPH and abs(speed_kph - int(speed_kph)) < 1e-9:
        return _build_trajectory(family, int(speed_kph))

    spd = speed_kph / 3.6
    p = _interp_speed_params(speed_kph)
    if family == 'lane_change':
        return generate_lane_change(
            lane_width=3.5, change_length=p['lc_len'], speed=spd)
    if family == 'double_lc':
        return generate_double_lane_change(
            lane_width=3.5, change_length=p['lc_len'], speed=spd)
    if family == 'clothoid_left':
        return generate_clothoid_turn(
            radius=p['r_clothoid'], turn_angle=p['clothoid_angle'], speed=spd)
    if family == 'clothoid_right':
        return generate_clothoid_turn(
            radius=p['r_clothoid'], turn_angle=-p['clothoid_angle'], speed=spd)
    if family == 's_curve':
        return generate_s_curve(
            radius=p['r_scurve'], arc_angle=p['scurve_angle'], speed=spd)
    if family == 'combined_decel':
        lead = max(30.0, spd * 6.0)
        base = generate_combined(
            speed=spd, radius=p['r_combined'], lead_in=lead,
            seg3_length=lead)
        return apply_curvature_speed_profile(base, v_cruise=spd)
    if family == 'clothoid_decel':
        r_decel = max(15.0, p['r_clothoid'] * 0.6)
        lead = max(30.0, spd * 6.0)
        base = generate_clothoid_turn(
            radius=r_decel, turn_angle=p['clothoid_angle'], speed=spd,
            lead_in=lead, lead_out=lead)
        return apply_curvature_speed_profile(base, v_cruise=spd)
    if family == 'lc_accel':
        base = generate_lane_change(
            lane_width=3.5, change_length=p['lc_len'], speed=spd)
        return apply_trapezoidal_speed_profile(base, v_base=spd)
    raise ValueError(f'Unknown E07 standard family: {family}')


def _mirror_y(points: list[TrajectoryPoint]) -> list[TrajectoryPoint]:
    return [
        TrajectoryPoint(
            x=p.x, y=-p.y, theta=-p.theta, kappa=-p.kappa,
            v=p.v, a=p.a, s=p.s, t=p.t,
        )
        for p in points
    ]


def build_e07_param228_specs(seed: int = 42) -> list[dict]:
    del seed  # Fixed grid for reproducibility; kept in the API for manifests.
    specs = []

    for key, _label, _gen in expand_trajectories(None):
        family, speed_token = key.rsplit('_', 1)
        speed_kph = float(speed_token.replace('kph', ''))
        specs.append(_spec(
            key=key,
            typ='standard48',
            family=family,
            speed_kph=speed_kph,
        ))

    for family in TRAJECTORY_TYPES:
        for speed_kph in EXTRA_STANDARD_SPEEDS_KPH:
            specs.append(_spec(
                key=f'{family}_{speed_kph}kph_extra',
                typ='standard_speed_extra',
                family=family,
                speed_kph=speed_kph,
            ))

    for radius in [8.0, 10.0, 12.0, 15.0, 20.0]:
        for speed_kph in [5.0, 8.0, 10.0, 15.0]:
            for direction in ['left', 'right']:
                specs.append(_spec(
                    key=f'uturn_r{radius:g}_{speed_kph:g}kph_{direction}',
                    typ='uturn',
                    family='uturn',
                    speed_kph=speed_kph,
                    params={'radius': radius, 'direction': direction},
                ))

    for radius in [10.0, 12.0, 15.0, 20.0, 25.0, 30.0]:
        for speed_kph in [5.0, 8.0, 10.0, 15.0, 18.0]:
            for direction in ['left', 'right']:
                specs.append(_spec(
                    key=f'high_curv_arc_r{radius:g}_{speed_kph:g}kph_{direction}',
                    typ='high_curvature_arc',
                    family='circle_arc',
                    speed_kph=speed_kph,
                    params={
                        'radius': radius,
                        'arc_angle': math.pi / 2,
                        'direction': direction,
                    },
                ))

    for speed_kph in [5.0, 8.0, 10.0, 15.0, 18.0, 25.0]:
        for accel_rate in [0.3, 0.6]:
            for length in [80.0, 120.0]:
                specs.append(_spec(
                    key=(
                        f'straight_accel_decel_{speed_kph:g}kph_'
                        f'a{accel_rate:g}_l{length:g}'
                    ),
                    typ='straight_accel_decel',
                    family='straight_accel_decel',
                    speed_kph=speed_kph,
                    params={'length': length, 'accel_rate': accel_rate},
                ))

    for speed_kph in [5.0, 8.0, 10.0, 15.0, 18.0, 25.0]:
        for decel_rate in [0.4, 0.8]:
            for stop_duration in [1.5, 3.0]:
                specs.append(_spec(
                    key=(
                        f'stop_go_{speed_kph:g}kph_'
                        f'd{decel_rate:g}_stop{stop_duration:g}'
                    ),
                    typ='stop_and_go',
                    family='stop_and_go',
                    speed_kph=speed_kph,
                    params={
                        'accel_rate': decel_rate,
                        'decel_rate': decel_rate,
                        'stop_duration': stop_duration,
                    },
                ))

    return specs


def build_e07_v2_candidate_specs(seed: int = 42) -> list[dict]:
    """Build the E07 v2 candidate manifest before controllability gating.

    E07 v2 keeps the standard 48 and mild speed-interpolation trajectories as
    core coverage.  Additional scenarios are candidates that must pass the
    downstream geometry/base-dynamics gate before entering the formal training
    manifest.
    """
    del seed  # Fixed grid for reproducibility; kept in the API for manifests.
    specs = []

    for key, _label, _gen in expand_trajectories(None):
        family, speed_token = key.rsplit('_', 1)
        speed_kph = float(speed_token.replace('kph', ''))
        specs.append(_spec(
            key=key,
            typ='standard48',
            family=family,
            speed_kph=speed_kph,
            gate_role='core',
        ))

    for family in TRAJECTORY_TYPES:
        for speed_kph in EXTRA_STANDARD_SPEEDS_KPH:
            specs.append(_spec(
                key=f'{family}_{speed_kph}kph_extra',
                typ='standard_speed_extra',
                family=family,
                speed_kph=speed_kph,
                gate_role='core',
            ))

    for speed_kph in [5.0, 8.0, 10.0, 15.0, 18.0, 25.0]:
        for accel_rate in [0.3, 0.6]:
            length = 120.0 if speed_kph >= 18.0 else 80.0
            specs.append(_spec(
                key=(
                    f'straight_accel_decel_{speed_kph:g}kph_'
                    f'a{accel_rate:g}_l{length:g}'
                ),
                typ='straight_accel_decel',
                family='straight_accel_decel',
                speed_kph=speed_kph,
                params={'length': length, 'accel_rate': accel_rate},
                gate_role='candidate',
            ))

    for speed_kph in [5.0, 8.0, 10.0, 15.0]:
        for decel_rate in [0.4, 0.8]:
            for stop_duration in [1.5, 3.0]:
                specs.append(_spec(
                    key=(
                        f'stop_go_{speed_kph:g}kph_'
                        f'd{decel_rate:g}_stop{stop_duration:g}'
                    ),
                    typ='stop_and_go',
                    family='stop_and_go',
                    speed_kph=speed_kph,
                    params={
                        'accel_rate': decel_rate,
                        'decel_rate': decel_rate,
                        'stop_duration': stop_duration,
                    },
                    gate_role='candidate',
                ))

    for radius in [15.0, 20.0, 30.0]:
        for speed_kph in [5.0, 8.0, 10.0]:
            for direction in ['left', 'right']:
                specs.append(_spec(
                    key=f'uturn_r{radius:g}_{speed_kph:g}kph_{direction}',
                    typ='uturn',
                    family='uturn',
                    speed_kph=speed_kph,
                    params={'radius': radius, 'direction': direction},
                    gate_role='candidate',
                ))

    for radius in [15.0, 20.0, 30.0]:
        for speed_kph in [5.0, 8.0, 10.0, 15.0]:
            for direction in ['left', 'right']:
                specs.append(_spec(
                    key=(
                        f'smooth_high_curv_arc_r{radius:g}_'
                        f'{speed_kph:g}kph_{direction}'
                    ),
                    typ='smooth_high_curvature_arc',
                    family='smooth_high_curvature_arc',
                    speed_kph=speed_kph,
                    params={
                        'radius': radius,
                        'arc_angle': math.pi / 2,
                        'direction': direction,
                        'lead_in': max(20.0, speed_kph / 3.6 * 4.0),
                        'lead_out': max(20.0, speed_kph / 3.6 * 4.0),
                    },
                    gate_role='candidate',
                ))

    for speed_kph in [8.0, 10.0, 15.0, 18.0]:
        radius = 25.0 if speed_kph <= 10.0 else 35.0
        for direction in ['left', 'right']:
            specs.append(_spec(
                key=f'intersection_turn_r{radius:g}_{speed_kph:g}kph_{direction}',
                typ='intersection_turn',
                family='intersection_turn',
                speed_kph=speed_kph,
                params={
                    'radius': radius,
                    'turn_angle': math.pi / 2,
                    'direction': direction,
                    'lead_in': max(25.0, speed_kph / 3.6 * 5.0),
                    'lead_out': max(25.0, speed_kph / 3.6 * 5.0),
                },
                gate_role='candidate',
            ))

    return specs


def build_e07_tiny_specs() -> list[dict]:
    return [
        _spec('tiny_arc_left', 'high_curvature_arc', 'circle_arc', 8.0,
              {'radius': 20.0, 'arc_angle': 0.2, 'direction': 'left'}),
        _spec('tiny_arc_right', 'high_curvature_arc', 'circle_arc', 8.0,
              {'radius': 20.0, 'arc_angle': 0.2, 'direction': 'right'}),
        _spec('tiny_uturn_left', 'uturn', 'uturn', 5.0,
              {'radius': 12.0, 'direction': 'left'}),
        _spec('tiny_uturn_right', 'uturn', 'uturn', 5.0,
              {'radius': 12.0, 'direction': 'right'}),
        _spec('tiny_accel_decel', 'straight_accel_decel',
              'straight_accel_decel', 8.0,
              {'length': 30.0, 'accel_rate': 0.5}),
        _spec('tiny_stop_go', 'stop_and_go', 'stop_and_go', 8.0,
              {'accel_rate': 0.6, 'decel_rate': 0.6, 'stop_duration': 0.4}),
    ]


def materialize_e07_specs(specs: list[dict]):
    items = []
    for spec in specs:
        speed = float(spec['speed_kph']) / 3.6
        params = spec.get('params') or {}
        typ = spec['type']
        family = spec.get('family', typ)

        if typ in {'standard48', 'standard_speed_extra'}:
            traj = _standard_family_trajectory(family, float(spec['speed_kph']))
        elif typ == 'uturn':
            radius = float(params['radius'])
            direction = params.get('direction', 'left')
            clothoid_ratio = float(params.get('clothoid_ratio', 0.3))
            lead_in = float(params.get('lead_in', 20.0))
            lead_out = float(params.get('lead_out', 20.0))
            if direction == 'left':
                traj = generate_clothoid_turn(
                    radius=radius, turn_angle=math.pi, speed=speed,
                    clothoid_ratio=clothoid_ratio,
                    lead_in=lead_in, lead_out=lead_out)
            elif direction == 'right':
                traj = generate_clothoid_turn(
                    radius=radius, turn_angle=-math.pi, speed=speed,
                    clothoid_ratio=clothoid_ratio,
                    lead_in=lead_in, lead_out=lead_out)
            else:
                raise ValueError(f'Unknown U-turn direction: {direction}')
        elif typ == 'high_curvature_arc':
            radius = float(params['radius'])
            arc_angle = float(params.get('arc_angle', math.pi / 2))
            traj = generate_circle(
                radius=radius,
                speed=speed,
                arc_angle=arc_angle,
            )
            if params.get('direction', 'left') == 'right':
                traj = _mirror_y(traj)
        elif typ == 'smooth_high_curvature_arc':
            radius = float(params['radius'])
            arc_angle = float(params.get('arc_angle', math.pi / 2))
            direction = params.get('direction', 'left')
            lead_in = float(params.get('lead_in', 20.0))
            lead_out = float(params.get('lead_out', 20.0))
            turn_angle = arc_angle if direction == 'left' else -arc_angle
            traj = generate_clothoid_turn(
                radius=radius, turn_angle=turn_angle, speed=speed,
                clothoid_ratio=float(params.get('clothoid_ratio', 0.3)),
                lead_in=lead_in, lead_out=lead_out)
        elif typ == 'intersection_turn':
            radius = float(params['radius'])
            turn_angle = float(params.get('turn_angle', math.pi / 2))
            direction = params.get('direction', 'left')
            if direction == 'right':
                turn_angle = -turn_angle
            traj = generate_clothoid_turn(
                radius=radius, turn_angle=turn_angle, speed=speed,
                clothoid_ratio=float(params.get('clothoid_ratio', 0.35)),
                lead_in=float(params.get('lead_in', 25.0)),
                lead_out=float(params.get('lead_out', 25.0)))
        elif typ == 'straight_accel_decel':
            length = float(params['length'])
            accel_rate = float(params['accel_rate'])
            base = generate_straight(length=length, speed=speed)
            traj = apply_trapezoidal_speed_profile(
                base, v_base=speed, delta_ratio=0.25, accel_rate=accel_rate)
        elif typ == 'stop_and_go':
            traj = generate_stop_and_go(
                cruise_speed=speed,
                accel_rate=float(params.get('accel_rate', 0.5)),
                decel_rate=float(params.get('decel_rate', 0.5)),
                stop_duration=float(params.get('stop_duration', 2.0)),
            )
        elif typ == 'e09_id_composite':
            from optim.e09_long_route import generate_e09_id_composite

            traj = generate_e09_id_composite(
                speed=speed,
                turn_direction=str(params.get('turn_direction', 'left')),
            )
        else:
            raise ValueError(f"Unknown E07 trajectory type: {typ}")

        items.append((spec['key'], traj))
    return items


def metadata_by_key(specs: list[dict]) -> dict:
    return {
        spec['key']: {
            'trajectory_type': spec['type'],
            'trajectory_family': spec.get('family', spec['type']),
            'split': spec.get('split', 'train'),
            'speed_kph': float(spec['speed_kph']),
            'gate_role': spec.get('gate_role'),
        }
        for spec in specs
    }


def summarize_specs(specs: list[dict], preset: str, seed: int) -> dict:
    type_counts = dict(Counter(spec['type'] for spec in specs))
    speeds = [float(spec['speed_kph']) for spec in specs]
    curvatures = []
    for spec in specs:
        params = spec.get('params') or {}
        radius = params.get('radius')
        if radius:
            curvatures.append(abs(1.0 / float(radius)))
    return {
        'preset': preset,
        'seed': int(seed),
        'trajectory_count': len(specs),
        'type_counts': type_counts,
        'speed_kph_min': min(speeds) if speeds else None,
        'speed_kph_max': max(speeds) if speeds else None,
        'max_abs_curvature_estimate': max(curvatures) if curvatures else None,
    }


def write_e07_manifest(specs: list[dict], path: str,
                       preset: str = 'param228', seed: int = 42) -> dict:
    summary = summarize_specs(specs, preset=preset, seed=seed)
    payload = {
        **summary,
        'specs': specs,
    }
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return summary


def load_e07_manifest(path: str) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        payload = yaml.safe_load(f)
    if isinstance(payload, dict) and 'specs' in payload:
        return payload['specs']
    if isinstance(payload, list):
        return payload
    raise ValueError(f'Invalid E07 trajectory manifest: {path}')
