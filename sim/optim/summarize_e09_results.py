"""Build the paired E09-B/C/D evidence table from formal YAML outputs."""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import yaml


def _load(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict) or not isinstance(payload.get('results'), list):
        raise ValueError(f'invalid E09 result YAML: {path}')
    return payload


def _index(payload: dict, path: str) -> dict[str, dict]:
    result = {}
    for row in payload['results']:
        key = str(row['key'])
        if key in result:
            raise ValueError(f'duplicate key {key} in {path}')
        result[key] = row
    return result


def _identity(payload: dict) -> tuple:
    run_spec = payload.get('run_spec') or {}
    return (
        run_spec.get('model_path'),
        run_spec.get('dc_config_path'),
        run_spec.get('plant'),
    )


def _require_identity(payloads: list[tuple[str, dict]]) -> dict:
    identities = {path: _identity(payload) for path, payload in payloads}
    populated = {identity for identity in identities.values() if any(identity)}
    if len(populated) != 1:
        raise ValueError(f'model/config/plant identity mismatch: {identities}')
    identity = next(iter(populated))
    return {
        'model_path': identity[0],
        'dc_config_path': identity[1],
        'plant': identity[2],
    }


def _method_summary(rows: list[dict], method: str) -> dict:
    losses = np.array([row[method] for row in rows], dtype=np.float64)
    dc = np.array([row['dc_loss'] for row in rows], dtype=np.float64)
    delta_pct = (losses - dc) / np.maximum(dc, 1e-8) * 100.0
    return {
        'count': int(len(rows)),
        'avg_loss': float(np.mean(losses)),
        'median_loss': float(np.median(losses)),
        'dc_avg_loss': float(np.mean(dc)),
        'delta_pct_from_aggregate_means': float(
            (np.mean(losses) - np.mean(dc)) / max(float(np.mean(dc)), 1e-8)
            * 100.0),
        'mean_per_scenario_delta_pct': float(np.mean(delta_pct)),
        'wins_vs_dc': int(np.sum(losses < dc)),
        'worst_case_delta_pct': float(np.max(delta_pct)),
        'worst_case_key': rows[int(np.argmax(delta_pct))]['key'],
    }


def _paired_group(group: str, one_shot_path: str,
                  rolling_path: str | None = None,
                  safe_path: str | None = None) -> tuple[list[dict], dict]:
    one_payload = _load(one_shot_path)
    one = _index(one_payload, one_shot_path)
    payloads = [(one_shot_path, one_payload)]
    rolling = safe = None
    if rolling_path:
        rolling_payload = _load(rolling_path)
        rolling = _index(rolling_payload, rolling_path)
        payloads.append((rolling_path, rolling_payload))
    if safe_path:
        safe_payload = _load(safe_path)
        safe = _index(safe_payload, safe_path)
        payloads.append((safe_path, safe_payload))
    identity = _require_identity(payloads)

    key_sets = [set(one)]
    if rolling is not None:
        key_sets.append(set(rolling))
    if safe is not None:
        key_sets.append(set(safe))
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError(f'{group} result keys are not paired: {key_sets}')

    rows = []
    for key in one:
        one_row = one[key]
        row = {
            'group': group,
            'key': key,
            'dc_loss': float(one_row['dc_loss']),
            'one_shot_loss': float(one_row['rl_loss']),
            'one_shot_is_ood': bool(one_row.get('is_ood', False)),
            'one_shot_action_saturation_ratio': float(
                one_row.get('action_saturation_ratio', 0.0)),
        }
        for method_name, indexed in (('rolling', rolling), ('safe', safe)):
            if indexed is None:
                continue
            method_row = indexed[key]
            method_dc = float(method_row['dc_loss'])
            if not np.isclose(method_dc, row['dc_loss'], rtol=1e-5, atol=1e-6):
                raise ValueError(
                    f'{group}/{key} DC mismatch: {row["dc_loss"]} vs {method_dc}')
            row[f'{method_name}_loss'] = float(method_row['rolling_rl_loss'])
            row[f'{method_name}_window_count'] = int(
                method_row.get('window_count', 0))
            row[f'{method_name}_ood_window_count'] = int(
                method_row.get('ood_window_count', 0))
            row[f'{method_name}_fallback_count'] = int(
                method_row.get('fallback_count', 0))
            row[f'{method_name}_action_saturation_ratio'] = float(
                method_row.get('mean_action_saturation_ratio', 0.0))
        rows.append(row)

    methods = ['one_shot_loss']
    if rolling is not None:
        methods.append('rolling_loss')
    if safe is not None:
        methods.append('safe_loss')
    summary = {
        'trajectory_count': len(rows),
        'identity': identity,
        'source_files': [os.path.abspath(path) for path, _ in payloads],
        'methods': {method.removesuffix('_loss'): _method_summary(rows, method)
                    for method in methods},
        'one_shot_ood_count': int(sum(row['one_shot_is_ood'] for row in rows)),
    }
    for method_name in ('rolling', 'safe'):
        if f'{method_name}_loss' not in rows[0]:
            continue
        window_count = sum(row[f'{method_name}_window_count'] for row in rows)
        ood_count = sum(row[f'{method_name}_ood_window_count'] for row in rows)
        fallback_count = sum(row[f'{method_name}_fallback_count'] for row in rows)
        summary[f'{method_name}_diagnostics'] = {
            'window_count': int(window_count),
            'ood_window_count': int(ood_count),
            'ood_window_ratio': float(ood_count / max(window_count, 1)),
            'fallback_count': int(fallback_count),
            'fallback_ratio': float(fallback_count / max(window_count, 1)),
            'mean_scenario_action_saturation_ratio': float(np.mean([
                row[f'{method_name}_action_saturation_ratio'] for row in rows
            ])),
        }
    return rows, summary


def _one_shot_group(group: str, path: str) -> tuple[list[dict], dict]:
    payload = _load(path)
    indexed = _index(payload, path)
    identity = _require_identity([(path, payload)])
    rows = [{
        'group': group,
        'key': key,
        'dc_loss': float(row['dc_loss']),
        'one_shot_loss': float(row['rl_loss']),
        'one_shot_is_ood': bool(row.get('is_ood', False)),
        'one_shot_action_saturation_ratio': float(
            row.get('action_saturation_ratio', 0.0)),
    } for key, row in indexed.items()]
    id_rows = [row for row in rows if not row['one_shot_is_ood']]
    ood_rows = [row for row in rows if row['one_shot_is_ood']]
    summary = {
        'trajectory_count': len(rows),
        'identity': identity,
        'source_files': [os.path.abspath(path)],
        'methods': {'one_shot': _method_summary(rows, 'one_shot_loss')},
        'one_shot_ood_count': len(ood_rows),
        'mean_action_saturation_ratio': float(np.mean([
            row['one_shot_action_saturation_ratio'] for row in rows
        ])),
        'id_one_shot': (
            _method_summary(id_rows, 'one_shot_loss') if id_rows else None),
        'ood_one_shot': (
            _method_summary(ood_rows, 'one_shot_loss') if ood_rows else None),
    }
    return rows, summary


def summarize(args) -> dict:
    groups = {
        'E09-B': _paired_group('E09-B', args.e09_b_one_shot),
        'E09-C': _paired_group(
            'E09-C', args.e09_c_one_shot, args.e09_c_rolling,
            args.e09_c_safe),
        'E09-D': _paired_group(
            'E09-D', args.e09_d_one_shot, args.e09_d_rolling,
            args.e09_d_safe),
    }
    e09_a_path = getattr(args, 'e09_a_one_shot', None)
    if e09_a_path:
        groups = {'E09-A': _one_shot_group('E09-A', e09_a_path), **groups}
    identities = {tuple(value[1]['identity'].values()) for value in groups.values()}
    if len(identities) != 1:
        raise ValueError(f'E09 groups use different model/config/plant: {identities}')
    all_rows = [row for rows, _summary in groups.values() for row in rows]
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'e09_per_scenario.csv')
    fieldnames = []
    for row in all_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    payload = {
        'protocol': {
            'loss_reference_mode': 'per_step_trajectory_v',
            'primary_horizon_m': 160.0,
            'primary_stride_m': 20.0,
            'identity': next(iter(groups.values()))[1]['identity'],
        },
        'groups': {name: summary for name, (_rows, summary) in groups.items()},
        'per_scenario_csv': os.path.abspath(csv_path),
    }
    sensitivity_rolling = getattr(args, 'e09_c_rolling_sensitivity', None)
    sensitivity_safe = getattr(args, 'e09_c_safe_sensitivity', None)
    if bool(sensitivity_rolling) != bool(sensitivity_safe):
        raise ValueError('provide both E09-C sensitivity result files')
    if sensitivity_rolling:
        _rows, sensitivity_summary = _paired_group(
            'E09-C-h120-sensitivity', args.e09_c_one_shot,
            sensitivity_rolling, sensitivity_safe)
        payload['horizon_sensitivity_120m'] = sensitivity_summary
    yaml_path = os.path.join(args.output_dir, 'e09_summary.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e09-a-one-shot')
    parser.add_argument('--e09-b-one-shot', required=True)
    parser.add_argument('--e09-c-one-shot', required=True)
    parser.add_argument('--e09-c-rolling', required=True)
    parser.add_argument('--e09-c-safe', required=True)
    parser.add_argument('--e09-c-rolling-sensitivity')
    parser.add_argument('--e09-c-safe-sensitivity')
    parser.add_argument('--e09-d-one-shot', required=True)
    parser.add_argument('--e09-d-rolling', required=True)
    parser.add_argument('--e09-d-safe', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    payload = summarize(args)
    print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


if __name__ == '__main__':
    main()
