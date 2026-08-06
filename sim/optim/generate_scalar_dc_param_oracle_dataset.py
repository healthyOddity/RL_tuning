"""Generate the E07 per-trajectory scalar DC oracle dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from time import perf_counter

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import apply_plant_override, load_config
from optim.e07_dc_param_oracle_common import (
    DEFAULT_DC_CONFIG,
    PARAM_VECTOR_DIM,
    apply_param_vector,
    baseline_param_vector,
    flatten_params,
    hard_loss_for_params,
    materialize_smoke_trajectories,
    materialize_standard_trajectories,
    observation_for_trajectory,
    parameter_names,
    project_params_,
    soft_loss_for_params,
)
from optim.e07_trajectory_manifest import (
    load_e07_manifest,
    materialize_e07_specs,
    metadata_by_key,
)
from optim.train import DiffControllerParams, tracking_loss
from sim_loop import run_simulation


DATASET_SCHEMA_VERSION = 3
SELECTION_METRIC = 'hard_closed_loop_loss'
LOSS_REFERENCE_MODE = 'per_step_trajectory_v'
NORM_ALPHA = 0.5
NORM_FLOOR = 1.0
LOSS_WEIGHTS = {
    'w_lat': 10.0,
    'w_head': 8.0,
    'w_speed': 3.0,
    'w_steer_rate': 0.05,
    'w_acc_rate': 0.01,
}


def _write_csv(path: str, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _atomic_write_yaml(path: str, payload: dict) -> None:
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp_path, path)


def _atomic_savez(path: str, **payload) -> None:
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'wb') as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp_path, path)


def _metadata_arrays(keys: list[str], trajectory_metadata: dict | None):
    metadata = trajectory_metadata or {}
    types = []
    families = []
    splits = []
    speeds = []
    for key in keys:
        row = metadata.get(key, {})
        typ = row.get('trajectory_type') or 'standard48'
        family = row.get('trajectory_family') or typ
        split = row.get('split') or ''
        speed = row.get('speed_kph', np.nan)
        types.append(str(typ))
        families.append(str(family))
        splits.append(str(split))
        speeds.append(float(speed))
    return (
        np.array(types),
        np.array(families),
        np.array(splits),
        np.array(speeds, dtype=np.float32),
    )


def _nanmean_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float('nan')
    return float(np.mean(finite))


def _progress_logger(output_dir: str, verbose: bool = True):
    log_path = os.path.join(output_dir, 'progress.log')

    def _log(message: str) -> None:
        stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{stamp}] {message}'
        if verbose:
            print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    return _log, log_path


def _sanitize_grad(grad):
    if grad is None:
        return None
    return torch.nan_to_num(
        grad, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e4, 1e4)


def _soft_loss_tensor(traj, params: DiffControllerParams,
                      tbptt_k: int) -> torch.Tensor:
    history = run_simulation(
        traj,
        init_speed=traj[0].v,
        init_x=traj[0].x,
        init_y=traj[0].y,
        init_yaw=traj[0].theta,
        cfg=params.cfg,
        lat_ctrl=params.lat_ctrl,
        lon_ctrl=params.lon_ctrl,
        differentiable=True,
        tbptt_k=int(tbptt_k),
    )
    return tracking_loss(
        history,
        ref_speed=None,
        **LOSS_WEIGHTS,
    )


def optimize_scalar_dc_params_for_trajectory(
    cfg: dict,
    traj,
    baseline_vec: np.ndarray,
    epochs: int = 10,
    lr: float = 0.01,
    lr_tables: float | None = None,
    grad_clip: float = 10.0,
    l2_weight: float = 0.01,
    tbptt_k: int = 150,
    reject_worse: bool = True,
):
    """Run the original scalar DC update protocol on one trajectory."""
    if lr_tables is None:
        lr_tables = lr
    if epochs < 0:
        raise ValueError('epochs must be non-negative')
    if tbptt_k < 0:
        raise ValueError('tbptt_k must be non-negative')

    params = DiffControllerParams(cfg)
    apply_param_vector(params, baseline_vec)
    initial_params = {
        name: p.detach().clone()
        for name, p in params.named_parameters()
    }

    table_params = []
    other_params = []
    for name, p in params.named_parameters():
        if '_y' in name:
            table_params.append(p)
        else:
            other_params.append(p)
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': float(lr)},
        {'params': table_params, 'lr': float(lr_tables)},
    ])
    scheduler = None
    if epochs > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(epochs), eta_min=float(lr) * 0.1)
    hooks = [p.register_hook(_sanitize_grad) for p in params.parameters()]

    baseline_soft_loss = soft_loss_for_params(
        traj, cfg, baseline_vec, tbptt_k=tbptt_k)
    baseline_hard_loss = hard_loss_for_params(traj, cfg, baseline_vec)
    norm_factor = max(
        max(float(baseline_soft_loss), 1e-6) ** NORM_ALPHA,
        NORM_FLOOR,
    )

    best_hard_loss = (
        float(baseline_hard_loss) if reject_worse else float('inf'))
    best_hard_epoch = 0
    best_vec = baseline_vec.copy()
    best_soft_loss = float(baseline_soft_loss)
    best_soft_epoch = 0
    epoch_rows = []
    finite_candidate_seen = False

    try:
        for epoch in range(1, int(epochs) + 1):
            lr_before = float(optimizer.param_groups[0]['lr'])
            lr_tables_before = float(optimizer.param_groups[1]['lr'])
            optimizer.zero_grad()
            soft_before = _soft_loss_tensor(traj, params, tbptt_k=tbptt_k)
            l2_reg = torch.zeros(
                (), dtype=soft_before.dtype, device=soft_before.device)
            for name, p in params.named_parameters():
                l2_reg = l2_reg + ((p - initial_params[name]) ** 2).sum()
            objective = (
                soft_before / float(norm_factor)
                + float(l2_weight) * l2_reg
            )
            if not torch.isfinite(objective):
                epoch_rows.append({
                    'epoch': epoch,
                    'lr': lr_before,
                    'lr_tables': lr_tables_before,
                    'norm_factor': float(norm_factor),
                    'soft_loss_before_step': float('nan'),
                    'objective_before_step': float('nan'),
                    'soft_loss_after_step': float('nan'),
                    'hard_loss_after_step': float('nan'),
                    'grad_norm': float('nan'),
                    'nan_grad_count': 0,
                    'soft_best_updated': False,
                    'hard_best_updated': False,
                })
                break

            objective.backward()
            nan_grad_count = 0
            for p in params.parameters():
                if p.grad is not None:
                    nan_grad_count += int(
                        (p.grad.abs() >= 1e4).sum().item())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(params.parameters()),
                max_norm=float(grad_clip),
            ).item()
            optimizer.step()
            project_params_(params, fallback=initial_params)
            if scheduler is not None:
                scheduler.step()

            current_vec = flatten_params(params)
            soft_after = soft_loss_for_params(
                traj, cfg, current_vec, tbptt_k=tbptt_k)
            hard_after = hard_loss_for_params(traj, cfg, current_vec)
            finite_candidate = (
                np.isfinite(current_vec).all()
                and np.isfinite(soft_after)
                and np.isfinite(hard_after)
            )
            finite_candidate_seen = finite_candidate_seen or finite_candidate

            soft_updated = (
                finite_candidate
                and soft_after < best_soft_loss - 1e-9
            )
            if soft_updated:
                best_soft_loss = float(soft_after)
                best_soft_epoch = epoch

            hard_updated = (
                finite_candidate
                and hard_after < best_hard_loss - 1e-9
            )
            if hard_updated:
                best_hard_loss = float(hard_after)
                best_hard_epoch = epoch
                best_vec = current_vec.copy()

            epoch_rows.append({
                'epoch': epoch,
                'lr': lr_before,
                'lr_tables': lr_tables_before,
                'norm_factor': float(norm_factor),
                'soft_loss_before_step': float(soft_before.detach().item()),
                'objective_before_step': float(objective.detach().item()),
                'soft_loss_after_step': float(soft_after),
                'hard_loss_after_step': float(hard_after),
                'grad_norm': float(grad_norm),
                'nan_grad_count': int(nan_grad_count),
                'soft_best_updated': bool(soft_updated),
                'hard_best_updated': bool(hard_updated),
            })
    finally:
        for hook in hooks:
            hook.remove()

    status = 'optimized'
    if not finite_candidate_seen and epochs > 0:
        status = 'fallback_nonfinite'
        best_vec = baseline_vec.copy()
        best_hard_loss = float(baseline_hard_loss)
        best_hard_epoch = 0
    elif not np.isfinite(best_vec).all() or not np.isfinite(best_hard_loss):
        status = 'fallback_nonfinite'
        best_vec = baseline_vec.copy()
        best_hard_loss = float(baseline_hard_loss)
        best_hard_epoch = 0
    elif reject_worse and (
        best_hard_epoch == 0
        or best_hard_loss >= baseline_hard_loss - 1e-9
    ):
        status = 'fallback_worse_than_dc'
        best_vec = baseline_vec.copy()
        best_hard_loss = float(baseline_hard_loss)
        best_hard_epoch = 0
    elif not reject_worse and best_hard_epoch == 0:
        status = 'fallback_worse_than_dc'
        best_vec = baseline_vec.copy()
        best_hard_loss = float(baseline_hard_loss)
    elif best_hard_loss >= baseline_hard_loss - 1e-9:
        status = 'accepted_worse'

    selected_soft_loss = soft_loss_for_params(
        traj, cfg, best_vec, tbptt_k=tbptt_k)
    return {
        'params': best_vec.astype(np.float32),
        'baseline_soft_loss': float(baseline_soft_loss),
        'baseline_hard_loss': float(baseline_hard_loss),
        'oracle_soft_loss': float(selected_soft_loss),
        'oracle_hard_loss': float(best_hard_loss),
        'best_soft_loss': float(best_soft_loss),
        'best_soft_epoch': int(best_soft_epoch),
        'best_hard_epoch': int(best_hard_epoch),
        'status': status,
        'norm_factor': float(norm_factor),
        'epoch_rows': epoch_rows,
    }


def _plain_value(value):
    if isinstance(value, dict):
        return {
            str(key): _plain_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, '__dict__'):
        return _plain_value(vars(value))
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _trajectory_digest(traj) -> str:
    payload = [_plain_value(point) for point in traj]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _build_run_spec(
    cfg: dict,
    trajectory_items,
    trajectory_metadata: dict | None,
    trajectory_manifest: str | None,
    epochs: int,
    lr: float,
    lr_tables: float,
    grad_clip: float,
    l2_weight: float,
    tbptt_k: int,
    scalar_validate: bool,
    reject_worse: bool,
) -> dict:
    protocol = {
        'epochs': int(epochs),
        'lr': float(lr),
        'lr_tables': float(lr_tables),
        'grad_clip': float(grad_clip),
        'l2_weight': float(l2_weight),
        'tbptt_k': int(tbptt_k),
        'norm_alpha': NORM_ALPHA,
        'norm_floor': NORM_FLOOR,
        'scheduler': 'cosine_annealing',
        'scheduler_eta_min': float(lr) * 0.1,
        'selection_metric': SELECTION_METRIC,
        'loss_reference_mode': LOSS_REFERENCE_MODE,
        'scalar_validate': bool(scalar_validate),
        'reject_worse': bool(reject_worse),
        'loss_weights': dict(LOSS_WEIGHTS),
    }
    body = {
        'schema_version': DATASET_SCHEMA_VERSION,
        'oracle_mode': 'per_trajectory_scalar_dc_hard_selected',
        'config': _plain_value(cfg),
        'trajectory_manifest': trajectory_manifest or '',
        'trajectory_metadata': _plain_value(trajectory_metadata or {}),
        'trajectory_keys': [str(key) for key, _traj in trajectory_items],
        'trajectory_geometry_sha256': {
            str(key): _trajectory_digest(traj)
            for key, traj in trajectory_items
        },
        'protocol': protocol,
    }
    canonical = json.dumps(
        body, sort_keys=True, separators=(',', ':'), allow_nan=False)
    body['fingerprint'] = hashlib.sha256(
        canonical.encode('utf-8')).hexdigest()
    return body


def _prepare_run_directory(output_dir: str, run_spec: dict,
                           resume: bool) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    item_dir = os.path.join(output_dir, 'items')
    run_spec_path = os.path.join(output_dir, 'run_spec.yaml')
    existing = [
        name for name in os.listdir(output_dir)
        if name not in ('.', '..')
    ]

    if existing and not resume:
        raise FileExistsError(
            f'output directory is not empty: {output_dir}; '
            'use --resume or choose a new directory')
    if resume and existing:
        if not os.path.exists(run_spec_path):
            raise RuntimeError(
                f'cannot resume without run_spec.yaml: {output_dir}')
        with open(run_spec_path, encoding='utf-8') as f:
            previous = yaml.safe_load(f) or {}
        if previous.get('fingerprint') != run_spec['fingerprint']:
            raise RuntimeError(
                'resume fingerprint mismatch: config, trajectories, or '
                'optimization protocol changed')
    else:
        _atomic_write_yaml(run_spec_path, run_spec)

    os.makedirs(item_dir, exist_ok=True)
    return item_dir, run_spec_path


def _item_path(item_dir: str, index: int) -> str:
    return os.path.join(item_dir, f'{index:04d}.npz')


def _save_item_checkpoint(path: str, fingerprint: str,
                          result: dict) -> None:
    _atomic_savez(
        path,
        fingerprint=np.array(fingerprint),
        index=np.array(result['index'], dtype=np.int32),
        trajectory_key=np.array(result['trajectory_key']),
        params=np.asarray(result['params'], dtype=np.float32),
        baseline_soft_loss=np.array(result['baseline_soft_loss']),
        baseline_hard_loss=np.array(result['baseline_hard_loss']),
        oracle_soft_loss=np.array(result['oracle_soft_loss']),
        oracle_hard_loss=np.array(result['oracle_hard_loss']),
        best_soft_loss=np.array(result['best_soft_loss']),
        best_soft_epoch=np.array(result['best_soft_epoch'], dtype=np.int32),
        best_hard_epoch=np.array(result['best_hard_epoch'], dtype=np.int32),
        status=np.array(result['status']),
        norm_factor=np.array(result['norm_factor']),
        epoch_rows_json=np.array(json.dumps(result['epoch_rows'])),
        elapsed_s=np.array(result['elapsed_s']),
    )


def _load_item_checkpoint(path: str, fingerprint: str,
                          index: int, key: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data['fingerprint'].item()) != fingerprint:
                return None
            if int(data['index'].item()) != index:
                return None
            if str(data['trajectory_key'].item()) != key:
                return None
            params = data['params'].astype(np.float32)
            if params.shape != (PARAM_VECTOR_DIM,) or not np.isfinite(params).all():
                return None
            return {
                'index': index,
                'trajectory_key': key,
                'params': params,
                'baseline_soft_loss': float(data['baseline_soft_loss'].item()),
                'baseline_hard_loss': float(data['baseline_hard_loss'].item()),
                'oracle_soft_loss': float(data['oracle_soft_loss'].item()),
                'oracle_hard_loss': float(data['oracle_hard_loss'].item()),
                'best_soft_loss': float(data['best_soft_loss'].item()),
                'best_soft_epoch': int(data['best_soft_epoch'].item()),
                'best_hard_epoch': int(data['best_hard_epoch'].item()),
                'status': str(data['status'].item()),
                'norm_factor': float(data['norm_factor'].item()),
                'epoch_rows': json.loads(str(data['epoch_rows_json'].item())),
                'elapsed_s': float(data['elapsed_s'].item()),
            }
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _configure_worker(torch_threads: int) -> None:
    threads = max(int(torch_threads), 1)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _run_trajectory_job(job: dict) -> dict:
    started = perf_counter()
    result = optimize_scalar_dc_params_for_trajectory(
        cfg=job['cfg'],
        traj=job['traj'],
        baseline_vec=job['baseline_vec'],
        epochs=job['epochs'],
        lr=job['lr'],
        lr_tables=job['lr_tables'],
        grad_clip=job['grad_clip'],
        l2_weight=job['l2_weight'],
        tbptt_k=job['tbptt_k'],
        reject_worse=job['reject_worse'],
    )
    if job['scalar_validate']:
        validated_hard = hard_loss_for_params(
            job['traj'], job['cfg'], result['params'])
        if not np.isclose(
            validated_hard,
            result['oracle_hard_loss'],
            rtol=1e-6,
            atol=1e-7,
        ):
            raise RuntimeError(
                f"hard validation mismatch for {job['trajectory_key']}: "
                f"{result['oracle_hard_loss']} vs {validated_hard}")
        result['oracle_hard_loss'] = float(validated_hard)
    return {
        'index': job['index'],
        'trajectory_key': job['trajectory_key'],
        **result,
        'elapsed_s': float(perf_counter() - started),
    }


def _collect_oracle_results(
    cfg: dict,
    trajectory_items,
    baseline_vec: np.ndarray,
    item_dir: str,
    fingerprint: str,
    epochs: int,
    lr: float,
    lr_tables: float,
    grad_clip: float,
    l2_weight: float,
    tbptt_k: int,
    scalar_validate: bool,
    reject_worse: bool,
    num_workers: int,
    worker_torch_threads: int,
    log_progress,
) -> tuple[list[dict], int]:
    results: dict[int, dict] = {}
    jobs = []
    resumed_count = 0
    for index, (key, traj) in enumerate(trajectory_items):
        path = _item_path(item_dir, index)
        cached = _load_item_checkpoint(path, fingerprint, index, key)
        if cached is not None:
            results[index] = cached
            resumed_count += 1
            log_progress(
                f"resume {index + 1}/{len(trajectory_items)} key={key} "
                f"hard_loss={cached['oracle_hard_loss']:.6g}")
            continue
        jobs.append({
            'index': index,
            'trajectory_key': key,
            'traj': traj,
            'cfg': cfg,
            'baseline_vec': baseline_vec,
            'epochs': int(epochs),
            'lr': float(lr),
            'lr_tables': float(lr_tables),
            'grad_clip': float(grad_clip),
            'l2_weight': float(l2_weight),
            'tbptt_k': int(tbptt_k),
            'scalar_validate': bool(scalar_validate),
            'reject_worse': bool(reject_worse),
        })

    failures = {}

    def record(result: dict) -> None:
        index = result['index']
        results[index] = result
        _save_item_checkpoint(
            _item_path(item_dir, index), fingerprint, result)
        log_progress(
            f"oracle {len(results)}/{len(trajectory_items)} "
            f"key={result['trajectory_key']} "
            f"dc_hard={result['baseline_hard_loss']:.6g} "
            f"oracle_hard={result['oracle_hard_loss']:.6g} "
            f"best_epoch={result['best_hard_epoch']} "
            f"status={result['status']} dt={result['elapsed_s']:.1f}s")

    if num_workers == 1:
        previous_torch_threads = torch.get_num_threads()
        torch.set_num_threads(max(int(worker_torch_threads), 1))
        try:
            for job in jobs:
                try:
                    record(_run_trajectory_job(job))
                except Exception as exc:
                    failures[job['trajectory_key']] = repr(exc)
                    log_progress(
                        f"failed key={job['trajectory_key']} error={exc!r}")
        finally:
            torch.set_num_threads(previous_torch_threads)
    else:
        context = mp.get_context('spawn')
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=context,
            initializer=_configure_worker,
            initargs=(worker_torch_threads,),
        ) as executor:
            future_jobs = {
                executor.submit(_run_trajectory_job, job): job
                for job in jobs
            }
            for future in as_completed(future_jobs):
                job = future_jobs[future]
                try:
                    record(future.result())
                except Exception as exc:
                    failures[job['trajectory_key']] = repr(exc)
                    log_progress(
                        f"failed key={job['trajectory_key']} error={exc!r}")

    if failures:
        failed_keys = ', '.join(sorted(failures))
        raise RuntimeError(
            f'{len(failures)} oracle trajectories failed: {failed_keys}; '
            'completed item checkpoints were preserved, rerun with --resume')
    if len(results) != len(trajectory_items):
        raise RuntimeError(
            f'expected {len(trajectory_items)} results, got {len(results)}')
    return [results[index] for index in range(len(trajectory_items))], resumed_count


def generate_oracle_dataset(
    cfg: dict,
    trajectory_items=None,
    trajectory_metadata: dict | None = None,
    trajectory_manifest: str | None = None,
    output_dir: str | None = None,
    epochs: int = 10,
    lr: float = 0.01,
    lr_tables: float | None = None,
    grad_clip: float = 10.0,
    l2_weight: float = 0.01,
    tbptt_k: int = 150,
    scalar_validate: bool = True,
    reject_worse: bool = True,
    num_workers: int = 1,
    worker_torch_threads: int = 1,
    resume: bool = False,
    verbose: bool = True,
):
    if lr_tables is None:
        lr_tables = lr
    if num_workers < 1:
        raise ValueError('num_workers must be at least 1')
    if worker_torch_threads < 1:
        raise ValueError('worker_torch_threads must be at least 1')
    if trajectory_items is None:
        trajectory_items = materialize_standard_trajectories()
    trajectory_items = list(trajectory_items)
    if not trajectory_items:
        raise ValueError('trajectory_items must not be empty')
    keys = [str(key) for key, _traj in trajectory_items]
    if len(set(keys)) != len(keys):
        raise ValueError('trajectory keys must be unique')

    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('results', 'e07_oracle', timestamp)
    run_spec = _build_run_spec(
        cfg=cfg,
        trajectory_items=trajectory_items,
        trajectory_metadata=trajectory_metadata,
        trajectory_manifest=trajectory_manifest,
        epochs=epochs,
        lr=lr,
        lr_tables=lr_tables,
        grad_clip=grad_clip,
        l2_weight=l2_weight,
        tbptt_k=tbptt_k,
        scalar_validate=scalar_validate,
        reject_worse=reject_worse,
    )
    item_dir, run_spec_path = _prepare_run_directory(
        output_dir, run_spec, resume=resume)
    log_progress, progress_log_path = _progress_logger(
        output_dir, verbose=verbose)

    total = len(trajectory_items)
    log_progress(
        f"start E07 scalar DC oracle v2: trajectories={total}, "
        f"epochs={int(epochs)}, lr={float(lr)}, "
        f"lr_tables={float(lr_tables)}, tbptt_k={int(tbptt_k)}, "
        f"selection={SELECTION_METRIC}, workers={int(num_workers)}, "
        f"worker_torch_threads={int(worker_torch_threads)}, "
        f"resume={bool(resume)}, reject_worse={bool(reject_worse)}")
    if trajectory_manifest:
        log_progress(f"trajectory_manifest={trajectory_manifest}")

    trajectory_types, trajectory_families, trajectory_splits, speed_kph = (
        _metadata_arrays(keys, trajectory_metadata)
    )
    param_names_arr = np.array(parameter_names(cfg))
    baseline_vec = baseline_param_vector(cfg)
    if len(baseline_vec) != PARAM_VECTOR_DIM:
        raise RuntimeError(
            f'expected {PARAM_VECTOR_DIM} DC params, got {len(baseline_vec)}')
    features = np.stack([
        observation_for_trajectory(traj, cfg)
        for _key, traj in trajectory_items
    ]).astype(np.float32)
    baseline_params = np.repeat(
        baseline_vec[None, :], total, axis=0).astype(np.float32)

    started = perf_counter()
    results, resumed_count = _collect_oracle_results(
        cfg=cfg,
        trajectory_items=trajectory_items,
        baseline_vec=baseline_vec,
        item_dir=item_dir,
        fingerprint=run_spec['fingerprint'],
        epochs=epochs,
        lr=lr,
        lr_tables=lr_tables,
        grad_clip=grad_clip,
        l2_weight=l2_weight,
        tbptt_k=tbptt_k,
        scalar_validate=scalar_validate,
        reject_worse=reject_worse,
        num_workers=num_workers,
        worker_torch_threads=worker_torch_threads,
        log_progress=log_progress,
    )

    oracle_params = np.stack([
        result['params'] for result in results
    ]).astype(np.float32)
    dc_soft_losses = np.array([
        result['baseline_soft_loss'] for result in results
    ], dtype=np.float32)
    oracle_soft_losses = np.array([
        result['oracle_soft_loss'] for result in results
    ], dtype=np.float32)
    dc_hard_losses = np.array([
        result['baseline_hard_loss'] for result in results
    ], dtype=np.float32)
    oracle_hard_losses = np.array([
        result['oracle_hard_loss'] for result in results
    ], dtype=np.float32)
    best_soft_epochs = np.array([
        result['best_soft_epoch'] for result in results
    ], dtype=np.int32)
    best_hard_epochs = np.array([
        result['best_hard_epoch'] for result in results
    ], dtype=np.int32)
    statuses = [result['status'] for result in results]
    delta_params = (oracle_params - baseline_params).astype(np.float32)
    improvement = (
        (oracle_hard_losses - dc_hard_losses)
        / np.maximum(dc_hard_losses, 1e-6)
    )

    epoch_rows = []
    for result in results:
        for row in result['epoch_rows']:
            epoch_rows.append({
                'trajectory_key': result['trajectory_key'],
                **row,
            })

    npz_path = os.path.join(output_dir, 'oracle_dataset.npz')
    csv_path = os.path.join(output_dir, 'oracle_dataset.csv')
    summary_path = os.path.join(output_dir, 'summary.yaml')
    epoch_csv_path = os.path.join(output_dir, 'oracle_epoch_trace.csv')

    _atomic_savez(
        npz_path,
        schema_version=np.array(DATASET_SCHEMA_VERSION, dtype=np.int32),
        selection_metric=np.array(SELECTION_METRIC),
        loss_reference_mode=np.array(LOSS_REFERENCE_MODE),
        run_fingerprint=np.array(run_spec['fingerprint']),
        trajectory_keys=np.array(keys),
        trajectory_manifest=np.array(trajectory_manifest or ''),
        trajectory_types=trajectory_types,
        trajectory_families=trajectory_families,
        trajectory_splits=trajectory_splits,
        speed_kph=speed_kph,
        param_names=param_names_arr,
        features=features,
        baseline_params=baseline_params,
        oracle_params=oracle_params,
        delta_params=delta_params,
        dc_soft_losses=dc_soft_losses,
        oracle_soft_losses=oracle_soft_losses,
        dc_hard_losses=dc_hard_losses,
        oracle_hard_losses=oracle_hard_losses,
        dc_losses=dc_hard_losses,
        oracle_scalar_losses=oracle_hard_losses,
        improvement_pct=improvement * 100.0,
        oracle_status=np.array(statuses),
        oracle_best_epoch=best_hard_epochs,
        oracle_best_hard_epoch=best_hard_epochs,
        oracle_best_soft_epoch=best_soft_epochs,
    )

    rows = []
    for i, key in enumerate(keys):
        row = {
            'trajectory_key': key,
            'trajectory_type': str(trajectory_types[i]),
            'trajectory_family': str(trajectory_families[i]),
            'trajectory_split': str(trajectory_splits[i]),
            'speed_kph': float(speed_kph[i]),
            'dc_soft_loss': float(dc_soft_losses[i]),
            'oracle_soft_loss': float(oracle_soft_losses[i]),
            'dc_hard_loss': float(dc_hard_losses[i]),
            'oracle_hard_loss': float(oracle_hard_losses[i]),
            'dc_loss': float(dc_hard_losses[i]),
            'oracle_scalar_loss': float(oracle_hard_losses[i]),
            'improvement_pct': float(improvement[i] * 100.0),
            'oracle_status': statuses[i],
            'oracle_best_epoch': int(best_hard_epochs[i]),
            'oracle_best_hard_epoch': int(best_hard_epochs[i]),
            'oracle_best_soft_epoch': int(best_soft_epochs[i]),
        }
        for j in range(PARAM_VECTOR_DIM):
            row[f'baseline_param_{j}'] = float(baseline_params[i, j])
            row[f'oracle_param_{j}'] = float(oracle_params[i, j])
            row[f'delta_param_{j}'] = float(delta_params[i, j])
        rows.append(row)
    _write_csv(csv_path, rows)
    _write_csv(epoch_csv_path, epoch_rows)

    status_counts = dict(Counter(statuses))
    elapsed_s = round(perf_counter() - started, 1)
    summary = {
        'schema_version': DATASET_SCHEMA_VERSION,
        'selection_metric': SELECTION_METRIC,
        'loss_reference_mode': LOSS_REFERENCE_MODE,
        'trajectory_count': total,
        'param_dim': PARAM_VECTOR_DIM,
        'feature_dim': int(features.shape[1]),
        'oracle_mode': 'per_trajectory_scalar_dc_hard_selected',
        'dc_equivalent_protocol': {
            'differentiable_training': True,
            'tbptt_k': int(tbptt_k),
            'loss_normalization': {
                'alpha': NORM_ALPHA,
                'floor': NORM_FLOOR,
            },
            'optimizer': 'Adam',
            'scheduler': 'CosineAnnealingLR',
            'hard_validation': True,
        },
        'epochs': int(epochs),
        'lr': float(lr),
        'lr_tables': float(lr_tables),
        'grad_clip': float(grad_clip),
        'l2_weight': float(l2_weight),
        'reject_worse': bool(reject_worse),
        'num_workers': int(num_workers),
        'worker_torch_threads': int(worker_torch_threads),
        'resumed_trajectory_count': int(resumed_count),
        'dc_soft_avg_loss': _nanmean_or_nan(dc_soft_losses),
        'oracle_soft_avg_loss': _nanmean_or_nan(oracle_soft_losses),
        'dc_hard_avg_loss': _nanmean_or_nan(dc_hard_losses),
        'oracle_hard_avg_loss': _nanmean_or_nan(oracle_hard_losses),
        'dc_avg_loss': _nanmean_or_nan(dc_hard_losses),
        'oracle_scalar_avg_loss': _nanmean_or_nan(oracle_hard_losses),
        'mean_improvement_pct': _nanmean_or_nan(improvement * 100.0),
        'trajectory_manifest': trajectory_manifest,
        'run_fingerprint': run_spec['fingerprint'],
        'run_spec': run_spec_path,
        'item_checkpoint_dir': item_dir,
        'progress_log': progress_log_path,
        'epoch_trace_csv': epoch_csv_path,
        'trajectory_type_counts': {
            str(k): int(v)
            for k, v in zip(*np.unique(
                trajectory_types, return_counts=True))
        },
        'oracle_status_counts': {
            str(k): int(v) for k, v in status_counts.items()
        },
        'total_time_s': elapsed_s,
    }
    _atomic_write_yaml(summary_path, summary)
    log_progress(f"wrote npz={npz_path}")
    log_progress(f"wrote csv={csv_path}")
    log_progress(f"wrote epoch_trace={epoch_csv_path}")
    log_progress(f"wrote summary={summary_path}")

    return {
        'output_dir': output_dir,
        'npz_path': npz_path,
        'csv_path': csv_path,
        'summary_path': summary_path,
        'epoch_csv_path': epoch_csv_path,
        'run_spec_path': run_spec_path,
        'summary': summary,
    }


def _load_cli_cfg(config_path: str | None, plant: str):
    cfg = load_config(config_path)
    apply_plant_override(cfg, plant)
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description='Generate E07 scalar DC-parameter oracle dataset')
    parser.add_argument('--plant', default='truck_trailer',
                        choices=['truck_trailer'])
    parser.add_argument('--config', default=DEFAULT_DC_CONFIG)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr-tables', type=float, default=None)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--l2-weight', type=float, default=0.01)
    parser.add_argument('--tbptt-k', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--worker-torch-threads', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-trajectories', type=int, default=None)
    parser.add_argument('--trajectory-manifest', default=None)
    parser.add_argument('--smoke-fixtures', action='store_true')
    parser.add_argument(
        '--no-scalar-validate',
        action='store_true',
        help='Skip the redundant final hard-loss consistency check; '
             'per-epoch hard selection remains enabled',
    )
    parser.add_argument(
        '--allow-worse',
        action='store_true',
        help='Diagnostic only: keep the best trained epoch even if its hard '
             'loss is worse than the global DC baseline',
    )
    args = parser.parse_args()

    cfg = _load_cli_cfg(args.config, args.plant)
    trajectory_metadata = None
    trajectory_manifest = None
    if args.smoke_fixtures:
        items = materialize_smoke_trajectories()
    elif args.trajectory_manifest:
        specs = load_e07_manifest(args.trajectory_manifest)
        if args.max_trajectories is not None:
            specs = specs[:int(args.max_trajectories)]
        items = materialize_e07_specs(specs)
        trajectory_metadata = metadata_by_key(specs)
        trajectory_manifest = args.trajectory_manifest
    else:
        items = materialize_standard_trajectories(args.max_trajectories)

    result = generate_oracle_dataset(
        cfg=cfg,
        trajectory_items=items,
        trajectory_metadata=trajectory_metadata,
        trajectory_manifest=trajectory_manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        lr_tables=args.lr_tables,
        grad_clip=args.grad_clip,
        l2_weight=args.l2_weight,
        tbptt_k=args.tbptt_k,
        scalar_validate=not args.no_scalar_validate,
        reject_worse=not args.allow_worse,
        num_workers=args.num_workers,
        worker_torch_threads=args.worker_torch_threads,
        resume=args.resume,
    )
    print(f"oracle npz: {result['npz_path']}")
    print(f"oracle csv: {result['csv_path']}")
    print(f"epoch trace: {result['epoch_csv_path']}")
    print(f"run spec: {result['run_spec_path']}")
    print(f"summary: {result['summary_path']}")


if __name__ == '__main__':
    main()
