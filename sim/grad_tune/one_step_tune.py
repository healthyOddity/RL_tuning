from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import apply_plant_override, load_config, save_tuned_config
from grad_tune.data_schema import default_run_id
from grad_tune.evaluate import run_evaluation
from grad_tune.param_policy import apply_gradient_step, freeze_except
from grad_tune.real_loss import (
    GradTuneLossWeights,
    compute_backward_loss,
    compute_real_report_loss,
)
from grad_tune.record_adapter import load_csv_sample, record_dir_to_csv
from grad_tune.refline_builder import build_trajectory_points
from optim.train import DiffControllerParams
from sim_loop import run_simulation


def _sim_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _make_run_dir(output_dir: Path, input_csv: str) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"{default_run_id(input_csv)}_{ts}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _maybe_disable_missing_truck_mlp(cfg: dict) -> str | None:
    if cfg.get('vehicle', {}).get('model_type') != 'truck_trailer':
        return None
    tt = cfg.get('truck_trailer_vehicle', {})
    ckpt = tt.get('checkpoint_path', '')
    if not ckpt:
        return None
    path = Path(ckpt)
    if not path.is_absolute():
        path = _sim_dir() / path
    if path.exists():
        return None
    fallback = _sim_dir() / 'configs' / 'checkpoints' / 'best_truck_trailer_error_model.pth'
    if fallback.exists():
        tt['checkpoint_path'] = 'configs/checkpoints/best_truck_trailer_error_model.pth'
        return f"truck_trailer checkpoint fallback used: {fallback}"
    tt['checkpoint_path'] = ''
    return f"truck_trailer checkpoint not found, disabled MLP: {path}"


def _to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.detach().item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def run_one_step(args) -> dict:
    if args.input_csv is None:
        if args.record_dir is None:
            raise ValueError("Either --input-csv or --record-dir is required")
        args.input_csv = str(record_dir_to_csv(args.record_dir))

    cfg = load_config(args.config)
    apply_plant_override(cfg, args.plant)
    warnings = []
    mlp_warning = _maybe_disable_missing_truck_mlp(cfg)
    if mlp_warning:
        warnings.append(mlp_warning)

    sample = load_csv_sample(
        args.input_csv, cfg, min_duration_s=args.min_duration,
        max_duration_s=args.window_duration,
        preprocess_mode=args.preprocess_mode,
        min_motion_speed=args.min_motion_speed,
        zero_threshold=args.zero_threshold,
        vy_jump_threshold=args.vy_jump_threshold,
        jump_dilate_frames=args.jump_dilate_frames)
    trajectory = build_trajectory_points(sample, source=args.refline_source)

    params = DiffControllerParams(cfg=cfg)
    freeze_except(params, args.param_mode)

    history = run_simulation(
        trajectory,
        init_speed=sample.init_v,
        init_x=sample.init_x,
        init_y=sample.init_y,
        init_yaw=sample.init_yaw,
        cfg=params.cfg,
        lat_ctrl=params.lat_ctrl,
        lon_ctrl=params.lon_ctrl,
        differentiable=True,
        tbptt_k=args.tbptt_k,
    )

    weights = GradTuneLossWeights(
        w_lat=args.w_lat,
        w_head=args.w_head,
        w_speed=args.w_speed,
        w_steer_rate=args.w_steer_rate,
        w_acc_rate=args.w_acc_rate,
    )
    real_loss, real_details = compute_real_report_loss(sample, weights)
    backward_loss, backward_details = compute_backward_loss(
        history, sample, weights, trajectory=trajectory)

    for p in params.parameters():
        if p.grad is not None:
            p.grad.zero_()
    backward_loss.backward()

    step_report = apply_gradient_step(
        params,
        args.param_mode,
        lr=args.lr,
        max_delta_ratio=args.max_delta_ratio,
        grad_clip=args.grad_clip,
    )

    cfg_out = params.to_config_dict()
    run_dir = _make_run_dir(Path(args.output_dir), args.input_csv)
    local_tuned_path = save_tuned_config(cfg_out, output_dir=str(run_dir), meta={
        'grad_tune': True,
        'input_csv': str(args.input_csv),
        'param_mode': args.param_mode,
        'real_loss': real_loss,
        'backward_loss': float(backward_loss.detach().item()),
        'plant': args.plant,
        'refline_source': args.refline_source,
    })

    tuned_copy_path = None
    if not args.no_tuned_copy:
        tuned_dir = Path(args.tuned_dir) if args.tuned_dir else _sim_dir() / 'configs' / 'tuned'
        tuned_copy_path = save_tuned_config(cfg_out, output_dir=str(tuned_dir), meta={
            'grad_tune': True,
            'source_run_tuned_path': str(local_tuned_path),
            'input_csv': str(args.input_csv),
            'param_mode': args.param_mode,
            'real_loss': real_loss,
            'backward_loss': float(backward_loss.detach().item()),
            'plant': args.plant,
            'refline_source': args.refline_source,
        })

    changed_files = [
        'sim/grad_tune/__init__.py',
        'sim/grad_tune/data_schema.py',
        'sim/grad_tune/record_adapter.py',
        'sim/grad_tune/preprocess.py',
        'sim/grad_tune/refline_builder.py',
        'sim/grad_tune/real_loss.py',
        'sim/grad_tune/param_policy.py',
        'sim/grad_tune/plotting.py',
        'sim/grad_tune/evaluate.py',
        'sim/grad_tune/one_step_tune.py',
        'sim/grad_tune/README.md',
        'sim/tests/test_grad_tune_refline.py',
        'sim/tests/test_grad_tune_evaluate.py',
        'sim/tests/test_grad_tune_loss.py',
        'sim/tests/test_grad_tune_param_policy.py',
        'sim/tests/test_grad_tune_smoke.py',
        'docs/grad_tune/specs/spec.md',
        'docs/grad_tune/specs/checklist.md',
        'docs/grad_tune/specs/tasks.md',
    ]

    report = {
        'input_csv': str(args.input_csv),
        'config': str(args.config),
        'plant': args.plant,
        'refline_source': args.refline_source,
        'param_mode': args.param_mode,
        'window': {
            'start_s': sample.window_start,
            'end_s': sample.window_end,
            'n_steps': sample.n_steps,
        },
        'preprocess': sample.preprocess_report,
        'real_loss': real_loss,
        'real_loss_details': real_details,
        'backward_loss': float(backward_loss.detach().item()),
        'backward_loss_details': backward_details,
        'step': step_report,
        'local_tuned_path': str(local_tuned_path),
        'standard_tuned_path': str(tuned_copy_path) if tuned_copy_path else None,
        'warnings': warnings,
        'record_smoothness_used_for_backward': False,
        'changed_files_for_git_add': changed_files,
    }
    report = _to_serializable(report)

    if args.eval_after:
        eval_args = argparse.Namespace(
            input_csv=args.input_csv,
            baseline_config=args.config,
            tuned_config=str(tuned_copy_path or local_tuned_path),
            plant=args.plant,
            refline_source=args.refline_source,
            output_dir=str(run_dir / 'evaluation'),
            window_duration=args.window_duration,
            min_duration=args.min_duration,
            preprocess_mode=args.preprocess_mode,
            min_motion_speed=args.min_motion_speed,
            zero_threshold=args.zero_threshold,
            vy_jump_threshold=args.vy_jump_threshold,
            jump_dilate_frames=args.jump_dilate_frames,
            no_plots=False,
            w_lat=args.w_lat,
            w_head=args.w_head,
            w_speed=args.w_speed,
            w_steer_rate=args.w_steer_rate,
            w_acc_rate=args.w_acc_rate,
        )
        report['evaluation'] = run_evaluation(eval_args)

    gradient_report = {
        'input_csv': str(args.input_csv),
        'plant': args.plant,
        'refline_source': args.refline_source,
        'param_mode': args.param_mode,
        'lr': args.lr,
        'grad_clip': args.grad_clip,
        'max_delta_ratio': args.max_delta_ratio,
        'backward_loss': float(backward_loss.detach().item()),
        'backward_loss_details': backward_details,
        'step': step_report,
    }
    gradient_report_yaml = run_dir / 'gradient_report.yaml'
    with open(gradient_report_yaml, 'w', encoding='utf-8') as f:
        yaml.safe_dump(_to_serializable(gradient_report), f,
                       allow_unicode=True, sort_keys=False)
    report['gradient_report_yaml'] = str(gradient_report_yaml)

    report_yaml = run_dir / 'grad_tune_report.yaml'
    report_txt = run_dir / 'grad_tune_report.txt'
    with open(report_yaml, 'w', encoding='utf-8') as f:
        yaml.safe_dump(report, f, allow_unicode=True, sort_keys=False)
    with open(report_txt, 'w', encoding='utf-8') as f:
        f.write(f"Grad tune input: {args.input_csv}\n")
        f.write(f"Param mode: {args.param_mode}\n")
        f.write(f"Real loss: {real_loss:.6f}\n")
        f.write(f"Backward loss: {float(backward_loss.detach().item()):.6f}\n")
        f.write(f"Local tuned path: {local_tuned_path}\n")
        f.write(f"Standard tuned path: {tuned_copy_path}\n")
        f.write("Changed files for git add:\n")
        for path in changed_files:
            f.write(f"- {path}\n")

    report['report_yaml'] = str(report_yaml)
    report['report_txt'] = str(report_txt)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='One-step proxy-gradient tuning from extracted real CSV')
    parser.add_argument('--input-csv', default=None)
    parser.add_argument('--record-dir', default=None)
    parser.add_argument('--config', default=str(_sim_dir() / 'configs' / 'default.yaml'))
    parser.add_argument('--plant', default='truck_trailer',
                        choices=['kinematic', 'dynamic', 'hybrid_dynamic',
                                 'hybrid_v2', 'truck_trailer',
                                 'truck_trailer_dynamics'])
    parser.add_argument('--refline-source', default='record',
                        choices=['record', 'csv'])
    parser.add_argument('--param-mode', default='all',
                        choices=['lat_offset', 'lon_speed', 'all'])
    parser.add_argument('--output-dir', default=str(_sim_dir() / 'results' / 'grad_tune'))
    parser.add_argument('--tuned-dir', default=str(_sim_dir() / 'configs' / 'tuned'))
    parser.add_argument('--no-tuned-copy', action='store_true')
    parser.add_argument('--eval-after', action='store_true',
                        help='Run baseline-vs-tuned evaluation on the same CSV refline')
    parser.add_argument('--window-duration', type=float, default=6.0)
    parser.add_argument('--min-duration', type=float, default=2.0)
    parser.add_argument('--preprocess-mode', default='combined',
                        choices=['combined', 'controller', 'none'])
    parser.add_argument('--min-motion-speed', type=float, default=0.05)
    parser.add_argument('--zero-threshold', type=float, default=1e-7)
    parser.add_argument('--vy-jump-threshold', type=float, default=2.0)
    parser.add_argument('--jump-dilate-frames', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--max-delta-ratio', type=float, default=0.05)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--tbptt-k', type=int, default=150)
    parser.add_argument('--w-lat', type=float, default=10.0)
    parser.add_argument('--w-head', type=float, default=8.0)
    parser.add_argument('--w-speed', type=float, default=3.0)
    parser.add_argument('--w-steer-rate', type=float, default=0.05)
    parser.add_argument('--w-acc-rate', type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    try:
        report = run_one_step(parse_args(argv))
    except Exception as exc:
        print(f"grad_tune failed: {exc}", file=sys.stderr)
        return 1
    print(f"local_tuned_path: {report['local_tuned_path']}")
    if report.get('standard_tuned_path'):
        print(f"standard_tuned_path: {report['standard_tuned_path']}")
    print(f"report_yaml: {report['report_yaml']}")
    if report.get('evaluation'):
        print(f"eval_report: {report['evaluation']['report_yaml']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
