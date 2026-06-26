"""Compare truck_trailer and truck_deeponet without tuning.

Outputs:
- real_replay_metrics.csv: open-loop replay against a real interpolated CSV.
- standard_48_metrics.csv: closed-loop tracking metrics on the 48 standard
  synthetic trajectories.
- summary.yaml and basic plots in the selected output directory.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import apply_plant_override, load_config
from model.trajectory import TRAJECTORY_TYPES, expand_trajectories
from model.truck_deeponet_vehicle import CONTROL_NAMES, HISTORY_FRAME_COUNT
from model.vehicle_factory import create_vehicle
from optim.post_training import _plot_comparison_grid
from sim_loop import run_simulation


PLANTS = ("truck_trailer", "truck_deeponet")
DEFAULT_REAL_CSV = Path(
    r"C:\Users\huangjiangyu\Desktop\hirain\L4"
    r"\train_file\data_process\data\turn\20kph"
    r"\20260417_162035.00000_interpolated.csv"
)


@dataclass
class RealReplayData:
    time: np.ndarray
    state: np.ndarray
    control: np.ndarray
    dt: np.ndarray


@dataclass
class StandardScenario:
    key: str
    label: str
    type_name: str
    speed_kph: int
    generator: object


def _first_existing(frame: pd.DataFrame, names: list[str]) -> str | None:
    if not names:
        return None
    lower_map = {name.lower(): name for name in frame.columns}
    for name in names:
        if name in frame.columns:
            return name
        mapped = lower_map.get(name.lower())
        if mapped is not None:
            return mapped
    return None


def _read_numeric(frame: pd.DataFrame, names: list[str],
                  field_name: str) -> np.ndarray:
    column = _first_existing(frame, names)
    if column is None:
        raise ValueError(f"Missing required column for {field_name}: {names}")
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values.ffill().bfill().fillna(0.0)
    return values.to_numpy(dtype=np.float32)


def _wrap_angle_np(angle: np.ndarray) -> np.ndarray:
    return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def load_real_replay_data(csv_path: str | Path) -> RealReplayData:
    """Load interpolated truck CSV as state/control rollout data.

    State is [x, y, yaw_rad, vx, vy, yaw_rate_radps].
    Control is [steer_sw_rad, torque_fl, torque_fr, torque_rl, torque_rr].
    """
    frame = pd.read_csv(csv_path)
    time = _read_numeric(frame, ["timestamp", "time", "time_s"], "time")
    if time.size < 2:
        raise ValueError(f"Not enough rows in real replay csv: {csv_path}")
    if float(np.nanmax(time)) > 1.0e6:
        time = (time - time[0]) / 1000.0

    x = _read_numeric(frame, ["position_enu.x", "X_t_m", "X_m"], "x")
    y = _read_numeric(frame, ["position_enu.y", "Y_t_m", "Y_m"], "y")
    yaw_deg = _read_numeric(
        frame, ["euler_angles.z", "heading", "Yaw_t_deg", "Yaw_deg"], "yaw")
    yaw = np.deg2rad(yaw_deg).astype(np.float32)
    vx = _read_numeric(frame, ["vx_veh", "Vx_t_mps", "Vx_mps"], "vx")
    vy = _read_numeric(frame, ["vy", "Vy_t_mps", "Vy_mps"], "vy")

    if _first_existing(frame, ["diff_yawrate"]) is not None:
        yaw_rate = _read_numeric(frame, ["diff_yawrate"], "yaw rate")
    else:
        yaw_rate_degps = _read_numeric(
            frame,
            ["VehicleInfoBDData.BD18F0090B_VDC2_YawRate",
             "YawRate_t_degps", "YawRate_degps"],
            "yaw rate")
        yaw_rate = np.deg2rad(yaw_rate_degps).astype(np.float32)

    steer_sw = _read_numeric(
        frame,
        ["Steer_SW_rad", "SteeringWheel_rad",
         "VehicleInfoADData.AD18F0090B_VDC2_SteeringWheelAngle"],
        "steering wheel angle")
    torque_fl = _read_numeric(frame, ["torque_fl", "Torque_FL_Nm_cmd"],
                              "front-left torque")
    torque_fr = _read_numeric(frame, ["torque_fr", "Torque_FR_Nm_cmd"],
                              "front-right torque")
    torque_rl = _read_numeric(frame, ["torque_rl", "Torque_RL_Nm_cmd"],
                              "rear-left torque")
    torque_rr = _read_numeric(frame, ["torque_rr", "Torque_RR_Nm_cmd"],
                              "rear-right torque")

    state = np.column_stack([x, y, yaw, vx, vy, yaw_rate]).astype(np.float32)
    control = np.column_stack(
        [steer_sw, torque_fl, torque_fr, torque_rl, torque_rr]).astype(
            np.float32)

    dt = np.diff(time.astype(np.float64))
    positive = dt[dt > 1.0e-6]
    fallback = float(np.median(positive)) if positive.size else 0.02
    dt = np.where(dt > 1.0e-6, dt, fallback).astype(np.float32)
    return RealReplayData(
        time=time.astype(np.float32),
        state=state,
        control=control[:-1].copy(),
        dt=dt,
    )


def build_standard_48_scenarios() -> list[StandardScenario]:
    scenarios: list[StandardScenario] = []
    for key, label, generator in expand_trajectories(TRAJECTORY_TYPES):
        type_name, speed_token = key.rsplit("_", 1)
        speed_kph = int(speed_token.replace("kph", ""))
        scenarios.append(StandardScenario(
            key=key,
            label=label,
            type_name=type_name,
            speed_kph=speed_kph,
            generator=generator,
        ))
    return scenarios


def compute_tracking_metrics(history: list[dict],
                             w_lat: float = 10.0,
                             w_head: float = 8.0) -> dict[str, float]:
    lat = np.asarray([float(item["lateral_error"]) for item in history],
                     dtype=np.float64)
    head = np.asarray([float(item["heading_error"]) for item in history],
                      dtype=np.float64)
    lat_mse = float(np.mean(lat * lat))
    head_mse = float(np.mean(head * head))
    return {
        "lat_rmse": math.sqrt(lat_mse),
        "head_rmse": math.sqrt(head_mse),
        "lat_max": float(np.max(np.abs(lat))),
        "head_max": float(np.max(np.abs(head))),
        "tracking_loss": w_lat * lat_mse + w_head * head_mse,
    }


def _config_for_plant(base_cfg: dict, plant: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    apply_plant_override(cfg, plant)
    return cfg


def _reset_deeponet_history(car) -> None:
    from collections import deque

    zero_control = car._state.new_zeros(len(CONTROL_NAMES))
    car._history_states = deque(
        [car._state.clone() for _ in range(HISTORY_FRAME_COUNT)],
        maxlen=HISTORY_FRAME_COUNT)
    car._history_controls = deque(
        [zero_control.clone() for _ in range(HISTORY_FRAME_COUNT)],
        maxlen=HISTORY_FRAME_COUNT)


def _make_replay_vehicle(cfg: dict, plant: str, state0: np.ndarray,
                         dt: float):
    speed0 = float(math.hypot(float(state0[3]), float(state0[4])))
    car = create_vehicle(
        cfg,
        x=float(state0[0]),
        y=float(state0[1]),
        yaw=float(state0[2]),
        v=speed0,
        dt=float(dt),
        differentiable=False,
    )
    state_t = torch.as_tensor(state0, dtype=torch.float32)
    if plant == "truck_deeponet":
        car._state = state_t.clone()
        _reset_deeponet_history(car)
    elif plant == "truck_trailer":
        car._state = torch.cat([state_t, state_t.clone()])
    else:
        raise ValueError(f"Unsupported replay plant: {plant}")
    return car


def _tractor_state_from_vehicle(car, plant: str) -> np.ndarray:
    if plant in ("truck_trailer", "truck_deeponet") and hasattr(car, "_state"):
        return car._state[:6].detach().cpu().numpy().astype(np.float32)
    return np.asarray([
        float(car.x.item()),
        float(car.y.item()),
        float(car.yaw.item()),
        float(car.v.item()),
        0.0,
        float(car.yawrate.item()),
    ], dtype=np.float32)


def replay_real_data(data: RealReplayData, base_cfg: dict,
                     max_steps: int | None = None) -> tuple[list[dict], dict]:
    transition_count = int(data.control.shape[0])
    if max_steps is not None:
        transition_count = min(transition_count, int(max_steps))
    results: list[dict] = []
    trajectories: dict[str, np.ndarray] = {}

    for plant in PLANTS:
        cfg = _config_for_plant(base_cfg, plant)
        steering_ratio = float(
            cfg[f"{plant}_vehicle"]["steering_ratio"]
            if plant == "truck_deeponet"
            else cfg["truck_trailer_vehicle"]["steering_ratio"])
        car = _make_replay_vehicle(cfg, plant, data.state[0], data.dt[0])
        pred_states = [_tractor_state_from_vehicle(car, plant)]
        for index in range(transition_count):
            control = data.control[index]
            car.dt = float(data.dt[index])
            delta_front = float(control[0]) / steering_ratio
            torque_wheel = float(control[3] + control[4])
            car.step(delta=delta_front, torque_wheel=torque_wheel)
            pred_states.append(_tractor_state_from_vehicle(car, plant))

        pred = np.asarray(pred_states, dtype=np.float32)
        truth = data.state[:pred.shape[0]].astype(np.float32)
        err_xy = pred[:, :2] - truth[:, :2]
        err_yaw = _wrap_angle_np(pred[:, 2] - truth[:, 2])
        err_vx = pred[:, 3] - truth[:, 3]
        err_vy = pred[:, 4] - truth[:, 4]
        err_r = pred[:, 5] - truth[:, 5]
        xy_step = np.sqrt(np.sum(err_xy * err_xy, axis=1))
        results.append({
            "plant": plant,
            "samples": int(pred.shape[0]),
            "duration_s": float(data.time[pred.shape[0] - 1] - data.time[0]),
            "xy_rmse_m": float(np.sqrt(np.mean(xy_step * xy_step))),
            "x_rmse_m": float(np.sqrt(np.mean(err_xy[:, 0] ** 2))),
            "y_rmse_m": float(np.sqrt(np.mean(err_xy[:, 1] ** 2))),
            "yaw_rmse_rad": float(np.sqrt(np.mean(err_yaw ** 2))),
            "vx_rmse_mps": float(np.sqrt(np.mean(err_vx ** 2))),
            "vy_rmse_mps": float(np.sqrt(np.mean(err_vy ** 2))),
            "yawrate_rmse_radps": float(np.sqrt(np.mean(err_r ** 2))),
            "final_xy_error_m": float(xy_step[-1]),
        })
        trajectories[plant] = pred
    return results, trajectories


def evaluate_standard_48(base_cfg: dict,
                         plot_output_dir: str | Path | None = None) -> list[dict]:
    rows: list[dict] = []
    all_base = []
    all_deeponet = []
    for scenario in build_standard_48_scenarios():
        trajectory = scenario.generator()
        init_speed = float(trajectory[0].v)
        scenario_outputs = {}
        for plant in PLANTS:
            cfg = _config_for_plant(base_cfg, plant)
            history = run_simulation(
                trajectory,
                init_speed=init_speed,
                cfg=cfg,
                differentiable=False,
            )
            metrics = compute_tracking_metrics(history)
            scenario_outputs[plant] = (history, metrics)
            rows.append({
                "scenario": scenario.key,
                "label": scenario.label,
                "type": scenario.type_name,
                "speed_kph": scenario.speed_kph,
                "plant": plant,
                "samples": len(history),
                **metrics,
            })
        all_base.append((
            scenario.key,
            scenario.label,
            trajectory,
            scenario_outputs["truck_trailer"][0],
            scenario_outputs["truck_trailer"][1],
            init_speed,
        ))
        all_deeponet.append((
            scenario.key,
            scenario.label,
            trajectory,
            scenario_outputs["truck_deeponet"][0],
            scenario_outputs["truck_deeponet"][1],
            init_speed,
        ))

    if plot_output_dir is not None:
        plot_dir = Path(plot_output_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        _plot_comparison_grid(
            all_base, all_deeponet, str(plot_dir),
            plot_type="trajectory", filename="standard_48_trajectory.png")
        _plot_comparison_grid(
            all_base, all_deeponet, str(plot_dir),
            plot_type="lateral_error", filename="standard_48_lateral_error.png")
        _plot_comparison_grid(
            all_base, all_deeponet, str(plot_dir),
            plot_type="speed_error", filename="standard_48_speed_error.png")
        _plot_comparison_grid(
            all_base, all_deeponet, str(plot_dir),
            plot_type="steer", filename="standard_48_steer.png")
        _plot_comparison_grid(
            all_base, all_deeponet, str(plot_dir),
            plot_type="acc", filename="standard_48_acc.png")
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_standard(rows: list[dict]) -> dict:
    summary = {}
    for plant in PLANTS:
        plant_rows = [row for row in rows if row["plant"] == plant]
        summary[plant] = {
            "scenario_count": len(plant_rows),
            "mean_lat_rmse": float(np.mean([r["lat_rmse"] for r in plant_rows])),
            "mean_head_rmse": float(np.mean([r["head_rmse"] for r in plant_rows])),
            "mean_tracking_loss": float(np.mean(
                [r["tracking_loss"] for r in plant_rows])),
        }
    return summary


def _plot_real_replay(path: Path, data: RealReplayData,
                      trajectories: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    truth = data.state
    ax.plot(truth[:, 0], truth[:, 1], "k-", label="real", linewidth=2.0)
    for plant, pred in trajectories.items():
        ax.plot(pred[:, 0], pred[:, 1], label=plant, linewidth=1.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Real replay trajectory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_standard_loss(path: Path, rows: list[dict]) -> None:
    summary = _summarize_standard(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    plants = list(PLANTS)
    values = [summary[plant]["mean_tracking_loss"] for plant in plants]
    ax.bar(plants, values)
    ax.set_ylabel("mean tracking loss")
    ax.set_title("Standard 48 closed-loop comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_vehicle_model_comparison(real_csv: str | Path,
                                 output_dir: str | Path | None = None,
                                 max_real_steps: int | None = None,
                                 skip_real: bool = False,
                                 skip_standard: bool = False) -> dict:
    sim_dir = Path(__file__).resolve().parents[1]
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = sim_dir / "results" / "model_compare" / timestamp
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config()
    summary = {
        "real_csv": str(real_csv),
        "output_dir": str(output_path),
        "plants": list(PLANTS),
    }

    if not skip_real:
        real_data = load_real_replay_data(real_csv)
        real_rows, trajectories = replay_real_data(
            real_data, base_cfg, max_steps=max_real_steps)
        _write_csv(output_path / "real_replay_metrics.csv", real_rows)
        _plot_real_replay(output_path / "real_replay_trajectory.png",
                          real_data, trajectories)
        summary["real_replay"] = real_rows

    if not skip_standard:
        standard_rows = evaluate_standard_48(
            base_cfg, plot_output_dir=output_path)
        _write_csv(output_path / "standard_48_metrics.csv", standard_rows)
        _plot_standard_loss(output_path / "standard_48_loss.png",
                            standard_rows)
        summary["standard_48"] = _summarize_standard(standard_rows)

    with (output_path / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare truck_trailer and truck_deeponet plants.")
    parser.add_argument("--real-csv", default=str(DEFAULT_REAL_CSV),
                        help="Interpolated real CSV for open-loop replay.")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory. Defaults to results/model_compare/<timestamp>.")
    parser.add_argument("--max-real-steps", type=int, default=None,
                        help="Limit real replay transitions for a quick smoke run.")
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--skip-standard", action="store_true")
    args = parser.parse_args(argv)

    summary = run_vehicle_model_comparison(
        real_csv=args.real_csv,
        output_dir=args.output_dir,
        max_real_steps=args.max_real_steps,
        skip_real=args.skip_real,
        skip_standard=args.skip_standard,
    )
    print(f"Results saved to: {summary['output_dir']}")
    if "standard_48" in summary:
        for plant, metrics in summary["standard_48"].items():
            print(
                f"{plant}: mean_lat_rmse={metrics['mean_lat_rmse']:.4f}, "
                f"mean_loss={metrics['mean_tracking_loss']:.4f}")
    if "real_replay" in summary:
        for row in summary["real_replay"]:
            print(
                f"{row['plant']} real replay: xy_rmse={row['xy_rmse_m']:.3f}m, "
                f"final_xy={row['final_xy_error_m']:.3f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
