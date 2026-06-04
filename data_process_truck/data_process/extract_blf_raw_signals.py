#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decode selected BLF CAN signals and align them to a uniform raw CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_SIGNALS = [
    {"message": "VCU_2", "signal": "VCU_VehicleCtrlMod", "column": "controller_enable_source"},
    {"message": "ADU_2", "signal": "ADU_TotalMotorRequestTorque", "column": "total_motor_request_torque_nm"},
    {"message": "EEC1", "signal": "VCU_OutputTorque", "column": "motor_output_torque_nm"},
    {"message": "EEC1", "signal": "VCU_OutputTorqueRtio", "column": "motor_output_torque_ratio_pct"},
    {"message": "TCU_Command1", "signal": "TCU_MotorRequestTorque", "column": "tcu_motor_request_torque_nm"},
    {"message": "VCU_MC1", "signal": "VCU_MotorRequestTorque", "column": "vcu_motor_request_torque_nm"},
    {"message": "VCU_VehSts", "signal": "TCU_CurrentGear", "column": "current_gear"},
    {"message": "EHPS_2", "signal": "EHPS_SteerWheelAng", "column": "steer_wheel_angle_abs_deg"},
    {"message": "EHPS_2", "signal": "SteerWheelAngDir", "column": "steer_wheel_angle_dir"},
    {"message": "CCVS1", "signal": "CCVS_VehSpd", "column": "vehicle_speed_kmh"},
    {"message": "VDC2", "signal": "VDC2_YawRate", "column": "yaw_rate_radps"},
    {"message": "VDC2", "signal": "VDC2_SteeringWheelAngle", "column": "vdc2_steer_angle_rad"},
    {"message": "ADU_6", "signal": "ADU_AngCmdReqValue", "column": "adu_target_steer_deg"},
    {"message": "ADU_6", "signal": "ADU_TrqCmdReqValue", "column": "adu_steer_torque_nm"},
    {"message": "ADU_6", "signal": "ADU_LatADCtrlMode", "column": "adu_lat_ctrl_mode"},
    {"message": "ADU_6", "signal": "ADU_SteerValid", "column": "adu_steer_valid"},
    {"message": "ADU_1", "signal": "ADU_RequestTargetSpeed", "column": "adu_target_speed_mps"},
    {"message": "ADU_1", "signal": "ADU_RequestTargetLongiPstn", "column": "adu_target_longitudinal_position_m"},
    {"message": "ADU_1", "signal": "ADU_RequestTargetAcceleration", "column": "adu_target_acc_mps2"},
    {"message": "TBox_2", "signal": "TBox_longitudeInfo", "column": "longitude_deg"},
    {"message": "TBox_2", "signal": "TBox_Latitude", "column": "latitude_deg"},
    {"message": "INS_A", "signal": "INS570D_INS_Latitude", "column": "ins_latitude_deg"},
    {"message": "INS_A", "signal": "INS570D_INS_Longitude", "column": "ins_longitude_deg"},
    {"message": "INS_A", "signal": "INS570D_INS_HeadingAngle", "column": "ins_heading_deg"},
    {"message": "INS_A", "signal": "INS570D_INS_PitchAngle", "column": "ins_pitch_deg"},
    {"message": "INS_A", "signal": "INS570D_INS_RollAngle", "column": "ins_roll_deg"},
    {"message": "INS_A", "signal": "INS570D_INS_VBx", "column": "ins_vbx_mps"},
    {"message": "INS_A", "signal": "INS570D_INS_VBy", "column": "ins_vby_mps"},
    {"message": "INS1", "signal": "INS1_Latitude", "column": "ins_latitude_deg"},
    {"message": "INS1", "signal": "INS1_Longitude", "column": "ins_longitude_deg"},
    {"message": "INS1", "signal": "INS1_HeadingAngle", "column": "ins_heading_deg"},
    {"message": "INS1", "signal": "INS1_PitchAngle", "column": "ins_pitch_deg"},
    {"message": "INS1", "signal": "INS1_RollAngle", "column": "ins_roll_deg"},
    {"message": "INS1", "signal": "INS1_EastSpd", "column": "ins_east_speed_mps"},
    {"message": "INS1", "signal": "INS1_NorthSpd", "column": "ins_north_speed_mps"},
]


@dataclass
class DbcSpec:
    path: Path
    db: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode BLF signals to aligned raw CSV")
    parser.add_argument("--dbc", required=True, type=Path, nargs="+", help="DBC file(s) or directory/directories")
    parser.add_argument("--blf", required=True, type=Path, help="BLF file")
    parser.add_argument("--signals", type=Path, help="Optional structured YAML signal config")
    parser.add_argument("--output", required=True, type=Path, help="Aligned raw CSV output path")
    parser.add_argument("--long-output", type=Path, help="Optional long-form decoded signal CSV")
    parser.add_argument("--sample-period-ms", type=float, default=10.0, help="Aligned output sample period")
    parser.add_argument("--max-gap-ms", type=float, default=50.0, help="Nearest sample max time gap")
    return parser.parse_args()


def _import_cantools():
    try:
        import cantools  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing cantools. Do not install without author approval.") from exc
    return cantools


def _import_can():
    try:
        import can  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing python-can. Do not install without author approval.") from exc
    return can


def expand_dbc_paths(paths: list[Path]) -> list[Path]:
    dbcs: list[Path] = []
    for path in paths:
        if path.is_file():
            dbcs.append(path)
        elif path.is_dir():
            dbcs.extend(sorted(p for p in path.rglob("*.dbc") if p.is_file()))
        else:
            raise FileNotFoundError(f"DBC path does not exist: {path}")
    return sorted(dict.fromkeys(dbcs))


def load_dbcs(paths: list[Path]) -> list[DbcSpec]:
    cantools = _import_cantools()
    specs = []
    for path in expand_dbc_paths(paths):
        specs.append(DbcSpec(path=path, db=cantools.database.load_file(str(path), strict=False)))
    return specs


def load_signals(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return DEFAULT_SIGNALS
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        return DEFAULT_SIGNALS
    if isinstance(data, dict):
        items = data.get("signals", data.get("target_columns", []))
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("signals YAML must be a list or contain a signals list")
    signals = []
    for item in items:
        if not isinstance(item, dict):
            continue
        message = item.get("message") or item.get("message_name")
        signal = item.get("signal") or item.get("signal_name")
        column = item.get("column") or item.get("target_column") or f"{message}.{signal}"
        if message and signal:
            signals.append({"message": str(message), "signal": str(signal), "column": str(column)})
    return signals or DEFAULT_SIGNALS


def build_message_index(specs: list[DbcSpec]) -> dict[int, list[Any]]:
    index: dict[int, list[Any]] = defaultdict(list)
    seen: set[tuple[int, str]] = set()
    for spec in specs:
        for msg in spec.db.messages:
            key = (int(msg.frame_id), msg.name)
            if key in seen:
                continue
            seen.add(key)
            index[int(msg.frame_id)].append(msg)
    return index


def decode_blf(
    blf_path: Path,
    specs: list[DbcSpec],
    signals: list[dict[str, str]],
) -> tuple[dict[str, dict[str, list[float]]], list[dict[str, Any]], dict[str, Any]]:
    can = _import_can()
    message_index = build_message_index(specs)
    wanted = {(s["message"], s["signal"]): s["column"] for s in signals}
    series: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"timestamps": [], "values": []})
    long_rows: list[dict[str, Any]] = []
    stats = {"frames_total": 0, "frames_matched": 0, "frames_decoded": 0, "samples": defaultdict(int)}

    with can.BLFReader(str(blf_path)) as reader:
        for frame in reader:
            stats["frames_total"] += 1
            frame_id = int(frame.arbitration_id)
            messages = message_index.get(frame_id, [])
            if not messages:
                continue
            stats["frames_matched"] += 1
            for dbc_msg in messages:
                relevant = [key for key in wanted if key[0] == dbc_msg.name]
                if not relevant:
                    continue
                try:
                    decoded = dbc_msg.decode(bytes(frame.data), decode_choices=False)
                except Exception:
                    continue
                stats["frames_decoded"] += 1
                ts_ms = float(frame.timestamp) * 1000.0
                for message_name, signal_name in relevant:
                    if signal_name not in decoded:
                        continue
                    value = decoded[signal_name]
                    try:
                        value_f = float(value)
                    except (TypeError, ValueError):
                        continue
                    column = wanted[(message_name, signal_name)]
                    series[column]["timestamps"].append(ts_ms)
                    series[column]["values"].append(value_f)
                    stats["samples"][column] += 1
                    long_rows.append({
                        "timestamp_ms": ts_ms,
                        "channel": getattr(frame, "channel", ""),
                        "can_id": f"0x{frame_id:X}",
                        "message": message_name,
                        "signal": signal_name,
                        "column": column,
                        "value": value_f,
                    })
    return series, long_rows, stats


def unique_sorted(data: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    pairs = sorted(zip(data["timestamps"], data["values"]), key=lambda x: x[0])
    if not pairs:
        return np.array([], dtype=float), np.array([], dtype=float)
    ts: list[float] = []
    vals: list[float] = []
    for t, v in pairs:
        if ts and t == ts[-1]:
            vals[-1] = v
        else:
            ts.append(float(t))
            vals.append(float(v))
    return np.array(ts), np.array(vals)


def nearest_align(base_ts: np.ndarray, src_ts: np.ndarray, src_vals: np.ndarray, max_gap_ms: float) -> np.ndarray:
    out = np.full(len(base_ts), np.nan)
    if len(src_ts) == 0:
        return out
    idx = np.searchsorted(src_ts, base_ts, side="left")
    right = np.clip(idx, 0, len(src_ts) - 1)
    left = np.clip(idx - 1, 0, len(src_ts) - 1)
    choose_left = np.abs(base_ts - src_ts[left]) <= np.abs(src_ts[right] - base_ts)
    nearest = np.where(choose_left, left, right)
    gap = np.abs(src_ts[nearest] - base_ts)
    valid = gap <= max_gap_ms
    out[valid] = src_vals[nearest[valid]]
    return out


def write_long(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["timestamp_ms", "channel", "can_id", "message", "signal", "column", "value"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_aligned(
    series: dict[str, dict[str, list[float]]],
    signals: list[dict[str, str]],
    output_path: Path,
    sample_period_ms: float,
    max_gap_ms: float,
) -> None:
    all_ts = [t for data in series.values() for t in data["timestamps"]]
    if not all_ts:
        raise RuntimeError("No selected signals were decoded from BLF")
    start = min(all_ts)
    end = max(all_ts)
    base_ts = np.arange(start, end + sample_period_ms / 2.0, sample_period_ms)

    columns = ["timestamp_ms", "timestamp_s"] + [s["column"] for s in signals]
    aligned: dict[str, np.ndarray] = {
        "timestamp_ms": base_ts,
        "timestamp_s": (base_ts - base_ts[0]) / 1000.0,
    }
    for sig in signals:
        column = sig["column"]
        src_ts, src_vals = unique_sorted(series.get(column, {"timestamps": [], "values": []}))
        aligned[column] = nearest_align(base_ts, src_ts, src_vals, max_gap_ms)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i in range(len(base_ts)):
            row = {}
            for col in columns:
                val = aligned[col][i]
                row[col] = "" if isinstance(val, float) and np.isnan(val) else val
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    try:
        specs = load_dbcs(args.dbc)
        signals = load_signals(args.signals)
        series, long_rows, stats = decode_blf(args.blf, specs, signals)
        if args.long_output:
            write_long(long_rows, args.long_output)
        write_aligned(series, signals, args.output, args.sample_period_ms, args.max_gap_ms)
        print(f"decoded frames: {stats['frames_decoded']} / matched frames: {stats['frames_matched']} / total frames: {stats['frames_total']}")
        for column, count in sorted(stats["samples"].items()):
            print(f"{column}: {count}")
        print(f"aligned raw csv: {args.output}")
        if args.long_output:
            print(f"long raw csv: {args.long_output}")
        return 0
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
