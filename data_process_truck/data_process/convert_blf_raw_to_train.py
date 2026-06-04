#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert decoded BLF raw signals to the existing truck training CSV format."""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VEHICLE_SPEED_COL = "VehicleInfoBDData.BDCFF10D0_VCU_VehicleSpeed"
YAW_RATE_COL = "VehicleInfoBDData.BD18F0090B_VDC2_YawRate"
PATH_REMAIN_COL = "debug.simple_lon_debug.path_remain"
GEAR_RATIO = {
    1: 39.8 * 0.95,
    2: 95.4 * 0.95,
    3: 39.8 * 0.95,
    4: 19.4 * 0.95,
    5: 11.2 * 0.95,
}
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
ORIGIN_CANDIDATES = [
    {"name": "tianjin", "lon": 117.064613, "lat": 39.068178, "height": 0.0},
    {"name": "beijing", "lon": 116.714101539, "lat": 40.181167491, "height": 0.0},
]
SUPPLEMENT_COLUMNS = ["ins_pitch_deg", "ins_roll_deg", "ins_vbx_mps", "ins_vby_mps"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BLF raw CSV to interpolated/train CSV")
    parser.add_argument("--raw", required=True, type=Path, help="raw_aligned.csv from extract_blf_raw_signals.py")
    parser.add_argument("--interpolated-output", required=True, type=Path, help="Output *_interpolated.csv path")
    parser.add_argument("--train-output", type=Path, help="Output train CSV path")
    parser.add_argument("--report-output", type=Path, help="Optional JSON report path")
    parser.add_argument("--sample-period-ms", type=float, default=20.0, help="Output sample period, default 20ms/50Hz")
    parser.add_argument(
        "--controller-enabled-values",
        default="3",
        help="Comma-separated values treated as controller_enable=1, default: 3",
    )
    parser.add_argument("--segment", action="store_true", help="Enable csvdata_new_truck segmentation")
    return parser.parse_args()


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def first_existing_nonempty(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns and not df[name].isna().all():
            return name
    return None


def nearest_resample(raw: pd.DataFrame, sample_period_ms: float) -> pd.DataFrame:
    if "timestamp_ms" not in raw.columns:
        raise ValueError("raw CSV must contain timestamp_ms")
    if raw.empty:
        raise ValueError("raw CSV is empty")
    raw = raw.sort_values("timestamp_ms").drop_duplicates("timestamp_ms", keep="first").reset_index(drop=True)
    start = float(raw["timestamp_ms"].iloc[0])
    end = float(raw["timestamp_ms"].iloc[-1])
    base_ts = np.arange(start, end + sample_period_ms * 0.5, sample_period_ms)
    nearest_idx = np.searchsorted(raw["timestamp_ms"].to_numpy(), base_ts, side="left")
    nearest_idx = np.clip(nearest_idx, 0, len(raw) - 1)
    prev_idx = np.clip(nearest_idx - 1, 0, len(raw) - 1)
    raw_ts = raw["timestamp_ms"].to_numpy()
    choose_prev = np.abs(raw_ts[prev_idx] - base_ts) <= np.abs(raw_ts[nearest_idx] - base_ts)
    nearest_idx = np.where(choose_prev, prev_idx, nearest_idx)
    out = raw.iloc[nearest_idx].copy().reset_index(drop=True)
    out["timestamp_ms"] = base_ts
    return out


def diff_angle(prev_angle: float, next_angle: float) -> float:
    diff = next_angle - prev_angle
    if diff > 180:
        return diff - 360
    if diff < -180:
        return diff + 360
    return diff


def compute_vehicle_velocity(df: pd.DataFrame, x_col: str, y_col: str, yaw_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = df["timestamp_ms"].to_numpy(dtype=float)
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    yaw = np.deg2rad(df[yaw_col].to_numpy(dtype=float))
    n = len(df)
    vx_enu = np.zeros(n)
    vy_enu = np.zeros(n)
    for i in range(n):
        if n == 1:
            dt = 0.0
            dx = dy = 0.0
        elif i == 0:
            dt = (ts[1] - ts[0]) / 1000.0
            dx = x[1] - x[0]
            dy = y[1] - y[0]
        elif i == n - 1:
            dt = (ts[i] - ts[i - 1]) / 1000.0
            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
        else:
            dt = (ts[i + 1] - ts[i - 1]) / 1000.0
            dx = x[i + 1] - x[i - 1]
            dy = y[i + 1] - y[i - 1]
        if dt > 0:
            vx_enu[i] = dx / dt
            vy_enu[i] = dy / dt
    vx_veh = vx_enu * np.cos(yaw) + vy_enu * np.sin(yaw)
    vy_veh = -vx_enu * np.sin(yaw) + vy_enu * np.cos(yaw)
    return vx_enu, vy_enu, vy_veh


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371000.0
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def choose_origin(lon: pd.Series, lat: pd.Series) -> dict[str, float | str]:
    valid = pd.DataFrame({"lon": lon, "lat": lat}).dropna()
    if valid.empty:
        return ORIGIN_CANDIDATES[0]
    first_lon = float(valid["lon"].iloc[0])
    first_lat = float(valid["lat"].iloc[0])
    return min(
        ORIGIN_CANDIDATES,
        key=lambda origin: haversine_m(first_lon, first_lat, float(origin["lon"]), float(origin["lat"])),
    )


def geodetic_to_ecef(lon_deg: np.ndarray, lat_deg: np.ndarray, height_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + height_m) * cos_lat * np.cos(lon)
    y = (n + height_m) * cos_lat * np.sin(lon)
    z = (n * (1 - WGS84_E2) + height_m) * sin_lat
    return x, y, z


def wgs84_to_enu(lon: pd.Series, lat: pd.Series, origin: dict[str, float | str]) -> tuple[np.ndarray, np.ndarray]:
    lon_vals = lon.astype(float).to_numpy()
    lat_vals = lat.astype(float).to_numpy()
    height = np.zeros(len(lon_vals))
    x, y, z = geodetic_to_ecef(lon_vals, lat_vals, height)
    lon0 = float(origin["lon"])
    lat0 = float(origin["lat"])
    h0 = float(origin.get("height", 0.0))
    x0, y0, z0 = geodetic_to_ecef(np.array([lon0]), np.array([lat0]), np.array([h0]))
    dx = x - x0[0]
    dy = y - y0[0]
    dz = z - z0[0]
    lon0_rad = math.radians(lon0)
    lat0_rad = math.radians(lat0)
    east = -math.sin(lon0_rad) * dx + math.cos(lon0_rad) * dy
    north = (
        -math.sin(lat0_rad) * math.cos(lon0_rad) * dx
        - math.sin(lat0_rad) * math.sin(lon0_rad) * dy
        + math.cos(lat0_rad) * dz
    )
    return east, north


def compute_diff_yawrate(df: pd.DataFrame, yaw_col: str) -> np.ndarray:
    yaw = df[yaw_col].to_numpy(dtype=float)
    ts = df["timestamp_ms"].to_numpy(dtype=float)
    diff = np.zeros(len(df))
    for i in range(1, len(df)):
        dt = (ts[i] - ts[i - 1]) / 1000.0
        if dt > 0:
            diff[i] = diff_angle(float(yaw[i - 1]), float(yaw[i])) / dt
    return diff


def gear_ratio_for_series(gear: pd.Series) -> pd.Series:
    rounded = gear.fillna(0).round().astype(int)
    return rounded.map(GEAR_RATIO).fillna(0.0).astype(float)


def append_supplement_columns(train: pd.DataFrame, interpolated: pd.DataFrame) -> pd.DataFrame:
    result = train.copy()
    if len(result) != len(interpolated):
        return result
    for col in SUPPLEMENT_COLUMNS:
        if col in interpolated.columns and col not in result.columns:
            result[col] = interpolated[col].to_numpy()
    return result


def build_interpolated(
    raw: pd.DataFrame,
    sample_period_ms: float = 20.0,
    controller_enabled_values: set[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sampled = nearest_resample(raw, sample_period_ms)
    report: dict[str, Any] = {"placeholder_columns": [], "source_columns": {}}
    enabled_values = controller_enabled_values or {3.0}
    out = pd.DataFrame()
    out["timestamp_ms"] = sampled["timestamp_ms"].astype(float)
    out["timestamp"] = (out["timestamp_ms"] - out["timestamp_ms"].iloc[0]) / 1000.0

    controller_col = first_existing(sampled, ["controller_enable", "controller_enable_source"])
    if controller_col:
        src = sampled[controller_col].astype(float)
        if src.max(skipna=True) <= 1.0:
            out["controller_enable"] = (src == 1.0).astype(int)
        else:
            out["controller_enable"] = src.isin(enabled_values).astype(int)
        report["source_columns"]["controller_enable"] = controller_col
    else:
        out["controller_enable"] = 0
        report["placeholder_columns"].append("controller_enable")

    steering_col = first_existing(sampled, ["adu_target_steer_deg", "steering_target", "vdc2_steer_angle_rad"])
    if steering_col == "vdc2_steer_angle_rad":
        out["steering_target"] = np.rad2deg(sampled[steering_col].astype(float))
    elif steering_col:
        out["steering_target"] = sampled[steering_col].astype(float)
    else:
        out["steering_target"] = 0.0
        report["placeholder_columns"].append("steering_target")
    if steering_col:
        report["source_columns"]["steering_target"] = steering_col

    torque_col = first_existing(sampled, [
        "total_motor_request_torque_nm",
        "tcu_motor_request_torque_nm",
        "vcu_motor_request_torque_nm",
        "motor_output_torque_nm",
    ])
    if torque_col:
        motor_torque = sampled[torque_col].astype(float)
        gear_col = first_existing(sampled, ["current_gear", "TCU_CurrentGear", "gear"])
        if gear_col:
            ratio = gear_ratio_for_series(sampled[gear_col].astype(float))
            report["source_columns"]["gear_ratio"] = gear_col
        else:
            ratio = pd.Series(np.zeros(len(sampled)))
            report["placeholder_columns"].append("current_gear")
        torque_wheel = motor_torque * ratio
        report["source_columns"]["target_torque"] = torque_col
        report["source_columns"][PATH_REMAIN_COL] = f"{torque_col} * gear_ratio"
    else:
        motor_torque = pd.Series(np.zeros(len(sampled)))
        torque_wheel = pd.Series(np.zeros(len(sampled)))
        report["placeholder_columns"].append(PATH_REMAIN_COL)
    out["target_torque"] = motor_torque
    out[PATH_REMAIN_COL] = torque_wheel
    out["torque_wheel"] = torque_wheel
    out["torque_fl"] = 0.0
    out["torque_fr"] = 0.0
    out["torque_rl"] = torque_wheel / 2.0
    out["torque_rr"] = torque_wheel / 2.0
    out["torque_filtered"] = np.nan

    speed_col = first_existing(sampled, ["vehicle_speed_kmh", VEHICLE_SPEED_COL])
    if speed_col:
        out[VEHICLE_SPEED_COL] = sampled[speed_col].astype(float)
        report["source_columns"][VEHICLE_SPEED_COL] = speed_col
    else:
        out[VEHICLE_SPEED_COL] = 0.0
        report["placeholder_columns"].append(VEHICLE_SPEED_COL)

    yawrate_col = first_existing(sampled, ["yaw_rate_radps", YAW_RATE_COL])
    if yawrate_col:
        out[YAW_RATE_COL] = sampled[yawrate_col].astype(float)
        report["source_columns"][YAW_RATE_COL] = yawrate_col
    else:
        out[YAW_RATE_COL] = 0.0
        report["placeholder_columns"].append(YAW_RATE_COL)

    x_col = first_existing_nonempty(sampled, ["position_enu.x", "x", "x_m", "X_m"])
    y_col = first_existing_nonempty(sampled, ["position_enu.y", "y", "y_m", "Y_m"])
    lon_col = first_existing_nonempty(sampled, ["ins_longitude_deg", "longitude_deg", "INS570D_INS_Longitude", "INS1_Longitude"])
    lat_col = first_existing_nonempty(sampled, ["ins_latitude_deg", "latitude_deg", "INS570D_INS_Latitude", "INS1_Latitude"])
    yaw_col = first_existing_nonempty(sampled, ["euler_angles.z", "yaw", "yaw_deg", "Yaw_deg", "ins_heading_deg", "INS570D_INS_HeadingAngle", "INS1_HeadingAngle"])
    if x_col and y_col:
        out["position_enu.x"] = sampled[x_col].astype(float)
        out["position_enu.y"] = sampled[y_col].astype(float)
    elif lon_col and lat_col:
        origin = choose_origin(sampled[lon_col].astype(float), sampled[lat_col].astype(float))
        east, north = wgs84_to_enu(sampled[lon_col], sampled[lat_col], origin)
        out["position_enu.x"] = east
        out["position_enu.y"] = north
        report["origin"] = origin
        report["source_columns"]["position_enu.x"] = f"{lon_col},{lat_col}"
        report["source_columns"]["position_enu.y"] = f"{lon_col},{lat_col}"
    else:
        out["position_enu.x"] = 0.0
        out["position_enu.y"] = 0.0
    out["euler_angles.z"] = sampled[yaw_col].astype(float) if yaw_col else 0.0
    if x_col:
        report["source_columns"]["position_enu.x"] = x_col
    elif not (lon_col and lat_col):
        report["placeholder_columns"].append("position_enu.x")
    if y_col:
        report["source_columns"]["position_enu.y"] = y_col
    elif not (lon_col and lat_col):
        report["placeholder_columns"].append("position_enu.y")
    if yaw_col:
        report["source_columns"]["euler_angles.z"] = yaw_col
    else:
        report["placeholder_columns"].append("euler_angles.z")

    for col in SUPPLEMENT_COLUMNS:
        src_col = first_existing_nonempty(sampled, [col])
        if src_col:
            out[col] = sampled[src_col].astype(float)

    vx_enu, vy_enu, vy_veh = compute_vehicle_velocity(out, "position_enu.x", "position_enu.y", "euler_angles.z")
    out["vx_enu"] = vx_enu
    out["vy_enu"] = vy_enu
    out["vx_veh"] = vx_enu * np.cos(np.deg2rad(out["euler_angles.z"])) + vy_enu * np.sin(np.deg2rad(out["euler_angles.z"]))
    vby_col = first_existing_nonempty(sampled, ["ins_vby_mps", "INS570D_INS_VBy"])
    if vby_col:
        out["vy"] = sampled[vby_col].astype(float)
        report["source_columns"]["vy"] = vby_col
    else:
        out["vy"] = vy_veh
    out["diff_yawrate"] = compute_diff_yawrate(out, "euler_angles.z")

    return out, report


def write_report(report: dict[str, Any], output: Path, interpolated: pd.DataFrame) -> None:
    data = {
        **report,
        "rows": int(len(interpolated)),
        "duration_s": float(interpolated["timestamp"].iloc[-1]) if len(interpolated) else 0.0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.raw)
    controller_enabled_values = {
        float(item.strip()) for item in args.controller_enabled_values.split(",") if item.strip()
    }
    interpolated, report = build_interpolated(
        raw,
        sample_period_ms=args.sample_period_ms,
        controller_enabled_values=controller_enabled_values,
    )
    args.interpolated_output.parent.mkdir(parents=True, exist_ok=True)
    interpolated.to_csv(args.interpolated_output, index=False)

    if args.report_output:
        write_report(report, args.report_output, interpolated)

    train_output = args.train_output or args.interpolated_output.with_name(
        f"{args.interpolated_output.stem}_train.csv"
    )
    import csvdata_new_truck
    csvdata_new_truck.convert_csv(
        args.interpolated_output,
        train_output,
        segment_static_data=args.segment,
    )
    if not args.segment:
        train_df = pd.read_csv(train_output)
        append_supplement_columns(train_df, interpolated).to_csv(train_output, index=False)
    print(f"interpolated csv: {args.interpolated_output}")
    print(f"train csv: {train_output}")
    if report["placeholder_columns"]:
        print("placeholder columns:", ", ".join(report["placeholder_columns"]))


if __name__ == "__main__":
    main()
