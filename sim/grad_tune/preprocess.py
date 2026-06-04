from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from grad_tune.data_schema import resolve_column


@dataclass
class PreprocessReport:
    mode: str
    selected_start_index: int
    selected_end_index: int
    selected_rows: int
    total_rows: int
    warnings: list[str]


def zero_small_values(values: np.ndarray, threshold: float) -> np.ndarray:
    result = values.copy()
    result[np.abs(result) < threshold] = 0.0
    return result


def find_continuous_segments(mask: np.ndarray,
                             min_length: int = 1) -> list[tuple[int, int]]:
    segments = []
    start = None
    for idx, ok in enumerate(mask):
        if ok and start is None:
            start = idx
        elif not ok and start is not None:
            if idx - start >= min_length:
                segments.append((start, idx - 1))
            start = None
    if start is not None and len(mask) - start >= min_length:
        segments.append((start, len(mask) - 1))
    return segments


def _as_float(df: pd.DataFrame, field: str) -> np.ndarray:
    col = resolve_column(df.columns, field)
    if col is None:
        raise ValueError(f"Missing required column for preprocessing: {field}")
    return pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)


def _optional_float(df: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray | None:
    for col in columns:
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
    return None


def _dilated(mask: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 0 or not mask.any():
        return mask
    out = mask.copy()
    for idx, ok in enumerate(mask):
        if ok:
            lo = max(0, idx - frames)
            hi = min(len(mask), idx + frames + 1)
            out[lo:hi] = True
    return out


def select_valid_segment(df: pd.DataFrame,
                         mode: str = 'combined',
                         min_motion_speed: float = 0.05,
                         zero_threshold: float = 1e-7,
                         min_duration_s: float = 2.0,
                         dt_min: float = 0.005,
                         dt_max: float = 0.039,
                         vy_jump_threshold: float = 2.0,
                         jump_dilate_frames: int = 3,
                         dt: float = 0.02) -> tuple[pd.DataFrame, PreprocessReport]:
    if mode == 'none':
        report = PreprocessReport(
            mode=mode, selected_start_index=0, selected_end_index=len(df) - 1,
            selected_rows=len(df), total_rows=len(df), warnings=[])
        return df.copy(), report
    if mode not in ('combined', 'controller'):
        raise ValueError(f"Unsupported preprocess mode: {mode}")

    warnings = []
    t = _as_float(df, 'timestamp')
    enabled = _as_float(df, 'controller_enable') == 1

    dt_raw = np.diff(t, prepend=t[0])
    dt_ok = np.ones(len(df), dtype=bool)
    dt_ok[1:] = (dt_raw[1:] >= dt_min) & (dt_raw[1:] <= dt_max)

    finite = np.isfinite(t) & np.isfinite(_as_float(df, 'x')) & np.isfinite(_as_float(df, 'y'))
    valid = enabled & dt_ok & finite

    if mode == 'combined':
        ego_v = zero_small_values(_as_float(df, 'ego_v'), zero_threshold)
        ref_v = zero_small_values(_as_float(df, 'ref_v'), zero_threshold)
        moving = np.maximum(np.abs(ego_v), np.abs(ref_v)) > min_motion_speed
        valid = valid & moving

        vy = _optional_float(df, ('Vy_mps', 'vy'))
        if vy is not None:
            vy = zero_small_values(vy, zero_threshold)
            jump = np.abs(vy) > vy_jump_threshold
            valid = valid & ~_dilated(jump, jump_dilate_frames)
        else:
            warnings.append("vy column missing, skipped vy jump filtering")

    min_rows = max(2, int(round(min_duration_s / dt)))
    segments = find_continuous_segments(valid, min_length=min_rows)
    if not segments:
        raise ValueError(
            "No valid grad_tune segment found after preprocessing. "
            "Try --preprocess-mode controller or lower --min-motion-speed.")

    start, end = max(segments, key=lambda item: item[1] - item[0])
    selected = df.iloc[start:end + 1].copy()
    report = PreprocessReport(
        mode=mode,
        selected_start_index=int(start),
        selected_end_index=int(end),
        selected_rows=int(end - start + 1),
        total_rows=int(len(df)),
        warnings=warnings,
    )
    return selected, report
