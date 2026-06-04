#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit DBC/BLF coverage for the truck training-data preprocessing pipeline."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MAPPING = SCRIPT_DIR / "config" / "blf_signal_mapping.yaml"


def _import_cantools():
    try:
        import cantools  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "缺少依赖 cantools，无法读取 DBC。请先征得作者同意后安装，例如：pip install cantools"
        ) from exc
    return cantools


def _import_can():
    try:
        import can  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "缺少依赖 python-can，无法读取 BLF。请先征得作者同意后安装，例如：pip install python-can"
        ) from exc
    return can


@dataclass
class BlfStat:
    channel: str
    can_id: int
    is_extended_id: bool
    frame_count: int = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None

    def update(self, timestamp: float) -> None:
        self.frame_count += 1
        if self.first_timestamp is None or timestamp < self.first_timestamp:
            self.first_timestamp = timestamp
        if self.last_timestamp is None or timestamp > self.last_timestamp:
            self.last_timestamp = timestamp

    @property
    def duration_s(self) -> float:
        if self.first_timestamp is None or self.last_timestamp is None:
            return 0.0
        return max(0.0, self.last_timestamp - self.first_timestamp)

    @property
    def estimated_hz(self) -> float:
        duration = self.duration_s
        if duration <= 0.0 or self.frame_count <= 1:
            return 0.0
        return (self.frame_count - 1) / duration


@dataclass
class DbcSpec:
    path: Path
    db: Any
    channels: set[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 DBC/BLF 对训练目标信号的覆盖情况")
    parser.add_argument("--dbc", required=True, type=Path, nargs="+", help="DBC 文件或目录，可传多个")
    parser.add_argument("--channel-dbc-map", type=Path, help="channel 到 DBC 文件列表的 YAML 映射")
    parser.add_argument("--blf", type=Path, help="BLF 文件或目录，可选")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING, help="BLF 信号映射 YAML")
    parser.add_argument("--output-dir", type=Path, default=Path("can_audit"), help="审计输出目录")
    parser.add_argument("--max-sample-rows", type=int, default=200, help="decoded_sample.csv 最大行数")
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"target_columns": []}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"target_columns": []}


def iter_blf_files(path: Path | None) -> list[Path]:
    if path is None:
        return []
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*.blf") if p.is_file())
    raise FileNotFoundError(f"BLF 路径不存在：{path}")


def expand_dbc_paths(paths: list[Path]) -> list[Path]:
    dbcs: list[Path] = []
    for path in paths:
        if path.is_file():
            dbcs.append(path)
        elif path.is_dir():
            dbcs.extend(sorted(p for p in path.rglob("*.dbc") if p.is_file()))
        else:
            raise FileNotFoundError(f"DBC 路径不存在：{path}")
    return sorted(dict.fromkeys(dbcs))


def load_single_dbc(dbc_path: Path):
    cantools = _import_cantools()
    return cantools.database.load_file(str(dbc_path), strict=False)


def load_dbc_specs(dbc_args: list[Path], channel_map_path: Path | None) -> list[DbcSpec]:
    if channel_map_path is None:
        return [DbcSpec(path=p, db=load_single_dbc(p)) for p in expand_dbc_paths(dbc_args)]

    with channel_map_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = channel_map_path.parent
    specs: list[DbcSpec] = []
    channels = raw.get("channels", {})
    for channel, paths in channels.items():
        if not isinstance(paths, list):
            paths = [paths]
        resolved = []
        for item in paths:
            p = Path(item)
            if not p.is_absolute():
                p = base / p
            resolved.append(p)
        for p in expand_dbc_paths(resolved):
            specs.append(DbcSpec(path=p, db=load_single_dbc(p), channels={str(channel)}))
    return specs


def dbc_applies_to_channel(spec: DbcSpec, channel: str) -> bool:
    return spec.channels is None or channel in spec.channels


def find_messages_for_frame(specs: list[DbcSpec], channel: str, frame_id: int) -> list[tuple[DbcSpec, Any]]:
    matches = []
    for spec in specs:
        if not dbc_applies_to_channel(spec, channel):
            continue
        try:
            msg = spec.db.get_message_by_frame_id(frame_id)
        except Exception:
            msg = None
        if msg is not None:
            matches.append((spec, msg))
    return matches


def write_dbc_inventory(specs: list[DbcSpec], output_path: Path) -> None:
    fieldnames = [
        "dbc_path", "channels", "message_id", "message_id_hex", "message_name", "is_extended_frame", "dlc", "cycle_time",
        "signal_name", "start_bit", "length", "byte_order", "is_signed", "scale", "offset",
        "unit", "minimum", "maximum",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for spec in specs:
            channels = "*" if spec.channels is None else ",".join(sorted(spec.channels))
            for msg in spec.db.messages:
                for sig in msg.signals:
                    writer.writerow({
                        "dbc_path": str(spec.path),
                        "channels": channels,
                        "message_id": msg.frame_id,
                        "message_id_hex": f"0x{msg.frame_id:X}",
                        "message_name": msg.name,
                        "is_extended_frame": getattr(msg, "is_extended_frame", False),
                        "dlc": msg.length,
                        "cycle_time": getattr(msg, "cycle_time", None),
                        "signal_name": sig.name,
                        "start_bit": sig.start,
                        "length": sig.length,
                        "byte_order": sig.byte_order,
                        "is_signed": sig.is_signed,
                        "scale": sig.scale,
                        "offset": sig.offset,
                        "unit": sig.unit,
                        "minimum": sig.minimum,
                        "maximum": sig.maximum,
                    })


def audit_blf_files(
    specs: list[DbcSpec],
    blf_files: list[Path],
    max_sample_rows: int,
) -> tuple[dict[tuple[str, int, bool], BlfStat], list[dict[str, Any]], set[str], set[str]]:
    can = _import_can()
    stats: dict[tuple[str, int, bool], BlfStat] = {}
    samples: list[dict[str, Any]] = []
    decode_ok_messages: set[str] = set()
    decode_error_messages: set[str] = set()

    for blf_path in blf_files:
        with can.BLFReader(str(blf_path)) as reader:
            for msg in reader:
                channel = str(getattr(msg, "channel", ""))
                can_id = int(msg.arbitration_id)
                is_extended = bool(getattr(msg, "is_extended_id", False))
                key = (channel, can_id, is_extended)
                stats.setdefault(key, BlfStat(channel, can_id, is_extended)).update(float(msg.timestamp))

                matches = find_messages_for_frame(specs, channel, can_id)
                if not matches:
                    continue

                for _, dbc_msg in matches:
                    try:
                        decoded = dbc_msg.decode(bytes(msg.data), decode_choices=False)
                    except Exception:
                        decode_error_messages.add(dbc_msg.name)
                        continue

                    decode_ok_messages.add(dbc_msg.name)
                    if len(samples) >= max_sample_rows:
                        continue
                    for signal_name, value in decoded.items():
                        if len(samples) >= max_sample_rows:
                            break
                        samples.append({
                            "blf_file": str(blf_path),
                            "timestamp": float(msg.timestamp),
                            "channel": channel,
                            "can_id": f"0x{can_id:X}",
                            "message_name": dbc_msg.name,
                            "signal_name": signal_name,
                            "value": value,
                        })

    return stats, samples, decode_ok_messages, decode_error_messages


def write_blf_coverage(
    specs: list[DbcSpec],
    stats: dict[tuple[str, int, bool], BlfStat],
    output_path: Path,
    decode_ok_messages: set[str],
    decode_error_messages: set[str],
) -> set[str]:
    seen_messages: set[str] = set()
    fieldnames = [
        "channel", "can_id", "can_id_hex", "is_extended_id", "dbc_message_name",
        "frame_count", "first_timestamp", "last_timestamp", "duration_s",
        "estimated_hz", "decode_status",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stat in sorted(stats.values(), key=lambda s: (s.channel, s.can_id, s.is_extended_id)):
            matches = find_messages_for_frame(specs, stat.channel, stat.can_id)
            decode_status = "not_in_dbc"
            message_name = ""
            if matches:
                names = []
                for _, dbc_msg in matches:
                    names.append(dbc_msg.name)
                    seen_messages.add(dbc_msg.name)
                message_name = ";".join(sorted(set(names)))
                if any(name in decode_ok_messages for name in names):
                    decode_status = "decoded"
                elif any(name in decode_error_messages for name in names):
                    decode_status = "decode_failed"
                else:
                    decode_status = "message_in_dbc"
            writer.writerow({
                "channel": stat.channel,
                "can_id": stat.can_id,
                "can_id_hex": f"0x{stat.can_id:X}",
                "is_extended_id": stat.is_extended_id,
                "dbc_message_name": message_name,
                "frame_count": stat.frame_count,
                "first_timestamp": stat.first_timestamp,
                "last_timestamp": stat.last_timestamp,
                "duration_s": stat.duration_s,
                "estimated_hz": stat.estimated_hz,
                "decode_status": decode_status,
            })
    return seen_messages


def write_decoded_sample(samples: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = ["blf_file", "timestamp", "channel", "can_id", "message_name", "signal_name", "value"]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def build_dbc_indexes(specs: list[DbcSpec]) -> tuple[dict[str, Any], dict[tuple[str, str], bool]]:
    messages_by_name = {}
    signal_pairs: dict[tuple[str, str], bool] = {}
    for spec in specs:
        for msg in spec.db.messages:
            messages_by_name[msg.name] = msg
            for sig in msg.signals:
                signal_pairs[(msg.name, sig.name)] = True
    return messages_by_name, signal_pairs


def target_rows(
    mapping: dict[str, Any],
    specs: list[DbcSpec],
    seen_messages: set[str] | None,
    decode_ok_messages: set[str] | None,
) -> list[dict[str, Any]]:
    messages_by_name, signal_pairs = build_dbc_indexes(specs)
    rows: list[dict[str, Any]] = []
    for item in mapping.get("target_columns", []):
        target = item.get("target_column", "")
        source_kind = item.get("source_kind", "")
        msg_name = item.get("message_name") or ""
        sig_name = item.get("signal_name") or ""
        message_in_dbc = bool(msg_name and msg_name in messages_by_name)
        signal_decodable = bool(msg_name and sig_name and signal_pairs.get((msg_name, sig_name), False))
        message_seen = bool(msg_name and seen_messages is not None and msg_name in seen_messages)

        if source_kind == "dbc_signal":
            if not msg_name or not sig_name:
                status = "unmapped"
            elif not message_in_dbc:
                status = "message_not_in_dbc"
            elif seen_messages is not None and not message_seen:
                status = "message_not_seen_in_blf"
            elif not signal_decodable:
                status = "signal_not_in_message"
            elif decode_ok_messages is not None and message_seen and msg_name not in decode_ok_messages:
                status = "message_decode_failed_in_blf"
            else:
                status = "covered"
        elif source_kind == "derived":
            status = "derived_rule_declared"
        elif source_kind == "constant":
            status = "constant_fill"
        elif source_kind == "missing_allowed":
            status = "missing_allowed"
        else:
            status = "unknown_source_kind"

        rows.append({
            "target_column": target,
            "required_level": item.get("required_level", ""),
            "source_kind": source_kind,
            "dbc_message": msg_name,
            "dbc_signal": sig_name,
            "message_in_dbc": message_in_dbc,
            "message_seen_in_blf": message_seen if seen_messages is not None else "",
            "signal_decodable": signal_decodable,
            "unit_transform": item.get("unit_transform", ""),
            "derive_rule": item.get("derive_rule", ""),
            "coverage_status": status,
            "notes": item.get("notes", ""),
        })
    return rows


def write_target_coverage(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "target_column", "required_level", "source_kind", "dbc_message", "dbc_signal",
        "message_in_dbc", "message_seen_in_blf", "signal_decodable", "unit_transform",
        "derive_rule", "coverage_status", "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_missing_report(rows: list[dict[str, Any]], output_path: Path, blf_files: list[Path]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        status = row["coverage_status"]
        if status in {"covered", "derived_rule_declared", "constant_fill", "missing_allowed"}:
            continue
        level = row["required_level"]
        if level == "blocking":
            group = "Blocking"
        elif level == "required_for_scene_split":
            group = "Degraded"
        else:
            group = "Optional"
        groups[group].append(row)

    lines = [
        "# CAN 信号缺失审计报告",
        "",
        f"- BLF 输入：{len(blf_files)} 个文件" if blf_files else "- BLF 输入：未提供，仅审计 DBC 和映射配置",
        "",
    ]
    for group in ["Blocking", "Degraded", "Optional"]:
        lines.append(f"## {group}")
        rows_in_group = groups.get(group, [])
        if not rows_in_group:
            lines.append("")
            lines.append("无。")
            lines.append("")
            continue
        lines.append("")
        for row in rows_in_group:
            lines.append(
                f"- `{row['target_column']}`: {row['coverage_status']} "
                f"(source={row['source_kind']}, dbc={row['dbc_message']}.{row['dbc_signal']})"
            )
            if row.get("notes"):
                lines.append(f"  - notes: {row['notes']}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        specs = load_dbc_specs(args.dbc, args.channel_dbc_map)
        mapping = load_mapping(args.mapping)
        write_dbc_inventory(specs, args.output_dir / "dbc_signal_inventory.csv")

        blf_files = iter_blf_files(args.blf)
        seen_messages: set[str] | None = None
        decode_ok_messages: set[str] | None = None
        if blf_files:
            stats, samples, decode_ok_messages, decode_error_messages = audit_blf_files(specs, blf_files, args.max_sample_rows)
            seen_messages = write_blf_coverage(
                specs,
                stats,
                args.output_dir / "blf_message_coverage.csv",
                decode_ok_messages,
                decode_error_messages,
            )
            write_decoded_sample(samples, args.output_dir / "decoded_sample.csv")

        rows = target_rows(mapping, specs, seen_messages, decode_ok_messages)
        write_target_coverage(rows, args.output_dir / "target_signal_coverage.csv")
        write_missing_report(rows, args.output_dir / "missing_signal_report.md", blf_files)
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    print(f"审计完成，输出目录: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
