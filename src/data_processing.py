from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PreparedData:
    df: pd.DataFrame
    min_ts: float
    max_ts: float


def _safe_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list) and len(value) == 3:
        return [float(v) for v in value]
    return None


def load_meta(meta_path: str | Path) -> dict[str, Any]:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plane_basis(plane: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回平面法向量n、平面原点p0、以及平面内基向量u/v."""
    a, b, c, d = [float(v) for v in plane]
    n = np.array([a, b, c], dtype=float)
    norm = np.linalg.norm(n)
    if norm == 0:
        n = np.array([0.0, 1.0, 0.0], dtype=float)
        norm = 1.0
    n = n / norm
    p0 = -d * n
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u = u / (np.linalg.norm(u) + 1e-9)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-9)
    return n, p0, u, v


def camera_to_world_plane_xy(point_cam: list[float], plane: list[float]) -> list[float]:
    """将相机坐标投影到桌面平面并在平面基上表达为[x, y, h]世界坐标."""
    p = np.array(point_cam, dtype=float)
    n, p0, u, v = plane_basis(plane)
    signed_dist = float(np.dot(n, p - p0))
    proj = p - signed_dist * n
    rel = proj - p0
    x = float(np.dot(rel, u))
    y = float(np.dot(rel, v))
    h = signed_dist
    return [x, y, h]


def load_pen_data(pen_path: str | Path, plane: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with open(pen_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tip_cam = _safe_vector(rec.get("tip_pos_cam"))
            tail_cam = _safe_vector(rec.get("tail_pos_cam"))
            tip_world = _safe_vector(rec.get("tip_pos_world"))
            tail_world = _safe_vector(rec.get("tail_pos_world"))

            if tip_world is None and tip_cam is not None:
                tip_world = camera_to_world_plane_xy(tip_cam, plane)
            if tail_world is None and tail_cam is not None:
                tail_world = camera_to_world_plane_xy(tail_cam, plane)

            rows.append(
                {
                    "frame_id": rec.get("frame_id"),
                    "timestamp_unix": float(rec["timestamp"]),
                    "tip_pos_world": tip_world,
                    "tail_pos_world": tail_world,
                }
            )

    df = pd.DataFrame(rows)
    df = df[df["tip_pos_world"].notna() & df["tail_pos_world"].notna()].copy()
    return df.sort_values("timestamp_unix").reset_index(drop=True)


def load_imu_txt(txt_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(txt_path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["时间"], errors="coerce")
    df = df[df["timestamp"].notna()].copy()
    df["timestamp_unix"] = df["timestamp"].astype("int64") / 1e9
    df["device_type"] = df["设备名称"].str.extract(r"(WT\d)")

    for c in df.columns:
        if c in {"时间", "设备名称", "timestamp", "device_type"}:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("timestamp_unix").reset_index(drop=True)


def interpolate_wt1(pen_df: pd.DataFrame, imu_df: pd.DataFrame) -> pd.DataFrame:
    wt1 = imu_df[imu_df["device_type"] == "WT1"].copy()
    if wt1.empty:
        return pen_df

    numeric_cols = [
        c
        for c in wt1.columns
        if c
        not in {
            "时间",
            "设备名称",
            "timestamp",
            "timestamp_unix",
            "device_type",
            "版本号()",
        }
        and pd.api.types.is_numeric_dtype(wt1[c])
    ]

    x = wt1["timestamp_unix"].to_numpy()
    valid = np.isfinite(x)
    x = x[valid]
    if len(x) < 2:
        return pen_df

    target = pen_df["timestamp_unix"].to_numpy()
    for col in numeric_cols:
        y = wt1[col].to_numpy()[valid]
        finite = np.isfinite(y)
        if finite.sum() < 2:
            continue
        pen_df[f"wt1_{col}"] = np.interp(target, x[finite], y[finite], left=np.nan, right=np.nan)
    return pen_df


def prepare_data(meta_path: str | Path, pen_path: str | Path, imu_path: str | Path) -> PreparedData:
    meta = load_meta(meta_path)
    plane = meta.get("desk_plane", [0, 1, 0, 0])
    pen_df = load_pen_data(pen_path, plane)
    imu_df = load_imu_txt(imu_path)
    out = interpolate_wt1(pen_df, imu_df)
    out["display_time"] = pd.to_datetime(out["timestamp_unix"], unit="s").dt.strftime("%y-%m-%d-%H:%M:%S")
    return PreparedData(df=out, min_ts=float(out["timestamp_unix"].min()), max_ts=float(out["timestamp_unix"].max()))



def smooth_trajectory(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """对 tip/tail 轨迹做滑动平均平滑，降低采集噪声。"""
    out = df.copy()
    window = max(1, int(window))
    if window == 1:
        return out

    for key in ("tip_pos_world", "tail_pos_world"):
        pts = np.array(out[key].tolist(), dtype=float)
        if len(pts) == 0:
            continue
        smoothed = pd.DataFrame(pts).rolling(window=window, min_periods=1, center=True).mean().to_numpy()
        out[key] = smoothed.tolist()
    return out
