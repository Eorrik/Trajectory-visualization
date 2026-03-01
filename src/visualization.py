from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _hex_color(base_rgb: tuple[int, int, int], intensity: float) -> str:
    intensity = float(np.clip(intensity, 0.2, 1.0))
    r, g, b = base_rgb
    rr = int(255 - (255 - r) * intensity)
    gg = int(255 - (255 - g) * intensity)
    bb = int(255 - (255 - b) * intensity)
    return f"#{rr:02x}{gg:02x}{bb:02x}"


def _to_points(series: pd.Series) -> list[list[float]]:
    return [[float(v[0]), float(v[1]), float(v[2])] for v in series]


def _color_scale(ts: np.ndarray, forcing: float | None, base: tuple[int, int, int]) -> list[str]:
    if forcing is None:
        weights = np.ones_like(ts, dtype=float) * 0.65
    else:
        scale = max((ts.max() - ts.min()) / 8.0, 1e-6)
        weights = np.exp(-np.abs(ts - forcing) / scale)
        weights = 0.25 + 0.75 * weights
    return [_hex_color(base, float(w)) for w in weights]


def _axis_ranges(tip_points: list[list[float]], tail_points: list[list[float]], pred_points: list[list[float]], contact_points: list[list[float]]) -> dict[str, list[float]]:
    all_pts = np.array(tip_points + tail_points + pred_points + contact_points, dtype=float)

    x_vals = all_pts[:, 0]
    y_vals = all_pts[:, 1]
    z_vals = all_pts[:, 2]

    x_center = float((x_vals.min() + x_vals.max()) / 2.0)
    y_center = float((y_vals.min() + y_vals.max()) / 2.0)
    xy_span = max(float(x_vals.max() - x_vals.min()), float(y_vals.max() - y_vals.min()), 1e-6)
    xy_half = (xy_span * 1.18) / 2.0  # 扩大绘制范围，避免预测点越界

    z_min = float(z_vals.min())
    z_max = float(z_vals.max())
    z_span = max(z_max - z_min, 1e-6)
    z_pad = z_span * 0.15 + 0.01

    return {
        "x": [x_center - xy_half, x_center + xy_half],
        "y": [y_center - xy_half, y_center + xy_half],
        "z": [z_min - z_pad, z_max + z_pad],
    }


def build_plotly_figure(df: pd.DataFrame, forcing: float | None, pen_length: float, contact_threshold: float) -> dict[str, Any]:
    tip_points = _to_points(df["tip_pos_world"])
    tail_points = _to_points(df["tail_pos_world"])
    ts = df["timestamp_unix"].to_numpy()
    colors_tip = _color_scale(ts, forcing, (220, 20, 60))
    colors_tail = _color_scale(ts, forcing, (30, 90, 220))

    tip_xyz = np.array(tip_points, dtype=float)
    tail_xyz = np.array(tail_points, dtype=float)

    traces: list[dict[str, Any]] = [
        {
            "type": "scatter3d",
            "mode": "lines+markers",
            "name": "Tip 轨迹",
            "x": tip_xyz[:, 0].tolist(),
            "y": tip_xyz[:, 1].tolist(),
            "z": tip_xyz[:, 2].tolist(),
            "line": {"width": 4, "color": colors_tip},
            "marker": {"size": 2.4, "color": colors_tip},
        },
        {
            "type": "scatter3d",
            "mode": "lines+markers",
            "name": "Tail 轨迹",
            "x": tail_xyz[:, 0].tolist(),
            "y": tail_xyz[:, 1].tolist(),
            "z": tail_xyz[:, 2].tolist(),
            "line": {"width": 3, "color": colors_tail},
            "marker": {"size": 2.0, "color": colors_tail},
        },
    ]

    conn_x: list[float | None] = []
    conn_y: list[float | None] = []
    conn_z: list[float | None] = []
    pred_x: list[float | None] = []
    pred_y: list[float | None] = []
    pred_z: list[float | None] = []

    pred_points: list[list[float]] = []
    contact_points: list[list[float]] = []

    for idx, row in df.iterrows():
        tip = np.array(row["tip_pos_world"], dtype=float)
        tail = np.array(row["tail_pos_world"], dtype=float)
        vec = tip - tail
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            continue
        unit = vec / norm
        pred = tail + unit * pen_length

        pred_points.append(pred.tolist())
        pred_x += [tail[0], pred[0], None]
        pred_y += [tail[1], pred[1], None]
        pred_z += [tail[2], pred[2], None]

        if idx % 25 == 0:
            conn_x += [tip[0], tail[0], None]
            conn_y += [tip[1], tail[1], None]
            conn_z += [tip[2], tail[2], None]

        # 需求：只要低于地面（z <= 0）就认为接触
        if pred[2] <= 0:
            contact_points.append([float(pred[0]), float(pred[1]), 0.0])

    if conn_x:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "Tip-Tail 姿态连线(采样)",
                "x": conn_x,
                "y": conn_y,
                "z": conn_z,
                "line": {"width": 1, "color": "#8a8a8a"},
            }
        )

    if pred_x:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "尾部延伸预测笔尖",
                "x": pred_x,
                "y": pred_y,
                "z": pred_z,
                "line": {"width": 2, "color": "#00aa66"},
            }
        )

    if contact_points:
        c = np.array(contact_points, dtype=float)
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": "桌面接触点",
                "x": c[:, 0].tolist(),
                "y": c[:, 1].tolist(),
                "z": c[:, 2].tolist(),
                "marker": {"size": 4.5, "color": "#ff0000"},
            }
        )

    if forcing is not None and len(df) > 0:
        nearest_idx = int(np.argmin(np.abs(ts - forcing)))
        p = tip_points[nearest_idx]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": "time_forcing",
                "x": [p[0]],
                "y": [p[1]],
                "z": [p[2]],
                "marker": {"size": 8, "color": "#111111", "symbol": "diamond"},
            }
        )

    ranges = _axis_ranges(tip_points, tail_points, pred_points, contact_points)

    layout: dict[str, Any] = {
        "title": "毛笔 3D 轨迹交互分析（Plotly）",
        "height": 760,
        "scene": {
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 0.8},
            "xaxis": {"title": "X", "range": ranges["x"], "nticks": 9},
            "yaxis": {"title": "Y", "range": ranges["y"], "nticks": 9},
            "zaxis": {"title": "Z", "range": ranges["z"]},
        },
        "legend": {"orientation": "h", "y": 1.03},
        "margin": {"l": 0, "r": 0, "b": 0, "t": 60},
    }

    return {"data": traces, "layout": layout}


def build_hand_plotly_figure(hand_df: pd.DataFrame) -> dict[str, Any]:
    if hand_df.empty:
        return {
            "data": [],
            "layout": {
                "title": "手部 3D 轨迹（right）",
                "height": 760,
                "annotations": [
                    {
                        "text": "当前无可用 right 手数据",
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.5,
                        "y": 0.5,
                        "showarrow": False,
                        "font": {"size": 16},
                    }
                ],
            },
        }

    points_3d = np.array(hand_df["right_hand_points"].tolist(), dtype=float)
    n_frames, n_joints, _ = points_3d.shape
    time_weights = np.linspace(0.35, 1.0, n_frames)

    # 21 点位：按点位索引分配色相梯度，再按帧序提升明度
    # 自定义从蓝->青->绿->黄->红的渐变带
    anchor = np.array(
        [
            [40, 80, 220],
            [40, 200, 220],
            [60, 200, 120],
            [250, 180, 40],
            [230, 60, 60],
        ],
        dtype=float,
    )
    anchor_x = np.linspace(0.0, 1.0, len(anchor))
    joint_x = np.linspace(0.0, 1.0, n_joints)
    joint_rgb = np.stack([np.interp(joint_x, anchor_x, anchor[:, i]) for i in range(3)], axis=1)
    joint_base_colors = [tuple(int(v) for v in rgb) for rgb in joint_rgb]

    traces: list[dict[str, Any]] = []
    for joint_idx in range(n_joints):
        xyz = points_3d[:, joint_idx, :]
        colors = [_hex_color(joint_base_colors[joint_idx], float(w)) for w in time_weights]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines+markers",
                "name": f"右手点位 {joint_idx:02d}",
                "x": xyz[:, 0].tolist(),
                "y": xyz[:, 1].tolist(),
                "z": xyz[:, 2].tolist(),
                "line": {"width": 2.2, "color": colors},
                "marker": {"size": 2.0, "color": colors},
                "showlegend": joint_idx in {0, 5, 17, 20},
            }
        )

    wrist_xyz = points_3d[:, 0, :]
    hand_back_a = points_3d[:, 5, :]
    hand_back_b = points_3d[:, 17, :]
    traces.extend(
        [
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "手腕轨迹(index=0)",
                "x": wrist_xyz[:, 0].tolist(),
                "y": wrist_xyz[:, 1].tolist(),
                "z": wrist_xyz[:, 2].tolist(),
                "line": {"width": 6, "color": "#111111"},
            },
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "手背点轨迹 A(index=5)",
                "x": hand_back_a[:, 0].tolist(),
                "y": hand_back_a[:, 1].tolist(),
                "z": hand_back_a[:, 2].tolist(),
                "line": {"width": 5, "color": "#1f8a3a"},
            },
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "手背点轨迹 B(index=17)",
                "x": hand_back_b[:, 0].tolist(),
                "y": hand_back_b[:, 1].tolist(),
                "z": hand_back_b[:, 2].tolist(),
                "line": {"width": 5, "color": "#b3368f"},
            },
        ]
    )

    all_points = points_3d.reshape(-1, 3)
    x_vals, y_vals, z_vals = all_points[:, 0], all_points[:, 1], all_points[:, 2]
    x_span = max(float(x_vals.max() - x_vals.min()), 1e-6)
    y_span = max(float(y_vals.max() - y_vals.min()), 1e-6)
    z_span = max(float(z_vals.max() - z_vals.min()), 1e-6)
    max_span = max(x_span, y_span, z_span)
    pad = max_span * 0.12

    layout: dict[str, Any] = {
        "title": "手部 3D 轨迹（right，21点位顺序梯度 + 帧明度渐变）",
        "height": 760,
        "scene": {
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": max(0.55, z_span / max(x_span, y_span, 1e-6))},
            "xaxis": {"title": "X", "range": [float(x_vals.min() - pad), float(x_vals.max() + pad)]},
            "yaxis": {"title": "Y", "range": [float(y_vals.min() - pad), float(y_vals.max() + pad)]},
            "zaxis": {"title": "Z", "range": [float(z_vals.min() - pad), float(z_vals.max() + pad)]},
        },
        "legend": {"orientation": "h", "y": 1.03},
        "margin": {"l": 0, "r": 0, "b": 0, "t": 60},
    }

    return {"data": traces, "layout": layout}
