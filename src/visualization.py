from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line3D


def _hex_color(base_rgb: tuple[int, int, int], intensity: float) -> str:
    intensity = float(np.clip(intensity, 0.2, 1.0))
    r, g, b = base_rgb
    rr = int(255 - (255 - r) * intensity)
    gg = int(255 - (255 - g) * intensity)
    bb = int(255 - (255 - b) * intensity)
    return f"#{rr:02x}{gg:02x}{bb:02x}"


def _to_points(series: pd.Series) -> list[list[float]]:
    return [[float(v[0]), float(v[1]), float(v[2])] for v in series]


def _styled_line_data(points: list[list[float]], ts: np.ndarray, forcing: float | None, base: tuple[int, int, int]) -> list[dict[str, Any]]:
    if forcing is None:
        weights = np.ones_like(ts, dtype=float) * 0.65
    else:
        scale = max((ts.max() - ts.min()) / 8.0, 1e-6)
        weights = np.exp(-np.abs(ts - forcing) / scale)
        weights = 0.25 + 0.75 * weights

    return [{"value": p, "itemStyle": {"color": _hex_color(base, float(w))}} for p, w in zip(points, weights)]

def _xy_equal_range(points_a: list[list[float]], points_b: list[list[float]], pad_ratio: float = 0.05) -> tuple[float, float, float, float]:
    """计算 x/y 统一跨度的坐标范围，使 3D 图中的 x、y 轴等距。"""
    points = np.array(points_a + points_b, dtype=float)
    x_vals = points[:, 0]
    y_vals = points[:, 1]

    x_center = float((x_vals.min() + x_vals.max()) / 2.0)
    y_center = float((y_vals.min() + y_vals.max()) / 2.0)

    x_span = float(x_vals.max() - x_vals.min())
    y_span = float(y_vals.max() - y_vals.min())
    span = max(x_span, y_span, 1e-6)
    half = (span * (1.0 + pad_ratio)) / 2.0

    return x_center - half, x_center + half, y_center - half, y_center + half


def build_chart(df: pd.DataFrame, forcing: float | None, pen_length: float, contact_threshold: float) -> Any:
    tip_points = _to_points(df["tip_pos_world"])
    tail_points = _to_points(df["tail_pos_world"])
    ts = df["timestamp_unix"].to_numpy()
    x_min, x_max, y_min, y_max = _xy_equal_range(tip_points, tail_points)
    chart = Line3D(init_opts=opts.InitOpts(width="1200px", height="760px"))
    chart.add(
        series_name="Tip 轨迹",
        data=_styled_line_data(tip_points, ts, forcing, (220, 20, 60)),
        xaxis3d_opts={"type": "value", "min": x_min, "max": x_max,"splitNumber": 8},
        yaxis3d_opts={"type": "value", "min": y_min, "max": y_max,"splitNumber": 8},
        zaxis3d_opts=opts.Axis3DOpts(type_="value"),
        grid3d_opts=opts.Grid3DOpts(width=140, height=100, depth=140, rotate_speed=22),
    )
    chart.add(
        series_name="Tail 轨迹",
        data=_styled_line_data(tail_points, ts, forcing, (30, 90, 220)),
    )

    pred_path = []
    connect_lines = []
    contact_points = []
    for idx, row in df.iterrows():
        tip = np.array(row["tip_pos_world"], dtype=float)
        tail = np.array(row["tail_pos_world"], dtype=float)
        vec = tip - tail
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            continue
        unit = vec / norm
        pred = tail + unit * pen_length

        pred_path.extend([tail.tolist(), pred.tolist()])
        if idx % 25 == 0:
            connect_lines.extend([tip.tolist(), tail.tolist()])

        if abs(pred[2]) <= contact_threshold:
            contact_points.extend([[pred[0], pred[1], 0.0], [pred[0] + 1e-5, pred[1], 0.0]])

    if connect_lines:
        chart.add(
            series_name="Tip-Tail 姿态连线(采样)",
            data=connect_lines,
            itemstyle_opts=opts.ItemStyleOpts(color="#8a8a8a", opacity=0.65),
        )

    if pred_path:
        chart.add(
            series_name="尾部延伸预测笔尖",
            data=pred_path,
            itemstyle_opts=opts.ItemStyleOpts(color="#00aa66", opacity=0.75),
        )

    if contact_points:
        chart.add(
            series_name="桌面接触点",
            data=contact_points,
            itemstyle_opts=opts.ItemStyleOpts(color="#ff0000", opacity=0.95),
        )

    if forcing is not None and len(df) > 0:
        nearest_idx = int(np.argmin(np.abs(ts - forcing)))
        p = tip_points[nearest_idx]
        chart.add(
            series_name="time_forcing",
            data=[p, [p[0] + 1e-4, p[1], p[2]]],
            itemstyle_opts=opts.ItemStyleOpts(color="#111111", opacity=1.0),
        )

    chart.set_global_opts(
        title_opts=opts.TitleOpts(title="毛笔 3D 轨迹交互分析"),
        legend_opts=opts.LegendOpts(pos_top="5%"),
        tooltip_opts=opts.TooltipOpts(is_show=True),
    )
    return chart
