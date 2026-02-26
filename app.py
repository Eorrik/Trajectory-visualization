from __future__ import annotations

from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request

from src.data_processing import prepare_data, smooth_trajectory
from src.visualization import build_chart

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
META_PATH = ROOT / "sample" / "meta.json"
PEN_PATH = ROOT / "sample" / "pen_data.jsonl"
IMU_PATH = ROOT / "sample" / "20260224152819.txt"

prepared = prepare_data(META_PATH, PEN_PATH, IMU_PATH)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def map_pct_to_ts(pct: float, min_ts: float, max_ts: float) -> float:
    ratio = clamp(pct, 0.0, 100.0) / 100.0
    return min_ts + ratio * (max_ts - min_ts)


def format_dt_local(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


@app.route("/", methods=["GET"])
def index():
    q = request.args

    start_pct = clamp(parse_float(q.get("start_pct"), 0.0), 0.0, 100.0)
    end_pct = clamp(parse_float(q.get("end_pct"), 100.0), 0.0, 100.0)
    if start_pct > end_pct:
        start_pct, end_pct = end_pct, start_pct

    forcing_pct = clamp(parse_float(q.get("forcing_pct"), start_pct), start_pct, end_pct)

    time_start = map_pct_to_ts(start_pct, prepared.min_ts, prepared.max_ts)
    time_end = map_pct_to_ts(end_pct, prepared.min_ts, prepared.max_ts)
    forcing_ts = map_pct_to_ts(forcing_pct, prepared.min_ts, prepared.max_ts)

    pen_length = parse_float(q.get("pen_length"), 0.18)
    contact_threshold = parse_float(q.get("contact_threshold"), 0.01)
    smooth_enabled = q.get("smooth") == "on"
    smooth_window = max(1, parse_int(q.get("smooth_window"), 5))

    df = prepared.df
    filtered = df[(df["timestamp_unix"] >= time_start) & (df["timestamp_unix"] <= time_end)].copy()
    if filtered.empty:
        filtered = df.head(10).copy()

    if smooth_enabled:
        filtered = smooth_trajectory(filtered, smooth_window)

    chart = build_chart(filtered, forcing_ts, pen_length=pen_length, contact_threshold=contact_threshold)
    chart_html = chart.render_embed()

    wt1_cols = [c for c in filtered.columns if c.startswith("wt1_")]
    sample_preview = filtered[["display_time", *wt1_cols[:4]]].head(12).to_dict(orient="records")

    return render_template(
        "index.html",
        chart_html=chart_html,
        start_pct=start_pct,
        end_pct=end_pct,
        forcing_pct=forcing_pct,
        forcing_pct_min=start_pct,
        forcing_pct_max=end_pct,
        time_start=format_dt_local(time_start),
        time_end=format_dt_local(time_end),
        time_forcing=format_dt_local(forcing_ts),
        pen_length=pen_length,
        contact_threshold=contact_threshold,
        smooth_enabled=smooth_enabled,
        smooth_window=smooth_window,
        min_ts=format_dt_local(prepared.min_ts),
        max_ts=format_dt_local(prepared.max_ts),
        rows=len(filtered),
        sample_preview=sample_preview,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
