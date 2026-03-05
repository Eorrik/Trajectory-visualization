from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from urllib.parse import urlencode

from flask import Flask, render_template, request, Response, redirect

from src.data_processing import (
    filter_hand_data_by_pct,
    load_right_hand_data,
    prepare_data,
    smooth_right_hand_trajectory,
    smooth_trajectory,
)
from src.visualization import build_hand_plotly_figure, build_plotly_figure

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
META_PATH = ROOT / "sample" / "meta.json"
PEN_PATH = ROOT / "sample" / "pen_data.jsonl"
IMU_PATH = ROOT / "sample" / "20260226151623.txt"
HAND_PREDS_PATH = ROOT / "sample" / "pred3d_world.jsonl"

prepared = prepare_data(META_PATH, PEN_PATH, IMU_PATH)
hand_prepared = load_right_hand_data(HAND_PREDS_PATH)


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


def _filename_token(value: float) -> str:
    s = f"{float(value):.1f}"
    return s.replace(".", "p").replace("-", "m")


def _jsonable(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value:
            return None
        return value
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return str(value)


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

    figure = build_plotly_figure(filtered, forcing_ts, pen_length=pen_length, contact_threshold=contact_threshold)
    hand_filtered = filter_hand_data_by_pct(hand_prepared, start_pct, end_pct)
    if smooth_enabled:
        hand_filtered = smooth_right_hand_trajectory(hand_filtered, smooth_window)
    hand_figure = build_hand_plotly_figure(hand_filtered)

    wt1_cols = [c for c in filtered.columns if c.startswith("wt1_")]
    sample_preview = filtered[["display_time", *wt1_cols[:4]]].head(12).to_dict(orient="records")

    saved_file = q.get("saved_file")
    saved_rows = q.get("saved_rows")

    return render_template(
        "index.html",
        plotly_figure=figure,
        hand_plotly_figure=hand_figure,
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
        saved_file=saved_file,
        saved_rows=saved_rows,
    )


@app.route("/save", methods=["POST"])
def save_clip() -> Response:
    q = request.form

    start_pct = clamp(parse_float(q.get("start_pct"), 0.0), 0.0, 100.0)
    end_pct = clamp(parse_float(q.get("end_pct"), 100.0), 0.0, 100.0)
    if start_pct > end_pct:
        start_pct, end_pct = end_pct, start_pct

    forcing_pct = clamp(parse_float(q.get("forcing_pct"), start_pct), start_pct, end_pct)
    pen_length = parse_float(q.get("pen_length"), 0.18)
    contact_threshold = parse_float(q.get("contact_threshold"), 0.01)
    smooth_enabled = q.get("smooth") == "on"
    smooth_window = max(1, parse_int(q.get("smooth_window"), 5))

    time_start = map_pct_to_ts(start_pct, prepared.min_ts, prepared.max_ts)
    time_end = map_pct_to_ts(end_pct, prepared.min_ts, prepared.max_ts)

    df = prepared.df
    clip = df[(df["timestamp_unix"] >= time_start) & (df["timestamp_unix"] <= time_end)].copy()
    if smooth_enabled:
        clip = smooth_trajectory(clip, smooth_window)

    exports_dir = ROOT / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"clip_{_filename_token(start_pct)}_{_filename_token(end_pct)}_{stamp}.jsonl"
    out_path = exports_dir / filename

    records = clip.to_dict(orient="records")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            clean = {str(k): _jsonable(v) for k, v in rec.items()}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    params = {
        "start_pct": start_pct,
        "end_pct": end_pct,
        "forcing_pct": forcing_pct,
        "pen_length": pen_length,
        "contact_threshold": contact_threshold,
        "smooth_window": smooth_window,
        "saved_file": filename,
        "saved_rows": len(records),
    }
    if smooth_enabled:
        params["smooth"] = "on"

    return redirect(f"/?{urlencode(params)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
