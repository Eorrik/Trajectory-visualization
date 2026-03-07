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
from src.visualization import build_imu_plotly_figures

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample"
META_PATH = SAMPLE_DIR / "meta.json"
PEN_PATH = SAMPLE_DIR / "pen_data.jsonl"
HAND_PREDS_PATH = SAMPLE_DIR / "pred3d_world.jsonl"


def resolve_imu_path(sample_dir: Path) -> Path:
    txt_files = sorted(sample_dir.glob("*.txt"))
    if len(txt_files) != 1:
        names = ", ".join(p.name for p in txt_files)
        raise RuntimeError(f"sample 目录下必须且只能有 1 个 .txt 作为 IMU 文件，当前数量={len(txt_files)}：{names}")
    return txt_files[0]


IMU_PATH = resolve_imu_path(SAMPLE_DIR)
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

    pen_length = parse_float(q.get("pen_length"), 0.21)
    contact_threshold = parse_float(q.get("contact_threshold"), 0.01)
    smooth_enabled = q.get("smooth") == "on"
    smooth_window = max(1, parse_int(q.get("smooth_window"), 5))

    df = prepared.pen_df
    imu_df = prepared.imu_df
    filtered = df[(df["timestamp_unix"] >= time_start) & (df["timestamp_unix"] <= time_end)].copy()
    imu_filtered = imu_df[(imu_df["timestamp_unix_aligned"] >= time_start) & (imu_df["timestamp_unix_aligned"] <= time_end)].copy()

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
    imu_figures = build_imu_plotly_figures(imu_filtered)
    
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
        imu_figures=imu_figures,
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

    df = prepared.pen_df
    imu_df = prepared.imu_df
    clip = df[(df["timestamp_unix"] >= time_start) & (df["timestamp_unix"] <= time_end)].copy()
    imu_clip = imu_df[(imu_df["timestamp_unix_aligned"] >= time_start) & (imu_df["timestamp_unix_aligned"] <= time_end)].copy()
    hand_clip = filter_hand_data_by_pct(hand_prepared, start_pct, end_pct)
    if smooth_enabled:
        clip = smooth_trajectory(clip, smooth_window)
        hand_clip = smooth_right_hand_trajectory(hand_clip, smooth_window)

    exports_dir = ROOT / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    pen_filename = "pen_data.jsonl"
    imu_filename = "imu.txt"
    hand_filename = "pred3d_world.jsonl"

    pen_path = exports_dir / pen_filename
    imu_path = exports_dir / imu_filename
    hand_path = exports_dir / hand_filename

    pen_records = clip.to_dict(orient="records")
    with open(pen_path, "w", encoding="utf-8") as f:
        for rec in pen_records:
            clean = {str(k): _jsonable(v) for k, v in rec.items()}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    imu_save_cols = [
        c
        for c in [
            "display_time",
            "device_type",
            "角度X(°)",
            "角度Y(°)",
            "角度Z(°)",
            "加速度X(g)",
            "加速度Y(g)",
            "加速度Z(g)",
            "accel_magnitude(g)",
            "timestamp_unix_aligned",
            "time_offset_seconds",
        ]
        if c in imu_clip.columns
    ]
    imu_clip.loc[:, imu_save_cols].to_csv(imu_path, sep="\t", index=False)

    hand_records = hand_clip.to_dict(orient="records")
    with open(hand_path, "w", encoding="utf-8") as f:
        for rec in hand_records:
            clean = {str(k): _jsonable(v) for k, v in rec.items()}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    params = {
        "start_pct": start_pct,
        "end_pct": end_pct,
        "forcing_pct": forcing_pct,
        "pen_length": pen_length,
        "contact_threshold": contact_threshold,
        "smooth_window": smooth_window,
        "saved_file": f"{pen_filename}, {imu_filename}, {hand_filename}",
        "saved_rows": len(pen_records),
    }
    if smooth_enabled:
        params["smooth"] = "on"

    return redirect(f"/?{urlencode(params)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

