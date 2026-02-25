from __future__ import annotations

from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request

from src.data_processing import prepare_data
from src.visualization import build_chart

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
META_PATH = ROOT / "sample" / "meta.json"
PEN_PATH = ROOT / "sample" / "pen_data.jsonl"
IMU_PATH = ROOT / "sample" / "20260224152819.txt"

prepared = prepare_data(META_PATH, PEN_PATH, IMU_PATH)


def parse_dt_local(value: str | None, fallback: float) -> float:
    if not value:
        return fallback
    try:
        dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        try:
            dt = datetime.strptime(value, "%Y-%m-%dT%H:%M")
        except ValueError:
            return fallback
    return dt.timestamp()


def format_dt_local(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


@app.route("/", methods=["GET"])
def index():
    q = request.args
    time_start = parse_dt_local(q.get("time_start"), prepared.min_ts)
    time_end = parse_dt_local(q.get("time_end"), prepared.max_ts)
    if time_start > time_end:
        time_start, time_end = time_end, time_start

    forcing = q.get("time_forcing")
    forcing_ts = parse_dt_local(forcing, (time_start + time_end) / 2.0) if forcing else None

    pen_length = float(q.get("pen_length", 0.18))
    contact_threshold = float(q.get("contact_threshold", 0.01))

    df = prepared.df
    filtered = df[(df["timestamp_unix"] >= time_start) & (df["timestamp_unix"] <= time_end)].copy()
    if filtered.empty:
        filtered = df.head(10).copy()

    chart = build_chart(filtered, forcing_ts, pen_length=pen_length, contact_threshold=contact_threshold)
    chart_html = chart.render_embed()

    wt1_cols = [c for c in filtered.columns if c.startswith("wt1_")]
    sample_preview = filtered[["display_time", *wt1_cols[:4]]].head(12).to_dict(orient="records")

    return render_template(
        "index.html",
        chart_html=chart_html,
        time_start=format_dt_local(time_start),
        time_end=format_dt_local(time_end),
        time_forcing=format_dt_local(forcing_ts) if forcing_ts else "",
        pen_length=pen_length,
        contact_threshold=contact_threshold,
        min_ts=format_dt_local(prepared.min_ts),
        max_ts=format_dt_local(prepared.max_ts),
        rows=len(filtered),
        sample_preview=sample_preview,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
