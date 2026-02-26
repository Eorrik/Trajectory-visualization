# Trajectory Visualization

基于 **Flask + Plotly** 的书法轨迹交互可视化应用。

## 1. 快速启动

```bash
cd /workspace/Trajectory-visualization
uv sync
uv run python app.py
```

启动后访问：
- `http://127.0.0.1:8000`

## 2. 交互能力

页面提供以下控制项：

1. `time_start` 滑条：设置时间窗口起点（0%~100%）。
2. `time_end` 滑条：设置时间窗口终点（0%~100%）。
3. `time_forcing` 滑条：设置高亮时刻（默认跟随开始时间）。
4. `pen_length`：笔长参数，用于尾部延伸预测笔尖。
5. `contact_threshold`：桌面接触阈值。
6. `smooth` + `smooth_window`：轨迹平滑选项（滑动平均）用于缓解采集噪声。

## 3. 数据来源

默认读取：
- `sample/meta.json`
- `sample/pen_data.jsonl`
- `sample/20260224152819.txt`

## 4. 开发检查

```bash
uv run python -m compileall app.py src
uv run python - <<'PY'
from app import app
c = app.test_client()
print(c.get('/').status_code)
print(c.get('/?start_pct=10&end_pct=60&forcing_pct=20&smooth=on&smooth_window=7').status_code)
PY
```
