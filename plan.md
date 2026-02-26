# 书法采集数据可视化执行计划（plan）

## 1. 范围与目标
- **目标**：基于 Python 3.x + Pyecharts，完成书法轨迹数据读取、坐标转换、时间对齐与 3D 交互可视化。
- **本期实现范围**：
  1. 解析 `sample/meta.json`、`sample/pen_data.jsonl`、IMU `TXT`。
  2. 将毛笔 marker 相机坐标转换到世界坐标，产出 `tip_pos_world` / `tail_pos_world`。
  3. 统一时间轴（内部 Unix 时间戳），对外展示 `yy-mm-dd-hh:mm:ss`。
  4. 生成可按 `time_start` ~ `time_end` 过滤的 3D 轨迹图，并支持 `time_forcing` 高亮。
  5. 在每个毛笔时间点附加 WT1 插值 IMU 数据。
- **暂不实现（仅规划）**：视频 + YOLO 手部关键位与 WT2/WT3/WT4 融合。

---

## 2. 里程碑与分阶段任务

### M1：需求澄清与数据勘察（0.5 天）
**产出**：字段映射表、时间戳格式说明、异常样本清单。
- 梳理 `meta.json` 中 `depth_intrinsics` 与 `desk_plane` 的数学含义。
- 统计 `pen_data.jsonl` 字段完整性、采样频率、缺失情况。
- 解析 `TXT` 的列定义（WT1~WT4 各维度 + 时间戳格式）。

### M2：数据解析层实现（1 天）
**产出**：统一数据模型 + 解析模块。
- `meta` 解析器：读取内参与桌面平面定义。
- `pen_data` 解析器：逐行读取 JSONL，提取两个 marker 的 3D 坐标与时间戳。
- `imu_txt` 解析器：支持 WT1~WT4 的多维时序读取。
- 建立统一结构（建议 DataFrame/TypedDict）：
  - `timestamp_unix`
  - `tip_pos_cam`, `tail_pos_cam`
  - `imu_wt1/2/3/4` 原始与标准化字段。

### M3：坐标转换与几何计算（1 天）
**产出**：世界坐标结果与几何工具函数。
- 依据 `depth_intrinsics` 完成相机坐标到世界坐标变换。
- 使用 `desk_plane`（原点 + 法向量）建立桌面坐标基准。
- 为每条 pen 记录计算并存储：
  - `tip_pos_world`
  - `tail_pos_world`
- 实现“预测笔尖点”：沿 `tail_pos_world` 方向按预设笔长外推。
- 桌面接触判定：预测笔尖到桌面距离 < 阈值记为接触点。

### M4：时间同步与插值（1 天）
**产出**：跨源时间对齐模块。
- 将 IMU 原始时间转换为 Unix 时间戳。
- 建立 pen 时间轴主索引，对 WT1 取最近前后两点线性插值。
- 生成对外展示时间字符串（`yy-mm-dd-hh:mm:ss`），但内部计算保持 Unix。
- 异常处理：
  - 超出 IMU 时间范围时（头尾外推策略或置空策略）。
  - 重复时间戳/乱序时间戳的清洗策略。

### M5：3D 可视化实现（1.5 天）
**产出**：可运行图表页面/HTML。
- 使用 Pyecharts 3D 图（Grid3D/Line3D/Scatter3D 组合）。
- 按参数过滤：`time_start`, `time_end`。
- 绘制规则：
  - `tip_pos_world` 红色轨迹（随时间渐变）。
  - `tail_pos_world` 蓝色轨迹（随时间渐变）。
  - 同时刻 tip-tail 灰色细连线。
  - tail 到预测笔尖的细线。
  - 桌面接触点使用红色地面散点。
- `time_forcing`：
  - 目标时刻附近轨迹颜色加深。
  - 离该时刻越远，颜色越浅。
- Tooltip 中附加 WT1 插值数据与格式化时间。

### M6：参数接口与最小交互（0.5 天）
**产出**：可配置运行入口。
- 输入参数：
  - `time_start`, `time_end`, `time_forcing`
  - `pen_length`
  - `contact_threshold`
- 输出：可交互 Python Web 应用（基于 HTML 页面）+ 可选中间结果 CSV/JSON。

### M7：测试与验收（1 天）
**产出**：测试报告 + 验收清单。
- 单元测试：
  - 坐标转换正确性（已知输入/输出）。
  - 时间插值正确性（线性插值边界案例）。
- 集成测试：
  - 从样本输入到 HTML 输出全链路跑通。
- 可视化验收：
  - 颜色、连线、高亮、接触点是否符合规范。
  - 时间筛选是否生效。

---

## 3. 建议目录结构
```text
Trajectory-visualization/
  plan.md
  project.md
  sample/
  src/
    parsers/
      meta_parser.py
      pen_parser.py
      imu_parser.py
    transforms/
      coordinate_transform.py
      time_align.py
      interpolation.py
    viz/
      trajectory_3d.py
    main.py
  tests/
    test_coordinate_transform.py
    test_time_align.py
```

---

## 4. 关键算法与实现要点
- **坐标变换**：优先明确 `depth_intrinsics` 对应模型（针孔模型/已去畸变）；若数据已在相机三维坐标，需明确是否仅做刚体变换到桌面世界系。
- **插值策略**：WT1 每个维度独立线性插值；对角度类字段需考虑跨 180/-180 的环绕问题。
- **高亮策略**：基于与 `time_forcing` 的时间差定义 alpha 或亮度权重。
- **性能**：先保证样本数据可视化正确，再考虑下采样（如每 N 点）优化渲染。

---

## 5. 风险与应对
- 风险 1：TXT 时间戳格式不稳定。  
  - 应对：实现多格式解析器 + 日志提示 + 回退策略。
- 风险 2：坐标系定义不完整导致世界坐标偏移。  
  - 应对：引入可视化校验（桌面平面、基准点），必要时增加手动校准参数。
- 风险 3：Pyecharts 3D 在大点数时卡顿。  
  - 应对：提供采样级别参数；默认展示关键区间。

---

## 6. 验收标准（Definition of Done）
- 能成功读取 sample 中 JSON/TXT 数据并完成解析。
- 每条 pen 记录都可得到 `tip_pos_world` / `tail_pos_world`。
- IMU 与 pen 时间统一到 Unix 时间戳，并可输出可读时间。
- 可生成满足视觉规范的 3D 轨迹图（含高亮、连线、接触点、WT1 提示）。
- 提供可复现运行命令与基础测试结果。

---

## 7. 后续规划（非本期）
- 接入视频（mp4）与 YOLO 手部关键点识别。
- 与 WT2/WT3/WT4 融合重建手背-手腕-手肘 3D 运动链。
- 在同一时轴中联动展示笔轨迹与上肢动作。
