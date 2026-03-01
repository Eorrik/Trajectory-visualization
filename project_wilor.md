# 项目文档：export_hand_to_coord.py 使用与输出定义

## 概述
- 功能：从视频逐帧估计手部关键点，导出为 JSONL。包含图像 2D 关键点、手局部 3D、相机空间 3D，以及基于桌面平面（desk_plane）的世界坐标 3D。
- 依赖：WiLoR-mini 推理管线（YOLO 手检测 + WiLoR 手部 3D 姿态）、MANO 模型、OpenCV 等。

## 环境
- 推荐使用虚拟环境：`d:\WiLoR-mini\WiLoR-mini\.venv-cu124\`
- 运行解释器：`d:\WiLoR-mini\WiLoR-mini\.venv-cu124\Scripts\python.exe`
- Python 版本建议：3.10

## 脚本位置
- `export_hand_to_coord.py`

## 命令行参数
- `--video_path` 必填。输入视频路径（例如 `assets/video.mp4`）。
- `--output_path` 选填。默认 `results/preds.jsonl`。导出图像 2D、局部 3D、相机 3D。
- `--world_output_path` 选填。默认 `results/preds_world.jsonl`。导出世界坐标 3D。
- `--meta_path` 选填。默认 `assets/meta.json`。读取相机与桌面平面信息（desk_plane）。
- `--hand_conf` 选填。默认 `0.3`。手检测置信度阈值。
- `--rescale_factor` 选填。默认 `2.5`。检测框扩展比例。
- `--max_frames` 选填。默认 `None`。最大处理帧数（不传则处理到视频结束）。

## 运行示例
```bash
d:\WiLoR-mini\WiLoR-mini\.venv-cu124\Scripts\python.exe export_hand_to_coord.py \
  --video_path assets/video.mp4 \
  --output_path results/preds.jsonl \
  --world_output_path results/preds_world.jsonl \
  --meta_path assets/meta.json
```

## 输出文件与字段定义

### 1) preds.jsonl（每行一帧）
- 结构：
  - `hands`: 按检测顺序的左右手标签数组（字符串 `"right"` 或 `"left"`）。
  - `pred2d`: 形如 `[[[x, y], ...], ...]`。每只手对应一组 2D 关键点；单位像素；坐标系为图像像素坐标（原点左上，x 向右，y 向下）。
  - `pred3d`: 形如 `[[[x, y, z], ...], ...]`。MANO 关节在手的局部坐标系下的 3D 点（已应用整体姿态旋转 global_orient，未叠加相机平移）；数值量级接近手部尺度。
  - `pred3d_cam`: 形如 `[[[x, y, z], ...], ...]`。相机空间 3D 点，计算方式为 `pred3d + pred_cam_t_full`。

> 说明：`pred2d` 通过透视投影得到，使用像素焦距与图像中心：`utils.perspective_projection(points, translation=pred_cam_t_full, focal_length=[fx, fy], camera_center=[cx, cy])`。

### 2) preds_world.jsonl（每行一帧）
- 结构：
  - `hands`: 与 `preds.jsonl` 一致。
  - `pred3d_world`: 形如 `[[[x, y, z], ...], ...]`。世界坐标 3D 点。

- 世界坐标系定义（desk_plane）：
  - 从 `assets/meta.json` 读取 `desk_plane=[a, b, c, d]`，平面方程为 `a x + b y + c z + d = 0`。
  - 法向量 `n=[a, b, c]`，单位法向 `Z_w = n / ||n||`。
  - 选取参考向量 `ref` 与 `Z_w` 正交，构造 `X_w = normalize(ref × Z_w)`，`Y_w = Z_w × X_w`。
  - 取世界原点为平面上的点 `p0 = -d * n / ||n||^2`（与相机原点沿法线最近的平面点）。
  - 相机空间点 `p_cam` 转换到世界坐标：`p_world = R_cam_to_world · (p_cam - p0)`，其中 `R_cam_to_world = [X_w, Y_w, Z_w]^T`。
  - 世界坐标系的 `Z_w` 轴为平面法向；`X_w`、`Y_w` 为平面内的正交基，桌面即世界坐标的 `x–y` 平面。

> 说明：若 `desk_plane` 未提供或数值非法，则不生成世界坐标行（对应行缺省）。

## 字段维度与单位
- 关键点数量：与 MANO/开源手部关键点定义一致（通常 21 个）。
- `pred2d`：单位像素；坐标在原始图像分辨率下（与输入帧尺寸一致）。
- `pred3d` 与 `pred3d_cam`：单位与 MANO 模型一致，量级为米级（与模型训练设置有关）。
- `pred3d_world`：同上单位；坐标系为桌面平面定义的世界坐标（`Z_w` 垂直平面）。

## 处理细节
- 左右手：对左手进行 `x` 轴镜像以保持与相机坐标左右一致，然后使用相机参数生成 2D 投影。
- 焦距：像素焦距由 `scaled_focal_length = FOCAL_LENGTH / 256 * max(image_side)` 得到（与管线内部一致）。
- 2D 投影：标准针孔相机模型，`x' = fx * (X/Z) + cx`，`y' = fy * (Y/Z) + cy`。

## 典型样例（单帧，字段截断）
```json
{
  "hands": ["right"],
  "pred2d": [[[1445.47, 586.09], ...]],
  "pred3d": [[[-0.0957, 0.00638, 0.00620], ...]],
  "pred3d_cam": [[[0.1298, 0.01289, 10.0586], ...]]
}
```

```json
{
  "hands": ["right"],
  "pred3d_world": [[[0.4013, -9.05, -3.97], ...]]
}
```

## 常见问题
- `pred3d` 与 `pred3d_cam`：前者为手的局部坐标（相机平移未加），后者为相机空间坐标（已加 `pred_cam_t_full`）。
- 世界坐标原点选择：当仅给定平面方程时，本脚本将原点选为距离相机原点最近的平面点 `p0`；如需指定其他参考点或引入完整外参，请在 `assets/meta.json` 中补充并扩展脚本转换逻辑。

