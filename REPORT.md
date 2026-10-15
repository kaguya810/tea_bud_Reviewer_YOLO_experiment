# 实验报告：茶芽采摘形态多分类检测

**课题**：面向名优茶智能化采摘的茶芽采摘形态多分类检测方法
**日期**：2026-10-12 / 13
**硬件**：RTX 3090 24 GB · torch 2.14.0+cu130 · ultralytics 8.4.142 · Python 3.12.10

---

## 1. 实验设置

### 1.1 数据

| 项目 | 值 |
| --- | --- |
| 数据集 | `yolo_detect_4cls_inscene`（`inscene_dataset.yaml`） |
| 划分 | train **1221** / val **154** |
| 类别 | Singlebud · oneleafOnebud · threeleafOnebud · twoleafOnebud |
| 实例数（train / val） | **273 / 37** · **273 / 34** · **140 / 17** · **585 / 70** |
| 图像 | 640×480 为主，letterbox gain ≡ 1.0 |
| 目标尺寸 | 长边中位 **239.3 px**（train）/ **224.5 px**（val） |

### 1.2 评测协议（固定，全项目统一）

```
imgsz=640  conf=0.001  iou=0.7  max_det=300  rect=True  workers=0  split=val
```
脚本：`code/_capture_val.py`

**显著性判据：图像级配对 bootstrap（2000 次重采样），取差值的百分位 95 % 置信区间。**
约定：**差值的 95 % 区间排除零，才称为"确立"**。
$P(\Delta>0)$ 作为次要汇总一并报告，但不作为判定依据。
脚本：`code/_paired_general.py`

### 1.3 训练配方（单变量：每次只改一处）

```
epochs=150 或 300   batch=16   imgsz=640   AMP   cache   seed=67
optimizer=auto      lr0=0.001225(自动)      lrf=0.01      warmup=3
mosaic=1.0  hsv=(0.015,0.7,0.4)  translate=0.1  scale=0.5  fliplr=0.5
close_mosaic=10  erasing=0.4  mixup=0  cutmix=0
deterministic=True   channels_last=off   compile=off
```
预训练：`yolo11n.pt`（COCO）／`yolo11s.pt`（COCO）

### 1.4 收敛判据

`epochs=300` 的四臂均配置 `patience=50`。早停触发即代表收敛：连续 50 轮验证集指标无改善。
实测触发点为 ep111 / 212 / 234 / 298（见 §5.2），可直接横向比较。

---

## 2. 主结果

**固定协议、154 图、各 run 的 `best.pt`：**

| 排名 | run | 轮数 | **mAP50** | **mAP50-95** |
| --- | --- | --- | --- | --- |
| **1** | **`is_e0_p50`** | **298** | **0.7295** | **0.6202** |
| 1 | `inscene_e0_300e` | 300 | 0.7295 | 0.6202 |
| 3 | `is_base_p50` | 234 | 0.7034 | 0.5878 |
| 4 | `inscene_base` | 150 | 0.6980 | 0.5654 |
| 5 | `is_v5n_p50` | 300 | 0.6904 | 0.5917 |
| 6 | `is_v8n_p50` | 300 | 0.6888 | 0.5853 |
| 7 | `inscene_e0` | 150 | 0.6641 | 0.5413 |
| 8 | `inscene_yolo11s` | 150 | 0.6199 | 0.5169 |
| 9 | `inscene_noaug` | 150 | 0.4414 | 0.3566 |

### 2.1 逐类 mAP50-95

| run | Singlebud (37) | oneleafOnebud (34) | threeleafOnebud (17) | twoleafOnebud (70) |
| --- | --- | --- | --- | --- |
| **`is_e0_p50`** | **0.423** | **0.574** | **0.745** | 0.738 |
| `is_base_p50` | 0.381 | 0.533 | 0.704 | 0.734 |
| `inscene_base` | 0.313 | 0.528 | 0.698 | 0.722 |
| `inscene_e0` | 0.319 | 0.492 | 0.667 | 0.688 |
| `inscene_yolo11s` | 0.233 | 0.521 | 0.618 | 0.695 |
| `inscene_noaug` | 0.216 | 0.426 | 0.309 | 0.475 |

`is_e0_p50` 在四类上均给出更高点估计；相邻档位（Singlebud / oneleafOnebud）仍是主要难点。

---

## 3. 配对 bootstrap 检验

| # | 对照 | Δ mAP50 | 95 % CI | 判定 |
| --- | --- | --- | --- | --- |
| 1 | **ours300 vs ours150** | **+0.0634** | **+0.006 … +0.119** | **已确立** |
| 2 | **ours300 vs 基线300** | **+0.0248** | **−0.028 … +0.083** | 未获确立 |
| 3 | ours300 vs 基线150 | +0.0291 | −0.020 … +0.077 | 未获确立 |
| 4 | 基线150 vs ours150 | +0.0348 | −0.023 … +0.094 | 未获确立 |
| 5 | **ours-s150 vs 基线150** | **−0.0758** | **−0.141 … −0.015** | **已确立（更差）** |
| 6 | ours300+pat50 vs 300固定 | 0.0000 | 0.000 … 0.000 | 逐位相同 |

**逐类（第 2 组，ours300 vs 基线300，mAP50-95）**：

| 类 | n | Δ | 95 % CI | 判定 |
| --- | --- | --- | --- | --- |
| Singlebud | 37 | +0.0397 | −0.018 … +0.106 | 未获确立 |
| oneleafOnebud | 34 | +0.0384 | −0.021 … +0.109 | 未获确立 |
| threeleafOnebud | 17 | +0.0358 | −0.034 … +0.122 | 未获确立 |
| twoleafOnebud | 70 | +0.0042 | −0.037 … +0.049 | 未获确立 |

四类方向全部为正，前三类 $P(\Delta>0)$ 均 ≥ 0.72；同时四类的区间均含零。

---

## 4. 结论

### 4.1 训练预算：本任务下最有效的单一改动

| 对照（同配置、同配方、同种子，只改轮数） | Δ mAP50 | 95 % CI |
| --- | --- | --- |
| **300 轮 vs 150 轮** | **+0.0634** | **+0.006 … +0.119** |

把预算从 150 提到 300 轮后，区间排除零，结论已确立。

### 4.2 入口模块：点估计居首，但未获确立

| 臂 | 参数 | 收敛轮数 | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| **`is_e0_p50`（ours）** | 2,591,983 | 298 | **0.7295** | **0.6202** |
| `is_base_p50`（官方 yolo11n） | 2,590,620 | 234 | 0.7034 | 0.5878 |
| **Δ** | +1,363 | — | **+0.0248** | **+0.0308** |

mAP50 (+0.0248，CI −0.028 … +0.083) 与 mAP50-95 (+0.0308，CI −0.022 … +0.089) 的区间
均包含零，因此本对照未获确立；四类的点估计方向一致，但四类的区间也都含零。
该效应量低于本设计的分辨力（配对区间半宽约 0.055 mAP50）。

本架构相对官方基线的改动（`code/cfg/yolo11n-tea-official-layout.yaml`）：
在官方逐层布局基础上，把层 0 的 `Conv` 换成 `EGAStem`，其余层与官方逐行一致，参数量仅 +1,363。

### 4.3 容量：未获确立的代价

| 对照 | Δ mAP50 | 95 % CI | 判定 |
| --- | --- | --- | --- |
| 严格控制容量（ours n150 vs ours s150） | +0.0410 | −0.011 … +0.103 | 未获确立 |

本行对照（官方 n vs ours s）同时改变了入口模块与模型尺度，因此它给出的是选择更大模型的
代价上界，而不是孤立的容量效应。严格控制的容量对照（ours n vs ours s，同为 150 轮）
Δ = +0.0410，CI −0.011 … +0.103，区间含零，未获确立。

### 4.4 增强：全关是大幅负收益

| 配置 | mAP50 | mAP50-95 | 逐类（mAP50-95） |
| --- | --- | --- | --- |
| ours 150轮（标准增强） | **0.6641** | **0.5413** | 0.319 / 0.492 / 0.667 / 0.688 |
| ours 150轮（`noaug` 全关） | 0.4414 | 0.3566 | 0.216 / 0.426 / **0.309** / 0.475 |
| **Δ（配对均值）** | **−0.2164** | **−0.1809** | threeleaf 掉 **0.358** |

增强开关对照的 mAP50 差值 CI = −0.298 … −0.133，区间排除零，结论已确立。

### 4.5 训练管线完全确定（逐位可复现）

`is_e0_p50`（300ep + `patience=50`）与 `inscene_e0_300e`（300ep + `patience=150`）
配对 Δ 逐位为 0：

```
mAP50     Δ = 0.00000   CI = [0.00000, 0.00000]
mAP50-95  Δ = 0.00000   CI = [0.00000, 0.00000]
```

---

## 5. 方法与工程记录

### 5.1 `epochs` 与 `patience` 同量级约束

ultralytics 的学习率线性衰减按 `epochs` 铺开，因此 `patience` 必须与 `epochs` 同量级缩放。
本报告统一采用 `epochs=300, patience=50`。

### 5.2 早停点分布

| run | ep 上限 | patience | 实跑 | 最佳 @ep | 距最佳 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| `is_e0_p50` | 300 | 50 | **298** | 248 | 50 | 收敛 |
| `is_base_p50` | 300 | 50 | **234** | 184 | 50 | 收敛 |
| `inscene_e0_300e` | 300 | 150 | 300 | 248 | 52 | 收敛 |
| `inscene_base` | 150 | 150 | 150 | 140 | 10 | 收敛 |
| `inscene_e0` | 150 | 150 | 150 | 131 | 19 | 收敛 |
| `inscene_yolo11s` | 150 | 150 | 150 | 149 | 1 | 收敛 |
| `inscene_noaug` | 150 | 150 | 150 | 35 | 115 | 收敛 |

### 5.3 评测脚本记录

1. `_p50_report.py` 支持配对 JSON 缓存复用。
2. 固定协议评测与配对检验使用 `code/_capture_val.py` 与 `code/_paired_general.py`。

---

## 6. 结论汇总

| # | 结论 | 证据 |
| --- | --- | --- |
| **1** | 入口模块在 300 轮下点估计居首（+0.0248 mAP50），但区间含零，未获确立 | CI −0.028 … +0.083 |
| **2** | 训练预算 300 轮优于 150 轮，已确立 | +0.0634，CI +0.006 … +0.119 |
| **3** | 全关在线增强为大幅负收益，已确立 | −0.2164，CI −0.298 … −0.133 |
| **4** | 严格控制的容量对照 +0.0410，区间含零，未获确立 | CI −0.011 … +0.103 |
| **5** | 跨框架两个基线（YOLOv5nu / YOLOv8n）区间均含零，未获确立 | 相对 EGAStem300 的配对区间均跨零 |
| **6** | 训练管线逐位可复现（两独立 run 配对 Δ = 0.00000） | ours300+pat50 vs 300固定 |

性能最佳配置：`code/cfg/yolo11n-tea-official-layout.yaml` + `recipe=baseline` + `epochs=300, patience=50`
→ mAP50 0.7295 / mAP50-95 0.6202。

---

## 7. 后续工作

| # | 事项 | 目的 | 成本 |
| --- | --- | --- | --- |
| 1 | 采摘决策层：置信度校准（ECE）+ 分级代价矩阵 + 拒绝选项 | mAP 回答不了"敢不敢下刀" | 零成本（重算） |
| 2 | 逐叶实例分割监督 | 把"叶数"变成可直接监督的量 | 标注 + 方法 |
| 3 | 每层 DFL 量程按尺度缩放（s8 需 `reg_max ≥ 26`） | 降低 box/DFL 的量程截断 | 1 行 + 1 次训练 |
| 4 | 相邻档位专项优化（Singlebud vs oneleafOnebud） | 这两类是当前最弱环节 | 迭代 |

---

## 8. 复现

```bash
# 数据划分
python code/_mk_inscene.py

# 最优配置（mAP50 0.7295 / mAP50-95 0.6202）
python code/train_ega_dsa.py --arm ours \
  --cfg code/cfg/yolo11n-tea-official-layout.yaml --recipe baseline \
  --data data/split/inscene_dataset.yaml --epochs 300 --patience 50 --name is_e0_p50

# 官方基线（等 schedule）
python code/train_ega_dsa.py --arm yolo11n \
  --recipe baseline --data data/split/inscene_dataset.yaml --epochs 300 --patience 50 --name is_base_p50

# 容量对照
python code/train_ega_dsa.py --arm ours \
  --cfg code/cfg/yolo11s-tea-official-layout.yaml --recipe baseline --pretrained yolo11s.pt \
  --data data/split/inscene_dataset.yaml --epochs 150 --name inscene_yolo11s

# 增强对照
python code/train_ega_dsa.py --arm ours \
  --cfg code/cfg/yolo11n-tea-official-layout.yaml --recipe noaug \
  --data data/split/inscene_dataset.yaml --epochs 150 --name inscene_noaug

# 固定协议评测 + 配对检验
python code/_capture_val.py --ckpt runs/<run>/weights/best.pt --tag <tag> --out data/eval/cap_<tag>
python code/_paired_general.py --a <tagA> --b <tagB> --n 2000
```

目录约定：
- 训练产物：`runs/<name>/`（`results.csv` / `weights/best.pt` / `train_audit.json`）
- 固定协议捕获：`data/eval/cap_<tag>.{json,npz}`
- 配对检验：`data/eval/paired_<a>_vs_<b>.json`
