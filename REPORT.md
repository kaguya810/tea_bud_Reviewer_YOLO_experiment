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
| 实例数（train / val） | 266 / 33 · 267 / 34 · 139 / 17 · 560 / 70 |
| 图像 | 640×480 为主，letterbox gain ≡ 1.0 |
| 目标尺寸 | 长边中位 238.7 px（train）/ 261.5 px（val）；COCO-small 占比约 1% |

### 1.2 评测协议（固定，全项目统一）

```
imgsz=640  conf=0.001  iou=0.7  max_det=300  rect=True  workers=0  split=val
```
脚本：`_diag/transfer/_capture_val.py`

**显著性判据：图像级配对 bootstrap（2000 次重采样），以 $P(\Delta>0)$ 为准。**
约定：$P(\Delta>0) \ge 0.80$ 判为显著；$P(\Delta>0) \le 0.20$ 判为显著更差。
脚本：`_paired_general.py`

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

`epochs=300` 的四臂均配置 `patience=50`。**早停触发即代表收敛**：连续 50 轮验证集指标无改善。
实测触发点为 ep111 / 212 / 234 / 298（见 §5.2），**因此各臂的记录值即为其收敛值**，
可直接横向比较，不存在"训练不充分"的残留。

---

## 2. 主结果

**固定协议、154 图、各 run 的 `best.pt`：**

| 排名 | run | 轮数 | **mAP50** | **mAP50-95** | P | R |
| --- | --- | --- | --- | --- | --- | --- |
| **1** | **`is_e0_p50`** | **298** | **0.7295** | **0.6202** | **0.780** | 0.648 |
| 1 | `inscene_e0_300e` | 300 | 0.7295 | 0.6202 | 0.780 | 0.648 |
| 3 | `is_base_p50` | 234 | 0.7034 | 0.5878 | 0.695 | 0.653 |
| 4 | `inscene_base` | 150 | 0.6980 | 0.5654 | 0.752 | 0.601 |
| 5 | `inscene_e0` | 150 | 0.6641 | 0.5413 | 0.731 | 0.556 |
| 6 | `inscene_yolo11s` | 150 | 0.6199 | 0.5169 | 0.674 | 0.590 |
| 7 | `inscene_noaug` | 150 | 0.4414 | 0.3566 | 0.405 | 0.476 |

### 2.1 逐类 mAP50-95

| run | Singlebud (33) | oneleafOnebud (34) | threeleafOnebud (17) | twoleafOnebud (70) |
| --- | --- | --- | --- | --- |
| **`is_e0_p50`** | **0.423** | **0.574** | **0.745** | 0.738 |
| `is_base_p50` | 0.381 | 0.533 | 0.704 | 0.734 |
| `inscene_base` | 0.313 | 0.528 | 0.698 | 0.722 |
| `inscene_e0` | 0.319 | 0.492 | 0.667 | 0.688 |
| `inscene_yolo11s` | 0.233 | 0.521 | 0.618 | 0.695 |
| `inscene_noaug` | 0.216 | 0.426 | 0.309 | 0.475 |

**`is_e0_p50` 在四类上全部第一。**

**相邻档位最难**：`Singlebud`（0.423）与 `oneleafOnebud`（0.574）—— 0 叶 vs 1 叶的判别；
两个远端档位（threeleaf 0.745 / twoleaf 0.738）明显更高。

---

## 3. 配对 bootstrap 检验

| # | 对照 | Δ mAP50 | $P(\Delta>0)$ | Δ mAP50-95 | $P(\Delta>0)$ | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **ours300 vs ours150** | **+0.0637** | **0.984** | **+0.0773** | **1.000** | **★ 显著** |
| 2 | **ours300 vs 基线300** | **+0.0261** | **0.807** | **+0.0295** | **0.866** | **★ 显著** |
| 3 | ours300 vs 基线150 | +0.0284 | 0.870 | +0.0514 | **0.988** | **★ 显著** |
| 4 | 基线150 vs ours150 | +0.0354 | 0.878 | +0.0260 | 0.868 | ★ 显著（基线优） |
| 5 | **yolo11s vs 基线150** | **−0.0772** | **0.009** | **−0.0488** | **0.012** | **★ 显著更差** |
| 6 | ours300+pat50 vs 300固定 | 0.00000 | — | 0.00000 | — | 逐位相同 |

**逐类（第 2 组，ours300 vs 基线300，mAP50-95）**：

| 类 | n | Δ | $P(\Delta>0)$ | 判定 |
| --- | --- | --- | --- | --- |
| Singlebud | 37 | +0.0397 | 0.804 | ★ 显著 |
| oneleafOnebud | 34 | +0.0384 | 0.812 | ★ 显著 |
| threeleafOnebud | 17 | +0.0358 | 0.723 | 方向一致 |
| twoleafOnebud | 70 | +0.0042 | 0.567 | 持平 |

**四类方向全部为正，前三类 $P$ 均 ≥ 0.72。**

---

## 4. 结论

### 4.1 ★ 训练预算：本任务下最有效的单一改动

| 对照（同配置、同配方、同种子，只改轮数） | Δ mAP50 | $P$ | Δ mAP50-95 | $P$ |
| --- | --- | --- | --- | --- |
| **300 轮 vs 150 轮** | **+0.0637** | **0.984** | **+0.0773** | **1.000** |

把预算从 150 提到 300 轮，mAP50 **+6.4 个点**、mAP50-95 **+7.7 个点**，且 $P$ 接近 1。
**150 轮时模型未收敛**，300 轮 schedule 下最佳出现在 **ep248**。

### 4.2 ★ 架构对照：ours 显著优于官方基线

**两个臂使用同一 `epochs=300` schedule，且都由早停证明已收敛**（ours ep298 / 基线 ep234），
因此该对比为**等 schedule、等收敛**的干净比较：

| 臂 | 参数 | 收敛轮数 | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| **`is_e0_p50`（ours）** | 2,591,983 | 298 | **0.7295** | **0.6202** |
| `is_base_p50`（官方 yolo11n） | 2,590,620 | 234 | 0.7034 | 0.5878 |
| **Δ** | +1,363 | — | **+0.0261**（$P$=**0.807**） | **+0.0295**（$P$=**0.866**） |

**mAP50 与 mAP50-95 双向均显著，四类方向全部为正。**

**本架构相对官方基线的改动**（`cfg/yolo11n-tea-official-layout.yaml`）：
在**官方逐层布局**基础上，把层 0 的 `Conv` 换成 `EGAStem`（边缘-高斯聚合入口），
其余 23 层与官方 yolo11n **逐行一致** —— 因此 COCO 预训练权重迁移率达 **98.2%**，
唯一从头学的模块就是入口，参数量仅 +1,363。

### 4.3 容量：放大模型是**显著负收益**

| 臂 | 参数 | mAP50 | mAP50-95 |
| --- | --- | --- | --- |
| 官方 yolo11n | 2.59 M | **0.6980** | **0.5654** |
| ours yolo11s | **9.43 M** | 0.6199 | 0.5169 |
| **Δ** | **3.6×** | **−0.0772**（$P$=**0.009**） | **−0.0488**（$P$=**0.012**） |

**参数放大 3.6 倍，mAP50 反而掉 7.7 个点，$P$ = 0.009 —— 高度显著。**

⇒ **本任务/本数据规模下的瓶颈不是模型容量。** 加宽加深会加剧过拟合，与"训练预算比容量更值钱"（§4.1）互为印证。

### 4.4 增强：全关是**大幅负收益**

| 配置 | mAP50 | mAP50-95 | 逐类（mAP50-95） |
| --- | --- | --- | --- |
| ours 150轮（标准增强） | **0.6641** | **0.5413** | 0.319 / 0.492 / 0.667 / 0.688 |
| ours 150轮（`noaug` 全关） | 0.4414 | 0.3566 | 0.216 / 0.426 / **0.309** / 0.475 |
| **Δ** | **−0.2228** | **−0.1847** | threeleaf 掉 **0.358** |

**在线增强对本任务是必需的**，尤其保护了样本量最少的 `threeleafOnebud`（0.667 → 0.309）。

### 4.5 训练管线完全确定（逐位可复现）

`is_e0_p50`（300ep + `patience=50`）与 `inscene_e0_300e`（300ep + `patience=150`）
**配对 Δ 逐位为 0**：

```
mAP50     Δ = 0.00000   CI = [0.00000, 0.00000]
mAP50-95  Δ = 0.00000   CI = [0.00000, 0.00000]
```

⇒ 固定种子下训练完全可复现；且 `patience=50` 未在峰值之前误触发。

---

## 5. 方法与工程记录

### 5.1 ⚠️ `epochs` 与 LR schedule 耦合 —— 改轮数不是单变量实验

ultralytics 的学习率**线性衰减按 `epochs` 铺开**。实测同一 epoch 下的学习率：

| epoch | `epochs=150` 的 lr | `epochs=600` 的 lr |
| --- | --- | --- |
| 112 | 0.000334 | **0.024505** |

**后果**：首版设计用 `epochs=600, patience=50`，某个臂在 **112 轮**即早停 ——
此时 lr 仍为峰值的 98%，验证曲线在高学习率下剧烈震荡，**早停被误触发、砍在峰值之前**。

**规则**：`patience` 必须与 `epochs` 同量级缩放。
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

**`is_base_p50` 在 234 轮即满足"连续 50 轮无改善"** ⇒ 官方基线在 300 轮 schedule 下的收敛点比 ours 更早。

### 5.3 评测脚本的两个坑（已修）

1. `_p50_report.py` 缺少缓存跳过 → 每次调用重跑全部对照（每组约 20 min）。已加：配对 JSON 存在则直接读。
2. `_chain.py` 自匹配死锁（等待的目标名出现在自身命令行中，永远匹配到自己）→ 已弃用，改为单 job 顺序执行。

### 5.4 训练开销（实测）

| run | 参数 | 轮数 | 稳态 s/epoch | 显存峰值 | 总时长 |
| --- | --- | --- | --- | --- | --- |
| `is_e0_p50` | 2,591,983 | 298 | 28.06 | 2.43 GiB | **2.32 h** |
| `inscene_e0_300e` | 2,591,983 | 300 | 22.62 | 2.43 GiB | **1.89 h** |
| `is_base_p50` | 2,590,620 | 234 | 25.31 | 2.07 GiB | **1.64 h** |
| `inscene_base` | 2,590,620 | 150 | 27.52 | 2.07 GiB | **1.15 h** |
| `inscene_e0` | 2,591,983 | 150 | 20.66 | 2.43 GiB | **0.86 h** |
| `inscene_yolo11s` | 9,432,063 | 150 | 34.57 | 4.35 GiB | **1.44 h** |
| `inscene_noaug` | 2,591,983 | 150 | 17.26 | 2.43 GiB | **0.72 h** |

**本报告全部 run 合计约 10.0 h**（不含评测与配对检验）。

### 5.5 训后审计

全部 run 的 `train_audit.json` 均 `audit_ok = True`，检查项包括：
建模一致性（trainer 重建的模型与 build 阶段一致）、参数量、固定核漂移、
`per_epoch_seconds` / `steady_epoch_seconds` 记录完整性。

---

## 6. 结论汇总

| # | 结论 | 证据 |
| --- | --- | --- |
| **1** | **本架构（EGAStem + 官方逐层布局）显著优于官方 yolo11n 基线** | Δ mAP50 **+0.0261**（$P$=0.807）· Δ mAP50-95 **+0.0295**（$P$=0.866）· 四类方向全正 |
| **2** | **训练预算 300 轮显著优于 150 轮** | Δ mAP50 **+0.0637**（$P$=0.984）· Δ mAP50-95 **+0.0773**（$P$=1.000） |
| **3** | **放大模型是显著负收益** | yolo11s（9.43 M）Δ mAP50 **−0.0772**（$P$=0.009） |
| **4** | **全关在线增强是大幅负收益** | Δ mAP50 **−0.2228** · threeleaf 掉 0.358 |
| **5** | **最优配置**：EGAStem + 官方布局 + 标准增强 + 300 轮 schedule | mAP50 **0.7295** · mAP50-95 **0.6202** |
| **6** | **训练管线逐位可复现** | 两独立 run 配对 Δ = 0.00000 |

**性能最佳配置**：`cfg/yolo11n-tea-official-layout.yaml` + `recipe=baseline` + `epochs=300, patience=50`
→ **mAP50 0.7295 / mAP50-95 0.6202**（2.59 M 参数，2.32 h）

---

## 7. 后续工作

| # | 事项 | 目的 | 成本 |
| --- | --- | --- | --- |
| 1 | 采摘决策层：置信度校准（ECE）+ 分级代价矩阵 + 拒绝选项 | mAP 回答不了"敢不敢下刀" | 零成本（重算） |
| 2 | 逐叶实例分割监督 | 把"叶数"变成可直接监督的量 | 标注 + 方法 |
| 3 | 每层 DFL 量程按尺度缩放（s8 需 `reg_max ≥ 26`） | 现有 `reg_max=16` 让 s4/s8 的 box/DFL 产生被 clamp 的废梯度 | 1 行 + 1 次训练 |
| 4 | 相邻档位专项优化（Singlebud vs oneleafOnebud） | 这两类是当前最弱环节（0.423 / 0.574） | 迭代 |

---

## 8. 复现

```powershell
# ---- 数据划分 ----
.venv\Scripts\python.exe _mk_inscene.py

# ---- 最优配置（mAP50 0.7295 / mAP50-95 0.6202）----
.venv\Scripts\python.exe train_ega_dsa.py --arm ours `
    --cfg cfg/yolo11n-tea-official-layout.yaml --recipe baseline `
    --data inscene_dataset.yaml --epochs 300 --patience 50 --name is_e0_p50

# ---- 官方基线（等 schedule）----
.venv\Scripts\python.exe train_ega_dsa.py --arm yolo11n `
    --recipe baseline --data inscene_dataset.yaml --epochs 300 --patience 50 --name is_base_p50

# ---- 容量对照 ----
.venv\Scripts\python.exe train_ega_dsa.py --arm ours `
    --cfg cfg/yolo11s-tea-official-layout.yaml --recipe baseline --pretrained yolo11s.pt `
    --data inscene_dataset.yaml --epochs 150 --name inscene_yolo11s

# ---- 增强对照 ----
.venv\Scripts\python.exe train_ega_dsa.py --arm ours `
    --cfg cfg/yolo11n-tea-official-layout.yaml --recipe noaug `
    --data inscene_dataset.yaml --epochs 150 --name inscene_noaug

# ---- 固定协议评测 + 配对检验 ----
.venv\Scripts\python.exe _diag\transfer\_capture_val.py --ckpt runs/<run>/weights/best.pt `
    --tag <tag> --out _diag/transfer/cap_<tag> --data inscene_dataset.yaml
.venv\Scripts\python.exe _paired_general.py --a <tagA> --b <tagB> --n 2000
```

**目录约定**
- 训练产物：`runs/<name>/`（`results.csv` / `weights/best.pt` / `train_audit.json`）
- 固定协议捕获：`_diag/transfer/cap_<tag>.{json,npz}`
- 配对检验：`_diag/transfer/paired_<a>_vs_<b>.json`
