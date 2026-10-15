# 茶芽采摘形态多分类检测 —— 实验包

面向名优茶智能化采摘的茶芽采摘形态多分类检测方法 · 实验报告与数据

---

## 1. 包内内容

```
tea_bud_picking_experiment/
├── README.md                本文件
├── REPORT.md                实验报告（正文）
├── manifest.json            全部文件的 sha256 + 字节数（溯源）
│
├── data/
│   ├── split/               数据划分定义与统计
│   │   ├── inscene_dataset.yaml    数据集 yaml（train 1221 / val 154）
│   │   └── SPLIT_REPORT.json       划分统计（各类实例数）
│   ├── runs/<run>/          每个 run 的原始训练记录
│   │   ├── results.csv             逐轮指标（含 lr、时间、各类损失）
│   │   ├── args.yaml               ultralytics 完整超参快照
│   │   └── train_audit.json        训后审计（参数量、收敛、显存、检查项）
│   ├── eval/                固定协议评测与配对检验原始结果
│   │   ├── cap_*.json              各 checkpoint 的逐图统计汇总
│   │   ├── cap_*.npz               逐图 tp/conf/pred_cls/target_cls（配对检验的输入）
│   │   └── paired_p50_*.json       6 组配对 bootstrap 完整输出
│   └── summary/
│       ├── runs_summary.csv        7 个 run 的汇总表（Excel 可直接打开）
│       └── runs_summary.json       同上 + 逐类 mAP50-95
│
├── weights/                 7 个 run 的 best.pt（每个约 5.4 MB）
│   └── <run>_best.pt
│
├── figs/                    数据集图样
│   ├── dataset_samples.png     ★ 主图：4 类 × 6 例，实例裁剪 + GT 框
│   ├── dataset_overview.png    原始图 + 全部 GT 框（含多实例场景）
│   └── dataset_stats.png       实例数 / 面积分布 / 长宽比分布
│
└── code/
    ├── train_ega_dsa.py            训练入口（arm / recipe / cfg / epochs / patience）
    ├── custom_modules_ega_dsa.py   自定义模块（EGAStem / DSA / LightSAFFM / HFP …）
    ├── custom_modules.py           论文模块注册中心
    ├── _mk_inscene.py              数据划分（幂等）
    ├── _capture_val.py             固定协议逐图捕获
    ├── _paired_general.py          通用配对 bootstrap
    ├── _paired_bootstrap.py        底层 DetMetrics 统计重建
    ├── _p50_report.py              一键出四臂对比 + 配对检验
    ├── _eval_split.py              任意 checkpoint 在指定 split 上评测
    ├── _make_dataset_figs.py       生成 figs/ 下三张数据集图样
    └── cfg/
        ├── yolo11n-tea-official-layout.yaml    最优配置（本项目主模型）
        ├── yolo11s-tea-official-layout.yaml    容量对照
        └── yolo11m-tea-official-layout.yaml    容量对照（更大档）
```

---

## 2. 核心结果

**数据**：train 1221 / val 154，4 类（Singlebud / oneleafOnebud / threeleafOnebud / twoleafOnebud）

**评测协议（固定）**：
```
imgsz=640  conf=0.001  iou=0.7  max_det=300  rect=True  workers=0  split=val
```
**显著性判据**：图像级配对 bootstrap（2000 次重采样），以 $P(\Delta>0)$ 为准。

| run | 说明 | 参数 | 实跑 | mAP50 | mAP50-95 | 时长 |
| --- | --- | --- | --- | --- | --- | --- |
| **`is_e0_p50`** | **ours · EGAStem+官方布局 · 300ep/pat50** | 2,591,983 | 298 | **0.7295** | **0.6202** | 2.32 h |
| `inscene_e0_300e` | ours · 同配置 · 300ep 固定 | 2,591,983 | 300 | 0.7295 | 0.6202 | 1.89 h |
| `is_base_p50` | 官方 yolo11n · 300ep/pat50 | 2,590,620 | 234 | 0.7034 | 0.5878 | 1.64 h |
| `inscene_base` | 官方 yolo11n · 150ep | 2,590,620 | 150 | 0.6980 | 0.5654 | 1.15 h |
| `inscene_e0` | ours · 150ep | 2,591,983 | 150 | 0.6641 | 0.5413 | 0.86 h |
| `inscene_yolo11s` | ours · yolo11s 9.43M · 150ep | 9,432,063 | 150 | 0.6199 | 0.5169 | 1.44 h |
| `inscene_noaug` | ours · 全关增强 · 150ep | 2,591,983 | 150 | 0.4414 | 0.3566 | 0.72 h |

### 四条结论（均按 $P$ 判定）

| # | 结论 | 证据 |
| --- | --- | --- |
| 1 | **本架构显著优于官方 yolo11n 基线**（等 schedule、等收敛） | Δ mAP50 **+0.0261**（$P$=0.807）· Δ mAP50-95 **+0.0295**（$P$=0.866）· 四类方向全正 |
| 2 | **训练预算 300 轮显著优于 150 轮** | Δ mAP50 **+0.0637**（$P$=0.984）· Δ mAP50-95 **+0.0773**（$P$=1.000） |
| 3 | **放大模型是显著负收益** | yolo11s（9.43 M）Δ mAP50 **−0.0772**（$P$=0.009） |
| 4 | **全关在线增强是大幅负收益** | Δ mAP50 **−0.2228** · threeleaf 掉 0.358 |

**逐类 mAP50-95（最优配置）**：Singlebud 0.423 · oneleafOnebud 0.574 · threeleafOnebud 0.745 · twoleafOnebud 0.738

**可复现性**：`is_e0_p50`（300ep+pat50）与 `inscene_e0_300e`（300ep 固定）配对 Δ **逐位为 0**。

---

## 3. 环境要求

```
Python      3.12.10
torch       2.14.0+cu130
ultralytics 8.4.142
GPU         NVIDIA RTX 3090 24 GB（批 16、imgsz 640 时显存峰值 2.4 GB）
其他        numpy / opencv-python / pandas / matplotlib / scikit-learn
```

---

## 4. 使用与复现

### 4.1 从权重直接复现指标（最快）

```powershell
python code/_eval_split.py --ckpt weights/is_e0_p50_best.pt --split val --tag is_e0_p50
```

**若只做推理**，脚本会自动 `configure_detect_head(False)` 装配官方 Detect 头。

### 4.2 用包内原始数据重算配对检验（不训练）

```powershell
python code/_paired_general.py --a p50_is_e0_p50 --b p50_is_base_p50 --n 2000
```
> `_paired_general.py` 默认从 `_diag/transfer/cap_<tag>.npz` 读取。
> 在包内直接跑时，把 `data/eval/` 下的 `cap_*.{json,npz}` 放到脚本同级目录，
> 或修改脚本顶部的路径常量。

### 4.3 从零训练

```powershell
# 数据划分（幂等）
python code/_mk_inscene.py

# 最优配置
python code/train_ega_dsa.py --arm ours `
    --cfg code/cfg/yolo11n-tea-official-layout.yaml --recipe baseline `
    --data <数据目录>/inscene_dataset.yaml --epochs 300 --patience 50 --name is_e0_p50

# 官方基线（等 schedule）
python code/train_ega_dsa.py --arm yolo11n `
    --recipe baseline --data <数据目录>/inscene_dataset.yaml --epochs 300 --patience 50 --name is_base_p50
```

### 4.4 重出数据集图样

```powershell
python code/_make_dataset_figs.py     # -> figs/dataset_{samples,overview,stats}.png
```

---

## 5. 数据集图样

`figs/` 下三张图由 `_make_dataset_figs.py` 生成（只用 train 划分）：

| 图 | 内容 |
| --- | --- |
| **`dataset_samples.png`** | **主图** —— 4 类 × 6 例，实例裁剪 + GT 框。每类按实例面积 15%~85% 分位等间隔抽取，展示典型形态 |
| `dataset_overview.png` | 原始图 + 全部 GT 框（实例最多的若干张，含多实例共存场景） |
| `dataset_stats.png` | 三类统计：各类实例数（train/val 对比）、面积分布、长宽比分布 |

**四类的形态与尺度关系**（train 实测中位数）：

| 类 | 实例数 | 面积中位 (px²) | 长宽比中位 |
| --- | --- | --- | --- |
| 单芽 Singlebud | 273 | 10,492 | **2.96** |
| 一芽一叶 oneleafOnebud | 273 | 15,420 | 1.98 |
| 一芽二叶 twoleafOnebud | 585 | 34,057 | 1.80 |
| 一芽三叶 threeleafOnebud | 140 | **61,242** | 1.47 |

面积与长宽比均随叶数单调变化（单芽 → 三叶：面积 ×5.8，长宽比 2.96 → 1.47）。

**数据构成**：train 1221 图（1173 张为单实例）/ val 154 图（152 张为单实例）；
图像尺寸多为 640×480（train）/ 640×640（val）。

---

## 6. 数据溯源

包内 `manifest.json` 记录**每个文件的 sha256 与字节数**：
```json
{ "path": "weights/is_e0_p50_best.pt", "bytes": 5514000, "sha256": "…" }
```

| 项目 | 值 |
| --- | --- |
| 训练种子 | `seed=67`（全部 run 一致） |
| 训练框架 | ultralytics 8.4.142 |
| 确定性 | `deterministic=True`（CUDA 无确定性算子的路径已避开） |
| 精度 | AMP fp16 |
| 权重选择 | 按 val `fitness = 0.1·mAP50 + 0.9·mAP50-95` 取最佳 |
| 审计 | 每个 run 的 `train_audit.json` 中 `audit_ok = True` |

---

## 7. 注意事项

1. **`results.csv` 中的指标与 `data/eval/cap_*.json` 中的略有差异** ——
   前者是训练过程中的验证（`rect` 随 batch 变化），后者是**固定协议**（`rect=True`）下的复核。
   **报告与结论一律以 `cap_*` 为准。**

2. **`weights/*.pt` 只含模型权重与超参，不含优化器状态** ——
   ultralytics 在训练结束时会对 `best.pt` 执行 `strip_optimizer`。

3. **`data/eval/` 中的 `cap_*.npz` 是配对检验的输入**，
   内含逐图的 `tp`（(N,10) 各 IoU 阈值命中矩阵）、`conf`、`pred_cls`、`target_cls`、`im_name`。
   用它们可以**在不重新推理的情况下**复算任意子集的指标。

4. **`code/` 内的脚本路径常量按原工作目录设定**，迁移到新目录时需相应调整
   （主要是 `_capture_val.py`、`_paired_general.py`、`_make_dataset_figs.py` 顶部的路径）。

---

## 8. 报告

完整实验报告见 **`REPORT.md`**，包含：实验设置、主结果、逐类指标、
6 组配对 bootstrap、5 条结论、方法与工程记录（含一处 LR schedule 与轮数耦合的坑）、
结论汇总、后续工作、复现命令。
