# 茶芽采摘形态多分类检测实验

本仓库用于复现实验结论与训练流程，核心目标是比较不同训练预算、模型配置与增强策略在 4 类茶芽检测任务上的效果。

## 项目结构

```text
tea_bud_Reviewer_YOLO_experiment/
├── README.md
├── REPORT.md
├── code/                 # 训练、评测、统计脚本
├── figs/                 # 数据集可视化图
├── weights/              # 已训练权重
├── manifest.json         # 文件校验信息
└── data/                 # 实验数据（默认忽略，不纳入版本控制）
```

## 关键结论

- 最优配置：`is_e0_p50`（EGAStem + 官方布局 + 标准增强 + 300 轮）
- 最优指标：`mAP50 = 0.7295`，`mAP50-95 = 0.6202`
- 与官方 `yolo11n` 基线（同 300 轮）相比有稳定提升
- 300 轮训练显著优于 150 轮训练
- 在本任务数据规模下，放大到 `yolo11s` 为负收益

详细实验过程、统计检验与工程记录见：[`REPORT.md`](./REPORT.md)

## 环境要求

- Python 3.12.10
- torch 2.14.0+cu130
- ultralytics 8.4.142
- numpy / opencv-python / pandas / matplotlib / scikit-learn
- GPU：RTX 3090 24GB（或同等级）

## 常用命令

### 1) 从权重直接评测

```bash
python code/_eval_split.py --ckpt weights/is_e0_p50_best.pt --split val --tag is_e0_p50
```

### 2) 重算配对 bootstrap（不训练）

```bash
python code/_paired_general.py --a p50_is_e0_p50 --b p50_is_base_p50 --n 2000
```

### 3) 从零训练（最优配置）

```bash
python code/train_ega_dsa.py --arm ours \
  --cfg code/cfg/yolo11n-tea-official-layout.yaml --recipe baseline \
  --data <数据目录>/inscene_dataset.yaml --epochs 300 --patience 50 --name is_e0_p50
```

## 数据文件说明

为便于发布代码仓库，`data/` 已配置为默认忽略目录：

- 本地可保留完整实验数据用于复现
- Git 提交默认不再包含 `data/` 内的数据文件
- 如需共享数据，请使用独立数据包或对象存储链接

## 复现建议

1. 先运行 `code/_mk_inscene.py` 生成/确认划分
2. 使用固定评测协议对比不同 run
3. 结果分析以 `REPORT.md` 中的配对检验结论为准
