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

- 本仓库是论文实验记录；论文中的数字与最终表述以论文正文为准，`REPORT.md` 提供工程复核日志
- 最优配置：`is_e0_p50`（EGAStem + 官方布局 + 标准增强 + 300 轮）
- 最优指标：`mAP50 = 0.7295`，`mAP50-95 = 0.6202`
- 入口模块（EGAStem）在 300 轮下取得九个配置中**最高的点估计**（mAP50 0.7295 vs 官方 0.7034），
  但其效应量低于本设计的可分辨量级，95 % 区间包含零，**未获确立**；150 轮下方向相反
- 300 轮训练优于 150 轮训练，差值的 95 % 区间排除零（+0.0634，CI +0.006 … +0.119），已确立
- 放大到 `yolo11s` 未获确立的收益；严格控制的容量对照（n vs s，150 轮）区间含零
- 数据集 1,375 图 / 1,429 个标注框；训练 1,221 图 / 1,271 框，验证 154 图 / 158 框

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
  --data data/split/inscene_dataset.yaml --epochs 300 --patience 50 --name is_e0_p50
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
