# -*- coding: utf-8 -*-
"""通用 train-split 评估：量化「记忆能力 vs 泛化能力」的差距。

为什么用 train-split mAP 而不是 loss：
  * loss 的口径（每层样本归一化、anchor 数、TAL 对齐质量）在两臂之间不可比
  * mAP 是纯函数空间量，没有这些口径问题
  * **train-split mAP 是"这个架构在熟悉数据上能达到什么水平"的上界**

这个数字是判断「0.7 目标需要什么」的关键：
  若 train-split mAP 也只有 0.3，那是**能力不足**（架构/优化问题）；
  若 train-split 很高（>0.8）而 val 很低，那是**泛化问题**（数据/分布问题）——
  两者的处方完全不同。

用法：python _eval_split.py --ckpt <path> --split train
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
DATA = "experiments/baseline_e5/tea_bud_4cls_v2k_detect.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ck = pathlib.Path(args.ckpt)
    if not ck.is_absolute():
        ck = ROOT / ck
    if not ck.exists():
        print(f"缺 {ck}")
        return 2

    sys.path.insert(0, str(ROOT))
    import custom_modules_ega_dsa as M
    M.configure_detect_head(False)   # 官方布局用官方 Detect

    from ultralytics import YOLO
    m = YOLO(str(ck))
    r = m.val(data=str(ROOT / DATA), split=args.split, imgsz=args.imgsz, batch=args.batch,
              device=0, workers=0, plots=False, verbose=False, conf=0.001, iou=0.7)
    b = r.box
    d = {
        "ckpt": str(ck.relative_to(ROOT)) if ck.is_relative_to(ROOT) else str(ck),
        "split": args.split, "imgsz": args.imgsz,
        "mAP50": float(b.map50), "mAP50-95": float(b.map),
        "P": float(b.mp), "R": float(b.mr),
        "per_class": {n: float(v) for n, v in zip(r.names.values(), b.maps)},
        "n_instances": int(len(r.box.ap_class_index)),
    }
    tag = args.tag or ck.parent.parent.name
    print(f"  [{tag}] split={args.split}  mAP50={d['mAP50']:.4f}  mAP50-95={d['mAP50-95']:.4f}"
          f"  P={d['P']:.4f}  R={d['R']:.4f}")
    for n, v in d["per_class"].items():
        print(f"      {n:18s} {v:.4f}")
    o = ROOT / "_diag" / f"split_{args.split}_{tag}.json"
    o.parent.mkdir(exist_ok=True)
    o.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已写出 {o.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
