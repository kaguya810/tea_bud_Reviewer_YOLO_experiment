# -*- coding: utf-8 -*-
"""Capture per-image detection statistics for one checkpoint, for a paired bootstrap.

Writes <out>.npz with, per validation image (in dataloader order):
    tp (N,10) float, conf (N,), pred_cls (N,), target_cls (M,), im_name
plus a small json with the scalar metrics and the exact protocol used.

Protocol is pinned explicitly so two models are always evaluated identically:
    imgsz=640, conf=0.001, iou=0.7, max_det=300, rect=True, workers=0, split=val

Usage:
    python code/_capture_val.py --ckpt <path> --tag ours --out data/eval/cap_ours
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.checks import check_imgsz
from ultralytics.data.utils import check_det_dataset

PROTOCOL = dict(imgsz=640, conf=0.001, iou=0.7, max_det=300, rect=True, workers=0, split="val")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/split/inscene_dataset.yaml")
    args = ap.parse_args()

    overrides = dict(model=None, data=str(ROOT / args.data), split="val", imgsz=PROTOCOL["imgsz"],
                     conf=PROTOCOL["conf"], iou=PROTOCOL["iou"], max_det=PROTOCOL["max_det"],
                     rect=PROTOCOL["rect"], workers=PROTOCOL["workers"], plots=False,
                     save_json=False, save_txt=False, verbose=False, single_cls=False,
                     task="detect", mode="val")
    val = DetectionValidator(args=overrides)

    om = YOLO(str(ROOT / args.ckpt))
    model = om.model
    ci = check_imgsz(PROTOCOL["imgsz"], stride=model.stride, min_dim=2)
    val.args.imgsz = int(ci[0]) if isinstance(ci, (list, tuple)) else int(ci)
    val.imgsz = val.args.imgsz
    val.stride = int(model.stride.max())
    val.data = check_det_dataset(val.args.data, autodownload=False)
    val.device = select_device(val.args.device, 1)
    model.to(val.device).float()
    val.dataloader = val.get_dataloader(val.data.get("val") or val.data["train"], 1)
    val.training = False
    val.init_metrics(model)
    model.eval()

    names = []
    with torch.no_grad():
        for batch in val.dataloader:
            names += [Path(f).name for f in batch["im_file"]]
            batch = val.preprocess(batch)
            preds = val.postprocess(model(batch["img"]))
            val.update_metrics(preds, batch)
    val.metrics.process(plot=False)

    st = {k: float(v) for k, v in val.metrics.results_dict.items()}
    n = len(names)
    tp = [np.atleast_1d(np.asarray(x)) for x in val.metrics.stats["tp"]]
    conf = [np.atleast_1d(np.asarray(x)) for x in val.metrics.stats["conf"]]
    pcls = [np.atleast_1d(np.asarray(x)) for x in val.metrics.stats["pred_cls"]]
    tcls = [np.atleast_1d(np.asarray(x)) for x in val.metrics.stats["target_cls"]]
    assert len(tp) == len(conf) == len(pcls) == len(tcls) == n, (len(tp), n)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out) + ".npz",
        im_name=np.array(names),
        tp=np.array(tp, dtype=object), conf=np.array(conf, dtype=object),
        pred_cls=np.array(pcls, dtype=object), target_cls=np.array(tcls, dtype=object),
    )
    meta = {"tag": args.tag, "ckpt": args.ckpt, "protocol": PROTOCOL, "n_images": n,
            "names": list(om.names.values()) if isinstance(om.names, dict) else list(om.names),
            "stats": st,
            "per_class": {om.names[int(c)]: {"ap50": float(val.metrics.box.ap50[j]),
                                             "ap50_95": float(val.metrics.box.ap[j])}
                          for j, c in enumerate(val.metrics.ap_class_index)}}
    (Path(str(out) + ".json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[ok] {out}.npz")


if __name__ == "__main__":
    main()
