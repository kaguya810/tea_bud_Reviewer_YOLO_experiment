# -*- coding: utf-8 -*-
"""按图随机切分，构造检测用的训练/验证集。

约束：
    规模与既有的 v2k 池一致：**1221 训练 / 154 验证**（合计 1375）
    切分单位是「图」；分层依据是每张图的首个类别，避免类别分布被切歪。
    随机源固定为 SEED，重建结果可复现。

依据：文件名 `1-NNN-_jpg.rf.<hash>` 里的 `NNN` 是源片编号，全池共 570 个。

产出：`yolo_detect_4cls_inscene/` + `inscene_dataset.yaml` + `SPLIT_REPORT.json`
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "yolo_detect_4cls_v2k"
DST = ROOT / "yolo_detect_4cls_inscene"
PAT = re.compile(r"^1-(\d+)-_jpg\.rf\.")
N_TRAIN, N_VAL = 1221, 154
SEED = 67


def source_id(name: str) -> int:
    m = PAT.match(name)
    if not m:
        raise ValueError(f"文件名不匹配: {name}")
    return int(m.group(1))


def classes_of(label: pathlib.Path, nc: int = 4) -> set[int]:
    if not label.exists():
        return set()
    out = set()
    for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
        t = line.split()
        if len(t) >= 5:
            out.add(int(float(t[0])))
    return out


def main() -> int:
    # 收集全部 1375 张（三个 split 合并）
    items = []
    for sp in ("train", "valid", "test"):
        for f in sorted((SRC / sp / "images").iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            lab = SRC / sp / "labels" / (f.stem + ".txt")
            items.append({"name": f.name, "src": f, "label": lab,
                          "source": source_id(f.name), "orig_split": sp,
                          "classes": classes_of(lab)})
    print(f"  收集 {len(items)} 张图 / {len({it['source'] for it in items})} 张源片")

    # 分层：按「首个类别」做分层随机切分，避免类别分布被切歪
    rng = random.Random(SEED)
    by_cls = collections.defaultdict(list)
    for it in items:
        key = min(it["classes"]) if it["classes"] else -1
        by_cls[key].append(it)
    for k in by_cls:
        rng.shuffle(by_cls[k])

    n_train_total, n_val_total = N_TRAIN, N_VAL
    train, val = [], []
    total = len(items)
    for k in sorted(by_cls):
        grp = by_cls[k]
        # 按比例分配该类的 train/val 名额
        n_val_k = round(len(grp) * n_val_total / total)
        val += grp[:n_val_k]
        train += grp[n_val_k:]
    # 名额微调（舍入误差）
    rng.shuffle(train)
    while len(train) > n_train_total:
        val.append(train.pop())
    while len(val) > n_val_total:
        train.append(val.pop())
    print(f"  切分后: train {len(train)} / val {len(val)}")

    # 写盘
    if DST.exists():
        shutil.rmtree(DST)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (DST / sub).mkdir(parents=True)
    for it in train:
        shutil.copy2(it["src"], DST / "images" / "train" / it["name"])
        if it["label"].exists():
            shutil.copy2(it["label"], DST / "labels" / "train" / it["label"].name)
    for it in val:
        shutil.copy2(it["src"], DST / "images" / "val" / it["name"])
        if it["label"].exists():
            shutil.copy2(it["label"], DST / "labels" / "val" / it["label"].name)

    y = ROOT / "inscene_dataset.yaml"
    y.write_text(
        "# 由 _mk_inscene.py 生成：按图随机切分，切分单位是「图」\n"
        f"path: {DST}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/val\n"
        "\n"
        "nc: 4\n"
        "names:\n"
        "  0: Singlebud\n"
        "  1: oneleafOnebud\n"
        "  2: threeleafOnebud\n"
        "  3: twoleafOnebud\n",
        encoding="utf-8")

    rep = {
        "seed": SEED, "n_train": len(train), "n_val": len(val),
        "n_sources_total": len({it["source"] for it in items}),
        "class_counts_train": dict(sorted(collections.Counter(
            c for it in train for c in it["classes"]).items())),
        "class_counts_val": dict(sorted(collections.Counter(
            c for it in val for c in it["classes"]).items())),
        "note": "按图随机切分；分层依据是每张图的首个类别",
    }
    (ROOT / "SPLIT_REPORT.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  类别分布 train={rep['class_counts_train']}")
    print(f"  类别分布 val  ={rep['class_counts_val']}")
    print(f"  已写出 {DST.relative_to(ROOT)} / inscene_dataset.yaml / SPLIT_REPORT.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
