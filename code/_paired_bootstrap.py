# -*- coding: utf-8 -*-
"""Paired image-level bootstrap: is ours vs baseline a real difference?

Both models are evaluated on the SAME 79 val images in the SAME order. The correct
uncertainty for "which model is better" is the distribution of the **difference** under
image-level resampling: resample image indices ONCE per replicate, recompute BOTH models'
AP on that same resample, and look at delta = baseline - ours.

Pairing removes the "is this a hard set of images" variance, which is the dominant term in
the single-model bootstrap.

Outputs per metric (overall and per class):
    delta_mean, delta_sd, 95% CI, P(delta<=0)  (= one-sided p that ours is NOT worse)
    plus the single-model (unpaired) sd for comparison.

Usage:
    python _diag/transfer/_paired_bootstrap.py --n 2000
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(r"D:\CodeWorkspace\DL_teabud-seg")
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from ultralytics.utils.metrics import DetMetrics


def load(tag):
    z = np.load(str(ROOT / f"_diag/transfer/cap_{tag}.npz"), allow_pickle=True)
    meta = json.loads((ROOT / f"_diag/transfer/cap_{tag}.json").read_text(encoding="utf-8"))
    return z, meta


def make_stats(z, idx):
    """Build a DetMetrics-compatible stats dict from resampled image indices."""
    return {
        "tp": [z["tp"][i] for i in idx],
        "conf": [z["conf"][i] for i in idx],
        "pred_cls": [z["pred_cls"][i] for i in idx],
        "target_cls": [np.atleast_1d(z["target_cls"][i]) for i in idx],
        # nb: only used for np.bincount(...) inside process(); must be a list of 1-D arrays
        "target_img": [np.atleast_1d(np.unique(z["target_cls"][i])) for i in idx],
    }


def evaluate(z, idx, names):
    m = DetMetrics(names=names)
    m.stats = make_stats(z, idx)
    m.process(plot=False)
    return {
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "ap50": {int(c): float(m.box.ap50[j]) for j, c in enumerate(m.box.ap_class_index)},
        "ap": {int(c): float(m.box.ap[j]) for j, c in enumerate(m.box.ap_class_index)},
    }


def summarise(delta):
    a = np.asarray(delta, dtype=float)
    return {
        "delta_mean": float(a.mean()),
        "delta_sd": float(a.std(ddof=1)),
        "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
        "P_delta_le_0": float((a <= 0).mean()),   # ours is better or equal
        "P_delta_lt_0": float((a < 0).mean()),
        "n": int(a.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    zo, mo = load("ours")
    zb, mb = load("baseline")
    no = [str(x) for x in zo["im_name"]]
    nb = [str(x) for x in zb["im_name"]]
    assert no == nb, "image order differs between the two captures -- cannot pair"
    assert mo["protocol"] == mb["protocol"], (mo["protocol"], mb["protocol"])
    names = mo["names"]
    n_img = len(no)

    # point estimates (full set)
    idx_all = np.arange(n_img)
    eo = evaluate(zo, idx_all, names)
    eb = evaluate(zb, idx_all, names)
    point = {k: {"ours": eo[k], "baseline": eb[k], "delta": eb[k] - eo[k]}
             for k in ("mAP50", "mAP50_95")}

    rng = np.random.default_rng(args.seed)
    D = {"mAP50": [], "mAP50_95": []}
    Dcls = {c: {"ap50": [], "ap": []} for c in range(len(names))}
    # also keep single-model samples, to show how much pairing helps
    S = {"ours": {"mAP50": [], "mAP50_95": []}, "baseline": {"mAP50": [], "mAP50_95": []}}

    for _ in range(args.n):
        idx = rng.integers(0, n_img, n_img)
        o = evaluate(zo, idx, names)
        b = evaluate(zb, idx, names)
        for k in ("mAP50", "mAP50_95"):
            D[k].append(b[k] - o[k])
            S["ours"][k].append(o[k])
            S["baseline"][k].append(b[k])
        for c in range(len(names)):
            if c in o["ap"] and c in b["ap"]:
                Dcls[c]["ap50"].append(b["ap50"][c] - o["ap50"][c])
                Dcls[c]["ap"].append(b["ap"][c] - o["ap"][c])

    out = {
        "protocol": mo["protocol"],
        "n_images": n_img,
        "n_boot": args.n,
        "names": names,
        "val_gt_per_class": {names[c]: int(np.sum([np.sum(np.atleast_1d(x) == c) for x in zo["target_cls"]]))
                             for c in range(len(names))},
        "point_estimate": point,
        "paired_delta": {k: summarise(D[k]) for k in D},
        "unpaired_sd_for_comparison": {
            "ours": {k: float(np.std(S["ours"][k], ddof=1)) for k in S["ours"]},
            "baseline": {k: float(np.std(S["baseline"][k], ddof=1)) for k in S["baseline"]},
            "paired_delta": {k: float(np.std(D[k], ddof=1)) for k in D},
        },
        "per_class": {},
    }
    for c in range(len(names)):
        if not Dcls[c]["ap"]:
            continue
        out["per_class"][names[c]] = {
            "n_val_instances": out["val_gt_per_class"][names[c]],
            "ours_ap50": eo["ap50"].get(c), "baseline_ap50": eb["ap50"].get(c),
            "delta_ap50": summarise(Dcls[c]["ap50"]),
            "ours_ap50_95": eo["ap"].get(c), "baseline_ap50_95": eb["ap"].get(c),
            "delta_ap50_95": summarise(Dcls[c]["ap"]),
        }

    o = ROOT / "_diag/transfer/paired_bootstrap.json"
    o.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[ok] {o}")


if __name__ == "__main__":
    main()
