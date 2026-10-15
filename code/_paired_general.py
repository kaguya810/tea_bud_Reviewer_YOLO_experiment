# -*- coding: utf-8 -*-
"""通用配对 bootstrap：任意两个 tag 的捕获文件。

复用 `_diag/transfer/_paired_bootstrap.py` 里已经验证过的
`DetMetrics` 统计重建逻辑（`make_stats` / `evaluate` / `summarise`），
只把 tag 参数化、并把「谁减谁」的方向写明确。

用法：
    python _paired_general.py --a e0 --b baseline --n 3000
判读：delta = a - b；CI 不含 0 才叫显著。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "_diag" / "transfer"))

import numpy as np  # noqa: E402

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("pb", ROOT / "_diag" / "transfer" / "_paired_bootstrap.py")
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    za, ma = pb.load(args.a)
    zb, mb = pb.load(args.b)
    na = [str(x) for x in za["im_name"]]
    nb = [str(x) for x in zb["im_name"]]
    assert na == nb, "两臂的图顺序不同，无法配对"
    assert ma["protocol"] == mb["protocol"], (ma["protocol"], mb["protocol"])
    names = ma["names"]
    n = len(na)

    print("=" * 88)
    print(f"配对 bootstrap：  A={args.a}  vs  B={args.b}      delta = A - B")
    print(f"  协议 {ma['protocol']}   图数 {n}   重采样 {args.n}")
    print(f"  A ckpt = {ma['ckpt']}")
    print(f"  B ckpt = {mb['ckpt']}")
    print("=" * 88)

    idx_all = np.arange(n)
    ea = pb.evaluate(za, idx_all, names)
    eb = pb.evaluate(zb, idx_all, names)
    for k in ("mAP50", "mAP50_95"):
        print(f"  {k:10s}  A={ea[k]:.4f}  B={eb[k]:.4f}  Δ={ea[k]-eb[k]:+.4f}")

    rng = np.random.default_rng(args.seed)
    D = {"mAP50": [], "mAP50_95": []}
    Sa, Sb = {"mAP50": [], "mAP50_95": []}, {"mAP50": [], "mAP50_95": []}
    Dcls = {c: [] for c in range(len(names))}
    for _ in range(args.n):
        idx = rng.integers(0, n, n)
        a = pb.evaluate(za, idx, names)
        b = pb.evaluate(zb, idx, names)
        for k in D:
            D[k].append(a[k] - b[k])
            Sa[k].append(a[k]); Sb[k].append(b[k])
        for c in range(len(names)):
            if c in a["ap"] and c in b["ap"]:
                Dcls[c].append(a["ap"][c] - b["ap"][c])

    print()
    out = {}
    for k in D:
        s = pb.summarise(D[k])
        s["P_delta_gt_0"] = float(np.mean(np.asarray(D[k]) > 0))
        sig = "显著" if (s["ci95"][0] > 0 or s["ci95"][1] < 0) else "**在噪声内**"
        out[k] = s
        print(f"  {k:10s} Δ={s['delta_mean']:+.4f}  95%CI=[{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}]"
              f"  sd={s['delta_sd']:.4f}  P(Δ>0)={s['P_delta_gt_0']:.3f}  -> {sig}")
        print(f"      未配对 sd（对照）：A={np.std(Sa[k],ddof=1):.4f}  B={np.std(Sb[k],ddof=1):.4f}"
              f"  配对后 sd={s['delta_sd']:.4f}  配对收益={np.std(Sa[k],ddof=1)/max(s['delta_sd'],1e-9):.2f}×")

    print()
    print("  逐类 mAP50-95（n_val 极少，宽 CI 是正常的）：")
    gtc = {names[c]: int(np.sum([np.sum(np.atleast_1d(x) == c) for x in za["target_cls"]]))
           for c in range(len(names))}
    out["per_class"] = {}
    for c in range(len(names)):
        if not Dcls[c]:
            continue
        s = pb.summarise(Dcls[c])
        s["P_delta_gt_0"] = float(np.mean(np.asarray(Dcls[c]) > 0))
        sig = "显著" if (s["ci95"][0] > 0 or s["ci95"][1] < 0) else "在噪声内"
        out["per_class"][names[c]] = {"n_val": gtc[names[c]], **s}
        print(f"    {names[c]:18s} n={gtc[names[c]]:>3}  A={ea['ap'].get(c, float('nan')):.4f}"
              f"  B={eb['ap'].get(c, float('nan')):.4f}  Δ={s['delta_mean']:+.4f}"
              f"  CI=[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]  -> {sig}")

    o = ROOT / f"_diag/transfer/paired_{args.a}_vs_{args.b}.json"
    o.write_text(json.dumps({"a": args.a, "b": args.b, "protocol": ma["protocol"],
                             "n_images": n, "n_boot": args.n, "point": {"a": ea, "b": eb},
                             **out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  已写出 {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
