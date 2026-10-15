# -*- coding: utf-8 -*-
"""四臂训练结束后的一键对比：读出所有 run，做配对 bootstrap，出一张表。

用法：python _p50_report.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

# (run 名, 协议, 臂, 参数量)
RUNS = [
    ("sd_base_p50", "场景不相交", "官方 yolo11n(300ep/pat50)", "2.59M"),
    ("sd_e0_p50", "场景不相交", "ours EGAStem(300ep/pat50)", "2.59M"),
    ("is_base_p50", "随机划分", "官方 yolo11n(300ep/pat50)", "2.59M"),
    ("is_e0_p50", "随机划分", "ours EGAStem(300ep/pat50)", "2.59M"),
    # 历史对照
    ("e0_official_aug", "场景不相交", "ours EGAStem(150轮)", "2.59M"),
    ("inscene_base", "随机划分", "官方 yolo11n(150轮)", "2.59M"),
    ("inscene_e0", "随机划分", "ours EGAStem(150轮)", "2.59M"),
    ("inscene_e0_300e", "随机划分", "ours EGAStem(300轮)", "2.59M"),
]

DATA = {"场景不相交": "_eval154_dataset.yaml", "随机划分": "inscene_dataset.yaml"}


def read_run(n: str):
    p = ROOT / "runs" / n / "results.csv"
    if not p.exists():
        return None
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(p.open(encoding="utf-8"))]
    if not rows:
        return None
    k50, k95 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"
    fit = lambda r: 0.1 * float(r[k50]) + 0.9 * float(r[k95])  # noqa: E731
    best = max(rows, key=fit)
    return {"epochs": len(rows), "best_ep": int(best["epoch"]),
            "mAP50": float(best[k50]), "mAP50_95": float(best[k95]),
            "fitness": fit(best)}


def main() -> int:
    print("=" * 96)
    print("epochs=300 / patience=50 四臂对比")
    print("=" * 96)
    print(f"{'run':<22}{'协议':<12}{'臂':<24}{'轮数':>6}{'mAP50':>9}{'mAP50-95':>10}{'@ep':>6}")
    print("-" * 96)
    got = {}
    for n, proto, arm, par in RUNS:
        d = read_run(n)
        if d is None:
            print(f"{n:<22}{proto:<12}{arm:<24}{'未完成/不存在':>6}")
            continue
        got[n] = (d, proto, arm)
        stop = "  ← 早停" if d["epochs"] < 300 and d["best_ep"] < d["epochs"] - 8 else ""
        print(f"{n:<22}{proto:<12}{arm:<24}{d['epochs']:>6}{d['mAP50']:>9.4f}"
              f"{d['mAP50_95']:>10.4f}{d['best_ep']:>6}{stop}")

    # ---- 配对检验 ----
    pairs = [
        ("sd_e0_p50", "e0_official_aug", "场景不相交：ours 300轮 vs 150轮  ★预算效应"),
        ("sd_e0_p50", "sd_base_p50", "场景不相交：ours300 vs 基线300  ★架构"),
        ("sd_base_p50", "e0_official_aug", "场景不相交：基线300 vs ours150"),
        ("is_e0_p50", "inscene_e0", "随机划分：ours 300轮 vs 150轮  ★预算效应"),
        ("is_e0_p50", "is_base_p50", "随机划分：ours300 vs 基线300  ★架构"),
        ("is_e0_p50", "inscene_e0_300e", "随机划分：ours 300ep+pat50 vs 300ep 固定"),
    ]
    print()
    print("=" * 96)
    print("配对 bootstrap（需先捕获逐图统计；缺捕获的会自动跳过）")
    print("=" * 96)
    for a, b, desc in pairs:
        if a not in got or b not in got:
            print(f"  {desc:<44} 跳过（run 未完成）")
            continue
        # ★ 缓存跳过：配对 JSON 已存在就直接读，不重跑（每组 ~20 min）
        done = ROOT / f"_diag/transfer/paired_p50_{a}_vs_p50_{b}.json"
        if done.exists():
            d = json.loads(done.read_text(encoding="utf-8"))
            print(f"  {desc}   [已缓存]")
            for key in ("mAP50", "mAP50_95"):
                s_ = d.get(key)
                if not s_:
                    continue
                sig = ("★显著" if (s_["ci95"][0] > 0 or s_["ci95"][1] < 0) else "在噪声内")
                print(f"      {key:9s} Δ={s_['delta_mean']:+.4f}  "
                      f"95%CI=[{s_['ci95'][0]:+.4f},{s_['ci95'][1]:+.4f}]  "
                      f"sd={s_['delta_sd']:.4f}  P(Δ>0)={s_.get('P_delta_gt_0', float('nan')):.3f}"
                      f"  -> {sig}")
            continue
        proto = got[a][1]
        data = DATA[proto]
        cap_a = ROOT / f"_diag/transfer/cap_p50_{a}.npz"
        cap_b = ROOT / f"_diag/transfer/cap_p50_{b}.npz"
        for run, cap in ((a, cap_a), (b, cap_b)):
            if not cap.exists():  # ★ 已有捕获就跳过（capture 需跑 GPU，约 1 min）
                ck = ROOT / "runs" / run / "weights" / "best.pt"
                if ck.exists():
                    subprocess.run([PY, "_diag/transfer/_capture_val.py", "--ckpt",
                                    str(ck.relative_to(ROOT)), "--tag", f"p50_{run}",
                                    "--out", str(cap.with_suffix("")), "--data", data],
                                   cwd=str(ROOT), capture_output=True)
        if not (cap_a.exists() and cap_b.exists()):
            print(f"  {desc:<44} 捕获失败")
            continue
        r = subprocess.run([PY, "_paired_general.py", "--a", f"p50_{a}", "--b", f"p50_{b}",
                            "--n", "2000"], cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        line50 = line95 = ""
        for ln in (r.stdout or "").splitlines():
            if ln.strip().startswith("mAP50 "):
                line50 = ln.strip()
            if ln.strip().startswith("mAP50_95 "):
                line95 = ln.strip()
        sig = "显著" if ("显著" in line50 and "噪声" not in line50) else "在噪声内"
        print(f"  {desc}")
        print(f"      {line50}")
        print(f"      {line95}")
        print(f"      -> {sig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
