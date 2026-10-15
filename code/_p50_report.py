# -*- coding: utf-8 -*-
"""读取已完成 run 的结果并输出简要对比。"""
from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys

CODE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
PY = sys.executable

RUNS = [
    ("is_base_p50", "official stem, 300ep/pat50", "2.59M"),
    ("is_e0_p50", "EGAStem, 300ep/pat50", "2.59M"),
    ("is_v5n_p50", "YOLOv5nu, 300ep/pat50", "1.89M"),
    ("is_v8n_p50", "YOLOv8n, 300ep/pat50", "3.01M"),
    ("inscene_base", "official stem, 150ep", "2.59M"),
    ("inscene_e0", "EGAStem, 150ep", "2.59M"),
    ("inscene_e0_300e", "EGAStem, 300ep fixed", "2.59M"),
    ("inscene_yolo11s", "EGAStem-s, 150ep", "9.43M"),
    ("inscene_noaug", "EGAStem, 150ep noaug", "2.59M"),
]


def read_run(name: str):
    p = REPO_ROOT / "data" / "runs" / name / "results.csv"
    if not p.exists():
        return None
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(p.open(encoding="utf-8"))]
    if not rows:
        return None
    k50, k95 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"
    fit = lambda r: 0.1 * float(r[k50]) + 0.9 * float(r[k95])  # noqa: E731
    best = max(rows, key=fit)
    return {
        "epochs": len(rows),
        "best_ep": int(best["epoch"]),
        "mAP50": float(best[k50]),
        "mAP50_95": float(best[k95]),
        "fitness": fit(best),
    }


def run_pair(a: str, b: str, n: int = 2000):
    out = REPO_ROOT / "data" / "eval" / f"paired_{a}_vs_{b}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    cmd = [PY, str(CODE_DIR / "_paired_general.py"), "--a", a, "--b", b, "--n", str(n)]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> int:
    print("=" * 96)
    print("固定协议结果汇总")
    print("=" * 96)
    print(f"{'run':<22}{'臂':<30}{'轮数':>6}{'mAP50':>9}{'mAP50-95':>10}{'@ep':>6}")
    print("-" * 96)

    got = {}
    for name, arm, _ in RUNS:
        d = read_run(name)
        if d is None:
            print(f"{name:<22}{arm:<30}{'缺少结果':>6}")
            continue
        got[name] = d
        print(f"{name:<22}{arm:<30}{d['epochs']:>6}{d['mAP50']:>9.4f}{d['mAP50_95']:>10.4f}{d['best_ep']:>6}")

    pairs = [
        ("is_e0_p50", "inscene_e0", "budget 300 vs 150"),
        ("is_e0_p50", "is_base_p50", "stem ours vs official at 300ep"),
        ("inscene_e0", "inscene_yolo11s", "capacity n vs s at 150ep"),
        ("inscene_e0", "inscene_noaug", "augmentation on vs off at 150ep"),
    ]

    print("\n" + "=" * 96)
    print("配对 bootstrap（delta = A - B）")
    print("=" * 96)
    for a, b, desc in pairs:
        if a not in got or b not in got:
            print(f"{desc:<40} 跳过（缺少 run）")
            continue
        d = run_pair(a, b)
        m50 = d.get("mAP50", {})
        ci = m50.get("ci95", [0.0, 0.0])
        flag = "已确立" if (ci[0] > 0 or ci[1] < 0) else "未获确立"
        print(f"{desc:<40} Δ={m50.get('delta_mean', float('nan')):+.4f}  CI=[{ci[0]:+.4f},{ci[1]:+.4f}]  {flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
