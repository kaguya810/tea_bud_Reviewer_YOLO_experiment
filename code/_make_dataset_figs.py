# -*- coding: utf-8 -*-
"""生成数据集图样（按类别）。

产出：
    figs/dataset_samples.png    4 类 × 6 例，实例裁剪 + GT 框（主图）
    figs/dataset_overview.png   原始图 + GT 框全景（若干张）
    figs/dataset_stats.png      类别实例数 / 面积分布 / 长宽比分布

只用 train 划分（val 154 张太少，不足以展示每类形态）。
选样策略：每类按实例面积排序，取 15%~85% 分位之间等间隔抽 6 个 ——
避开极端大/小样本，展示该类"典型形态"。
"""
from __future__ import annotations

import collections
import pathlib

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
DS = ROOT / "yolo_detect_4cls_inscene"
OUT = ROOT / "figs"
OUT.mkdir(exist_ok=True)

NAMES = {0: "Singlebud", 1: "oneleafOnebud", 2: "threeleafOnebud", 3: "twoleafOnebud"}
CN = {0: "单芽", 1: "一芽一叶", 2: "一芽三叶", 3: "一芽二叶"}
COLOR = {0: "#ff4d5e", 1: "#ffa53d", 2: "#5fd68a", 3: "#5b8dd6"}


def use_cjk_font():
    for f in ("Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"):
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [f]
            plt.rcParams["axes.unicode_minus"] = False
            return f
        except Exception:
            continue
    return None


def load_split(split: str):
    """返回 [(img_path, cls, xyxy, W, H), ...]"""
    out = []
    idir = DS / "images" / split
    ldir = DS / "labels" / split
    for lp in sorted(ldir.glob("*.txt")):
        ip = idir / (lp.stem + ".jpg")
        if not ip.exists():
            continue
        im = cv2.imread(str(ip))
        if im is None:
            continue
        H, W = im.shape[:2]
        for line in lp.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(p[0])
            cx, cy, w, h = (float(v) for v in p[1:5])
            x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
            x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
            out.append((ip, c, (x1, y1, x2, y2), W, H))
    return out


def crop_with_box(inst, pad=0.18):
    ip, c, (x1, y1, x2, y2), W, H = inst
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad, bh * pad
    a = max(int(x1 - px), 0); b = max(int(y1 - py), 0)
    d = min(int(x2 + px), W); e = min(int(y2 + py), H)
    im = cv2.imread(str(ip))
    sub = im[b:e, a:d].copy()
    box = (x1 - a, y1 - b, x2 - a, y2 - b)
    return cv2.cvtColor(sub, cv2.COLOR_BGR2RGB), box


def fig_samples():
    inst = load_split("train")
    by = collections.defaultdict(list)
    for it in inst:
        x1, y1, x2, y2 = it[2]
        by[it[1]].append(((x2 - x1) * (y2 - y1), it))

    nrow, ncol = 4, 6
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow))
    for r in range(nrow):
        lst = sorted(by[r], key=lambda t: t[0])
        n = len(lst)
        lo, hi = int(n * 0.15), int(n * 0.85)
        idx = np.linspace(lo, hi - 1, ncol).astype(int)
        for j, k in enumerate(idx):
            ax = axes[r][j]
            rgb, (bx1, by1, bx2, by2) = crop_with_box(lst[k][1])
            ax.imshow(rgb)
            ax.add_patch(mpatches.Rectangle((bx1, by1), bx2 - bx1, by2 - by1,
                                            fill=False, edgecolor=COLOR[r], linewidth=1.8))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(COLOR[r]); s.set_linewidth(1.4)
            if j == 0:
                ax.set_ylabel(f"{CN[r]}\n{NAMES[r]}\nn={n}", fontsize=10,
                              color=COLOR[r], fontweight="bold", rotation=0,
                              ha="right", va="center", labelpad=42)
            ax.set_xlabel(f"{int(bx2-bx1)}×{int(by2-by1)} px", fontsize=8)
    fig.suptitle("茶芽采摘形态数据集 · 四类样本（train，按面积 15%~85% 分位等间隔抽取）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.04, 0, 1, 0.97))
    p = OUT / "dataset_samples.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  [1] 按类别样本 -> {p.relative_to(ROOT)}  ({p.stat().st_size//1024} KB)")


def fig_overview(k=4):
    """挑 k 张行实例数最多的图，画全部 GT 框。"""
    inst = load_split("train")
    by_img = collections.defaultdict(list)
    for it in inst:
        by_img[it[0]].append(it)
    picks = sorted(by_img.items(), key=lambda t: -len(t[1]))[:k]

    fig, axes = plt.subplots(1, k, figsize=(4.0 * k, 3.4))
    for ax, (ip, lst) in zip(np.atleast_1d(axes), picks):
        im = cv2.imread(str(ip))
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        for it in lst:
            x1, y1, x2, y2 = it[2]
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                            edgecolor=COLOR[it[1]], linewidth=2.0))
            ax.text(x1, max(y1 - 4, 10), CN[it[1]], fontsize=8, color="white",
                    bbox=dict(facecolor=COLOR[it[1]], edgecolor="none", pad=1.4))
        ax.set_title(f"{ip.name[:18]}…  ({im.shape[1]}×{im.shape[0]})", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [mpatches.Patch(color=COLOR[c], label=f"{CN[c]} / {NAMES[c]}") for c in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("原始图 + 全部 GT 框（train 中实例最多的若干张）", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    p = OUT / "dataset_overview.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  [2] 原图全景   -> {p.relative_to(ROOT)}  ({p.stat().st_size//1024} KB)")


def fig_stats():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))

    tr = load_split("train"); va = load_split("val")
    cnt_tr = collections.Counter(i[1] for i in tr)
    cnt_va = collections.Counter(i[1] for i in va)
    x = np.arange(4); w = 0.38
    axes[0].bar(x - w / 2, [cnt_tr[c] for c in range(4)], w, label="train", color="#5b8dd6")
    axes[0].bar(x + w / 2, [cnt_va[c] for c in range(4)], w, label="val", color="#ffa53d")
    for i, c in enumerate(range(4)):
        axes[0].text(i - w / 2, cnt_tr[c] + 8, str(cnt_tr[c]), ha="center", fontsize=8)
        axes[0].text(i + w / 2, cnt_va[c] + 8, str(cnt_va[c]), ha="center", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{CN[c]}\n{NAMES[c][:10]}" for c in range(4)], fontsize=8)
    axes[0].set_ylabel("实例数"); axes[0].legend(fontsize=9)
    axes[0].set_title("各类实例数", fontsize=11); axes[0].grid(axis="y", alpha=0.3)

    for c in range(4):
        a = [( (i[2][2]-i[2][0]) * (i[2][3]-i[2][1]) ) for i in tr if i[1] == c]
        axes[1].hist(a, bins=30, alpha=0.55, label=f"{CN[c]}({np.median(a):.0f})",
                     color=COLOR[c])
    axes[1].set_xlabel("实例面积 (px²)"); axes[1].set_ylabel("频数")
    axes[1].set_title("面积分布（图例为各类中位数）", fontsize=11)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    for c in range(4):
        ar = []
        for i in tr:
            if i[1] != c:
                continue
            bw = i[2][2]-i[2][0]; bh = i[2][3]-i[2][1]
            ar.append(max(bw, bh) / max(min(bw, bh), 1e-6))
        axes[2].hist(ar, bins=30, alpha=0.55, label=f"{CN[c]}({np.median(ar):.2f})", color=COLOR[c])
    axes[2].set_xlabel("长宽比 (长边/短边)"); axes[2].set_ylabel("频数")
    axes[2].set_title("形状（长宽比）分布", fontsize=11)
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    fig.suptitle("数据集统计（train 1221 / val 154）", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / "dataset_stats.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  [3] 数据集统计 -> {p.relative_to(ROOT)}  ({p.stat().st_size//1024} KB)")

    print("\n  各类中位面积 / 中位长宽比：")
    for c in range(4):
        a = [((i[2][2]-i[2][0])*(i[2][3]-i[2][1])) for i in tr if i[1] == c]
        ar = [max(i[2][2]-i[2][0], i[2][3]-i[2][1]) / max(min(i[2][2]-i[2][0], i[2][3]-i[2][1]), 1e-6)
              for i in tr if i[1] == c]
        print(f"    {CN[c]:<6}{NAMES[c]:<18} n={len(a):>4}  面积中位 {np.median(a):>8.0f} px²  "
              f"长宽比中位 {np.median(ar):.2f}")


def main() -> int:
    f = use_cjk_font()
    print(f"  中文字体: {f or '未找到（标签将显示为方块）'}")
    fig_samples()
    fig_overview()
    fig_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
