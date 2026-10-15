# -*- coding: utf-8 -*-
"""训练 YOLOv11n-P2 检测模型（EGA-Stem + DSA + LightSAFFM），或同配方的官方基线。

两条臂共用本脚本，**配方逐项相同**，唯一差别是模型：
    --arm ours      自定义 cfg（默认 3 尺度 cfg/yolo11n-tea-ega-dsa-saffm.yaml；--p2 切 4 尺度）
    --arm yolo11n   官方 yolo11n.yaml（3 尺度）
对照实验里「同配方 + 只换模型」是唯一能支撑结论的设计；只跑一条臂说明不了任何事。

⚠️ 为什么两条臂不能共用一个进程：`custom_modules_ega_dsa` 在 import 时就把
   `tasks.C3k2` 换成 `C3k2DSA`、`tasks.Detect` 换成 `DetectP2Half`。
   官方基线若在同一个进程里构建，第一个尺度会被 `DetectP2Half` 减半 —— 静默出错。
   所以 `--arm yolo11n` 走「不 import 自定义模块」的分支。

用法：
    # 冒烟（1 epoch，含 val / 保存 / best.pt 重载 / 冻结核审计）
    .venv\\Scripts\\python.exe train_ega_dsa.py --arm ours --epochs 1 --name smoke_ega_dsa

    # 正式
    .venv\\Scripts\\python.exe train_ega_dsa.py --arm ours --epochs 150 --batch 16 --name ega_dsa_150e

训后审计（本脚本自动做）：
    * 从 best.pt / last.pt 重新加载模型（= ultralytics 的 .pt 重建路径）
    * 断言固定核仍然是 buffer、且数值仍等于解析核（EGA「修复核冻结」在**真实训练回路**里的验证）
    * **分臂**断言参数量：ours 要求 trainer 重建后与建模态逐位一致；
      yolo11n 与独立推导的参照值（yolo11n.yaml + nc=4）比对，并断言头部是官方 Detect。
      （旧版一律拿 trainer 重建后的模型自比 —— 那在架构被污染时也会"通过"，
        审计报告 §4 里有演示。）
    * 断言审计确实执行了 >0 项检查（0 项=通过 是最隐蔽的 fail-open）
    * 无论审计是否抛异常，train_audit.json 一定落盘
"""

from __future__ import annotations

import argparse
import json
import pathlib
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent

# 两套配方。`baseline` 逐项对齐 experiments/baseline_e5/runs/yolov11n_det/args.yaml
# （那份 150 epoch 的 yolo11n 检测基线），以便与之直接对比。
# ⚠️ 但要注意：那份基线是 **COCO 预训练 + mosaic/HSV 开启**；我们的自定义结构
#    没有配套预训练权重，只能靠 intersect_dicts 部分匹配（实测按参数量仅 43.8%，
#    且整个 neck 为 0%——见 _pretrained_transfer_probe.py）。
#    因此这一口径下「结构差异」与「预训练量差异」是混在一起的，无法干净归因。
RECIPES = {
    "custom": dict(
        epochs=150, batch=16, patience=0, pretrained="",
        mosaic=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.1,
        scale=0.5, fliplr=0.5, close_mosaic=10, erasing=0.4, plots=True,
    ),
    "long": dict(
        # 300 轮 + 余弦退火 + aug2 的增强成分。
        # 依据：归档基线 best@ep81 后掉 −14%（150 轮不够/不对），且 ours 在 ep150 仍在上升。
        # 成本：23.5 s/epoch × 300 ≈ 1.96 h。
        epochs=300, batch=16, patience=0, pretrained="yolo11n.pt",
        mosaic=0.5, hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, translate=0.1,
        scale=0.25, fliplr=0.5, close_mosaic=30, erasing=0.0, plots=False,
        cos_lr=True,
    ),
    "aug2": dict(
        # 保留正则、去掉失真。逐项差异见本文件上方注释 —— 只改增强，其余与 baseline 相同。
        # 设计依据：基线过拟合（需要正则）+ 当前 mosaic/scale/hsv_s 造成 2.09× 尺度失配（有害）。
        epochs=150, batch=16, patience=150, pretrained="yolo11n.pt",
        mosaic=0.5, hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, translate=0.1,
        scale=0.25, fliplr=0.5, close_mosaic=30, erasing=0.0, plots=False,
    ),
    "noaug": dict(
        # 全关增强。与 baseline 只差这一块，其余逐项相同 -> 可直接对比。
        # mosaic/scale 造成的 2.09× 尺度失配见文件头与本配方的推导（_aug_audit.py）。
        epochs=150, batch=16, patience=150, pretrained="yolo11n.pt",
        mosaic=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.0,
        # fliplr 也按「全关」设为 0。它是这批增强里唯一通常安全的一个
        # （类别是按叶片数量定义的，翻转应保标签），所以如果你后面发现
        # 无增强版本过拟合，第一个该恢复的就是它 —— 改成 0.5 即可。
        scale=0.0, fliplr=0.0, close_mosaic=0, erasing=0.0, plots=False,
    ),
    "baseline": dict(
        # batch 由 8 提到 16：ultralytics 的 nbs=64 会做梯度累积
        #   accumulate = max(round(nbs/batch), 1)，且 weight_decay 按
        #   wd * batch * accumulate / nbs 归一、scheduler 的 iterations 用
        #   ceil(N / max(batch, nbs)) —— 三者都保证「等效 batch 恒为 64」。
        # 所以改物理 batch **不改变优化动态**，只改 BatchNorm 的微批统计量与显存。
        # 实测（batch 扫描，见 EGA_DSA_HFP_TEST.md §9.6）：
        #   b8 192s/3.4GiB · b16 181s/6.6GiB · b32 180s/13.1GiB · b48 病态(抖动，不可用)
        # 收益在 b16 就饱和（b16→b32 只快 0.6%），故取 16。
        epochs=150, batch=16, patience=150, pretrained="yolo11n.pt",
        mosaic=1.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1,
        scale=0.5, fliplr=0.5, close_mosaic=10, erasing=0.4, plots=False,
    ),
}


def _sha256(path: str) -> str:
    """给模型 YAML 留摘要，便于事后确认训练用的到底是哪一版配置。"""
    try:
        import hashlib

        if os.path.isfile(path):
            return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    except Exception:
        pass
    return "n/a"


def _ultra_version() -> str:
    try:
        import ultralytics

        return ultralytics.__version__
    except Exception:
        return "n/a"


_ULTRA_VER = _ultra_version()


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv11n-P2 (EGA/DSA/LightSAFFM) or the official baseline")
    p.add_argument("--arm", choices=["ours", "yolo11n"], default="ours",
                   help="ours = 自定义 cfg；yolo11n = 官方 3 尺度基线（同配方）")
    p.add_argument("--recipe", choices=sorted(RECIPES), default="baseline",
                   help="baseline = 逐项对齐已有的 150ep yolo11n 检测基线（预训练 + mosaic/HSV）；"
                        "custom = 从零 + 关闭 mosaic/HSV。显式给出的命令行参数优先于预设。")
    p.add_argument("--cfg", default="", help="覆盖默认模型 YAML")
    p.add_argument("--data", default="experiments/baseline_e5/tea_bud_4cls_v2k_detect.yaml",
                   help="数据集 YAML（v2k 4 类检测集，1221/79/75）")
    p.add_argument("--name", default="ega_dsa_run1")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--seed", type=int, default=67)
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader 进程数。默认 0：本机 Windows 上 workers>0 会在 "
                        "popen_spawn_win32 里 pickle 数据集时抛 OSError: [Errno 22] Invalid argument "
                        "（cache='ram' 时数据集对象很大）。仓库既有 train_yolo_detect.py 也用 0。")
    p.add_argument("--no-amp", action="store_true", help="关闭 AMP（排查 NaN 时用）")
    p.add_argument("--allow-cpu", action="store_true", help="允许在 CPU 上训练（默认直接报错）")
    p.add_argument("--cos-lr", action="store_true", default=None,
                   help="启用余弦退火（默认由 --recipe 决定）")
    p.add_argument("--p2", action="store_true",
                   help="改用 4 尺度 P2 变体（cfg/%s）+ 打开 P2 头减半。"
                        "默认是 3 尺度 P2/8-16-32，且**不做任何通道减半**。" % "cfg/yolo11n-tea-ega-dsa-saffm-p2.yaml")
    p.add_argument("--exist-ok", action="store_true",
                   help="允许写入已存在的 run 目录（默认拒绝，避免静默覆盖上一次训练）")
    p.add_argument("--no-cache", action="store_true", help="关闭 RAM 缓存")
    p.add_argument("--channels-last", choices=["auto", "on", "off"], default="off",
                   help="auto = ultralytics 默认（None -> 在 CUDA 上**开启** NHWC）。"
                        "默认改成 off：实测两条臂都因此明显变快（见 §9.8）——"
                        "本模型 173.0 -> 119.4 s/epoch，官方 yolo11n 57.9 -> 38.7 s/epoch。"
                        "两条臂同设置，对照仍然公平。")
    p.add_argument("--compile", choices=["off", "default", "reduce-overhead", "max-autotune-no-cudagraphs"],
                   default="off",
                   help="torch.compile。需要 triton（本机已装 triton-windows 3.8.0）。"
                        "⚠️ 首次编译很慢（本模型约 10+ 分钟，一次性）；"
                        "且 compile 与 channels_last 有强交互 —— 编译后 NHWC 反而更快（见 §9.9）。")
    p.add_argument("--resume", action="store_true")
    # ---- 以下默认值由 --recipe 预设提供；显式传参则覆盖预设 ----
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--patience", type=int, default=None, help="早停轮数，0 = 关闭")
    p.add_argument("--pretrained", default=None, help="预训练权重路径，'' = 从零")
    p.add_argument("--mosaic", type=float, default=None)
    p.add_argument("--hsv-h", type=float, default=None)
    p.add_argument("--hsv-s", type=float, default=None)
    p.add_argument("--hsv-v", type=float, default=None)
    p.add_argument("--translate", type=float, default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--fliplr", type=float, default=None)
    p.add_argument("--close-mosaic", type=int, default=None)
    p.add_argument("--erasing", type=float, default=None)
    p.add_argument("--plots", action="store_true", default=None)

    args = p.parse_args()
    # 预设只填「用户没显式指定」的项
    for k, v in RECIPES[args.recipe].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


# --------------------------------------------------------------------------------
# 训后审计
# --------------------------------------------------------------------------------
FIXED_KEYS = ("scharr_w", "gauss_w")


def _fixed_deviation(buf: torch.Tensor, analytic: torch.Tensor) -> float:
    """固定核与解析核的最大偏差，**扣除 .pt 的 fp16 存储舍入**。

    ultralytics 保存 checkpoint 时是 `deepcopy(model).half()`，回读后再转回 fp32，
    所以核天然会带 ~1e-5 量级的 fp16 量化误差。第一版审计拿它跟 fp32 解析核比、
    阈值卡 1e-6，于是把「存储精度」误报成了「核被训练改写」：
        偏差 7.309e-06 > 1e-6 -> ❌   （假阳性）
    正确做法是让参照物走一遍**同样的 fp16 往返**再比：
        实测 buffer == fp16(解析核) 逐位相等 -> 偏差 0.0
    这样阈值仍能卡住真正的问题（真被训练改动的话偏差是 1e-3 量级，远超 fp16 舍入）。
    """
    ref = analytic.to(buf.dtype)
    if ref.dtype == torch.float32:
        ref = ref.half().float()  # 与 checkpoint 的存储精度对齐
    return float((buf.detach().cpu().float() - ref.detach().cpu().float()).abs().max())


def _audit_in_process(trained, snapshot: dict, log: dict) -> bool:
    """**进程内、fp32、逐位**比较训练前后的固定核 —— 绕过 .pt 的 fp16 存储，证据最强。

    trainer.model 是真正被优化的模型（AMP 只影响前向计算，权重始终 fp32），
    trainer.ema.ema 是要被保存的 EMA 模型。两者都应保持固定核不变。
    （EMA 的 update 会遍历 state_dict（含 buffer）做 v*d + (1-d)*v，
      代数上是恒等，但 fp32 下有 ~1 ulp 的舍入，故阈值取 1e-9 而不是严格 0。）
    """
    ok = True
    tr = getattr(trained, "trainer", None)
    candidates = [("trainer.model", getattr(tr, "model", None)),
                  ("trainer.ema.ema", getattr(getattr(tr, "ema", None), "ema", None))]
    for tag, mod in candidates:
        if mod is None:
            continue
        stem = mod.model[0]
        if not all(k in dict(stem.named_buffers()) for k in FIXED_KEYS):
            print(f"[audit] {tag}: 找不到固定核 buffer")
            ok = False
            continue
        dev = max(float((getattr(stem, k).detach().cpu().float() - snapshot[k].cpu()).abs().max())
                  for k in FIXED_KEYS)
        good = dev < 1e-9
        print(f"[audit] {tag} 固定核训练前后偏差 = {dev:.3e}（fp32 逐位比较）-> {'OK 未被改动' if good else '❌ 被改动了'}")
        log[f"inproc_drift_{tag}"] = dev
        ok &= good
    return ok


def audit_checkpoint(ckpt: Path, arm: str, expect_params: int, log: dict) -> bool:
    """重新加载 .pt，验证固定核仍冻结 + 参数量一致。

    这一步是本脚本存在的主要理由之一：`EGAStem` 的冻结写在 __init__ 内，
    但 ultralytics 训练器会**从 YAML 重建模型再灌权重**，
    冻结若写错位置就会在这里静默解冻 —— 必须在真实训练回路里验一次，不能只信建模时的自检。
    """
    ok = True
    print(f"\n[audit] 重新加载 {ckpt.name} ...")
    from ultralytics import YOLO

    m = YOLO(str(ckpt)).model
    n = sum(p.numel() for p in m.parameters())
    same_params = n == expect_params
    print(f"[audit] 参数量 {n:,}（期望 {expect_params:,}） -> {'OK' if same_params else '❌ 不一致'}")
    log["reloaded_params"] = n
    log["params_match"] = bool(same_params)
    ok &= same_params

    if arm == "ours":
        import custom_modules_ega_dsa as M

        stem = m.model[0]
        is_buf = all(k in dict(stem.named_buffers()) for k in ("scharr_w", "gauss_w"))
        not_param = not any(("scharr" in n or "gauss" in n) for n, _ in m.named_parameters())
        print(f"[audit] 固定核是 buffer 且不在 named_parameters -> {'OK' if is_buf and not_param else '❌'}")
        log["fixed_is_buffer"] = bool(is_buf and not_param)
        ok &= is_buf and not_param

        # 数值：核必须仍等于解析核（trainer 若解冻并训练了它，这里会露出来）
        d_gauss = _fixed_deviation(stem.gauss_w, M.gaussian_kernel(stem.gauss_k, stem.sigma))
        d_scharr = _fixed_deviation(stem.scharr_w, M.scharr_depthwise_weight(stem.c1))
        print(f"[audit] 高斯核与解析核偏差（已扣除 fp16 存储舍入）= {d_gauss:.3e} -> {'OK' if d_gauss < 1e-6 else '❌ 被改写'}")
        print(f"[audit] Scharr 核与解析核偏差（同上）            = {d_scharr:.3e} -> {'OK' if d_scharr < 1e-6 else '❌ 被改写'}")
        log["gauss_drift"] = d_gauss
        log["scharr_drift"] = d_scharr
        ok &= d_gauss < 1e-6 and d_scharr < 1e-6

        head = m.model[-1]
        if isinstance(head, M.DetectP2Half):
            print(f"[audit] 检测头通道 cv2={head.head_ch['cv2']} cv3={head.head_ch['cv3']}")
            log["head_ch"] = head.head_ch
    return ok


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    # m5：`resume=True` 只有在 model 是 last.pt 时才有意义；本脚本的 model_src 永远是
    # YAML / yolo11n.pt，ultralytics 会把它降级成 False 并只打一条 warning。而
    # exist_ok=True 会让"resume"在同一个目录里**从头重训**并清掉 results.csv。
    # 与其静默重来，不如拒绝。
    if args.resume:
        raise SystemExit(
            "--resume 在本脚本里不可用：model_src 是 YAML/官方 .pt，不是可续训的 last.pt，"
            "ultralytics 会把 resume 降级为 False，而 run 目录会被原地覆盖。"
            "要续训请直接用 ultralytics 的 `YOLO('runs/<name>/weights/last.pt').train(resume=True)`。"
        )

    if args.arm == "ours":
        import custom_modules_ega_dsa as M  # 必须在构建模型之前

        # 检测头必须**显式**选：parse_model 的 `elif m in frozenset({Detect, ...})` 是拿
        # tasks 模块 globals 里的 `Detect` 现算的，也就是说它比对的是「这个名字当前指向谁」。
        # 一旦全局指向 DetectP2Half，任何写 `Detect` 的配置都会走 DetectP2Half 分支，
        # 而它默认 p2_index=0 -> **静默把第 0 个尺度（P3）的头隐藏通道减半**，不报错。
        M.configure_detect_head(bool(args.p2))
    else:
        # 官方基线：绝不能加载自定义模块（它会把 tasks.C3k2 换成 C3k2DSA、tasks.Detect 换成
        # DetectP2Half）。用 raise 而不是 assert —— assert 在 `python -O` 下会被整条剥离。
        if "custom_modules_ega_dsa" in sys.modules:
            raise RuntimeError(
                "官方基线臂里加载了 custom_modules_ega_dsa —— tasks.C3k2/tasks.Detect 已被替换，"
                "该臂不再是官方基线。请用独立进程运行 `--arm yolo11n`。"
            )

    from ultralytics import YOLO
    import ultralytics.nn.modules as _U
    import ultralytics.nn.tasks as _tasks

    # **正向**检查：确认 tasks 命名空间里的关键类还是官方的。
    # 只靠「本模块没被 import」是不够的：custom_modules.py 也会 monkey-patch parse_model，
    # 而它不在上面那条守卫的覆盖范围内。
    if args.arm == "yolo11n":
        _bad = [n for n, a, b in (("C3k2", _tasks.C3k2, _U.C3k2), ("Detect", _tasks.Detect, _U.Detect)) if a is not b]
        if _bad:
            raise RuntimeError(
                f"官方基线臂：tasks 命名空间里的 {_bad} 已被替换（不是 ultralytics 原始类），"
                f"该臂不是干净的对照。请用独立进程运行。"
            )

    # cfg 解析：
    #   * 显式 --cfg         -> 相对本仓库根目录
    #   * ours               -> 本仓库的自定义 YAML
    #   * yolo11n            -> ultralytics **内置**配置，必须原样交给它（不能拼 ROOT，
    #                           否则会去找 D:\...\yolo11n.yaml 而 FileNotFoundError）
    # M5：--cfg 与「yolo11n + baseline 配方」互斥 —— 后者会把 model_src 覆盖成 yolo11n.pt，
    # 于是 --cfg 被静默忽略（也不会报错）。这种组合直接拒绝，别让人以为换成了自己的配置。
    if args.cfg and args.arm != "ours" and args.recipe == "baseline":
        raise SystemExit(
            "--cfg 与 `--arm yolo11n --recipe baseline` 互斥：该组合的 model_src 固定是 "
            "yolo11n.pt，--cfg 会被忽略。请改用 --arm ours，或加 --recipe custom。"
        )
    if args.cfg:
        cfg = args.cfg if os.path.isabs(args.cfg) else str(ROOT / args.cfg)
    elif args.arm == "ours":
        _cfg_name = "yolo11n-tea-ega-dsa-saffm-p2.yaml" if args.p2 else "yolo11n-tea-ega-dsa-saffm.yaml"
        cfg = str(ROOT / "cfg" / _cfg_name)
    else:
        cfg = "yolo11n.yaml"
    if not torch.cuda.is_available() and not args.allow_cpu:
        # 默认 150 epoch，静默跑在 CPU 上等于白跑一晚。
        raise SystemExit("CUDA 不可用。若确实要用 CPU 跑，请显式加 --allow-cpu。")
    dev = 0 if torch.cuda.is_available() else "cpu"

    # 模型来源：
    #   * ours                      -> 自定义 YAML（可选 .load(预训练)）
    #   * yolo11n + recipe=baseline -> 直接 YOLO('yolo11n.pt')，与既有基线 args.yaml 的
    #                                  `model: ...\yolo11n.pt` 完全一致（架构+权重都来自 ckpt）
    #   * yolo11n + recipe=custom   -> ultralytics 内置 yolo11n.yaml（从零）
    def _resolve_pt(v):
        """把配方/命令行给的 pretrained 解析成本地 .pt 绝对路径；不是文件就返回 None。"""
        if not v or not str(v).lower().endswith(".pt"):
            return None
        q = pathlib.Path(v)
        if not q.is_absolute():
            q = ROOT / q
        return str(q) if q.is_file() else None

    if args.arm == "ours":
        model_src = cfg
    else:
        # ⚠️ 不能拿「配方名 == baseline」当「用 yolo11n.pt」的代理条件 ——
        #    加 noaug 配方时这个隐式耦合就断了，官方臂会静默变成"从零构建"。
        #    改成直接看配方里的 pretrained 是不是一个真实存在的 .pt。
        model_src = _resolve_pt(args.pretrained) or cfg

    print("=" * 88)
    print(f"arm={args.arm}  recipe={args.recipe}")
    print(f"model_src={model_src}")
    print(f"data={args.data}")
    print(f"epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} workers={args.workers} "
          f"amp={not args.no_amp} cache={not args.no_cache} seed={args.seed} patience={args.patience}")
    print(f"aug: mosaic={args.mosaic} hsv=({args.hsv_h},{args.hsv_s},{args.hsv_v}) "
          f"translate={args.translate} scale={args.scale} fliplr={args.fliplr} "
          f"erasing={args.erasing} close_mosaic={args.close_mosaic}")
    print(f"pretrained={args.pretrained or '（从零）'}")
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}"
          + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    print("=" * 88)

    model = YOLO(model_src)
    n_build = sum(p.numel() for p in model.model.parameters())
    n_train = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    _note = "" if n_train else "  ← .pt 加载后 ultralytics 默认把所有张量设为 requires_grad=False（推理用）；\n" \
                             "             trainer 在 _setup_train 里会重新解冻，见 train_audit.json 的 params_trainer"
    print(f"建模完成：{len(model.model.model)} 层 / {n_build:,} 参数（可训练 {n_train:,}）{_note}")

    if args.pretrained and args.arm == "ours":
        pw = str(ROOT / args.pretrained) if not os.path.isabs(args.pretrained) else args.pretrained
        model.load(pw)
        print(f"已加载预训练权重：{args.pretrained}"
              f"（⚠️ 自定义结构只能部分匹配，见 _pretrained_transfer_probe.py）")

    # 训练前快照固定核（fp32）—— 训练后做进程内逐位比较，这是「核没被训练改动」的最强证据
    snap_fixed = ({k: getattr(model.model.model[0], k).detach().clone() for k in FIXED_KEYS}
                  if args.arm == "ours" else {})

    # m1：reset 必须在训练**之前**，否则读到的是进程级高水位（含建模 / AMP 探针 /
    # 训练后的 final_eval），标签却写着「训练显存峰值」。
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model.train(
        data=str(ROOT / args.data) if not os.path.isabs(args.data) else args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=dev,
        cache=not args.no_cache,
        amp=not args.no_amp,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        resume=args.resume,
        # 增强口径由 --recipe 决定（baseline 逐项对齐已有 150ep 基线）
        mosaic=args.mosaic,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        close_mosaic=args.close_mosaic,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        translate=args.translate,
        fliplr=args.fliplr,
        flipud=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        erasing=args.erasing,
        scale=args.scale,
        # 对检测任务是惰性的（ultralytics 只在 classify_augmentations 里消费它），
        # 但"全关增强"就把话说明白，别留隐式默认值。
        auto_augment="none",
        # 余弦退火开关（默认关，保持与历史配方可比）。由 --recipe 控制。
        cos_lr=bool(getattr(args, "cos_lr", False)),
        project=str(ROOT / "runs"),
        name=args.name,
        # m5：exist_ok=True 会在同名时原地覆盖上一次 run（连 results.csv 一起清掉）。
        # runs/ega_dsa_150e 里已经躺着一个旧配置的 args.yaml 就是这么来的。
        exist_ok=args.exist_ok,
        plots=args.plots,
        val=True,
        deterministic=True,
        # M2：`pretrained` 必须显式转发。它默认是 True —— 对 `--arm yolo11n`（model_src 是
        # yolo11n.pt）意味着 Model.train 走 ckpt 分支，于是 `--pretrained ''` 想「从零」
        # 也做不到，而 `--pretrained other.pt` 会被静默丢弃。
        # 传 False 时 ultralytics 用 self.model（即我们已构建/已 load 的那个），语义正确。
        pretrained=(args.pretrained or False),
        # ultralytics: channels_last=None 时在 CUDA 上默认**开启**（会把模型转 NHWC）。
        channels_last={"auto": None, "on": True, "off": False}[args.channels_last],
        compile=False if args.compile == "off" else args.compile,
    )
    wall = time.time() - t0

    # 显存峰值 + 梯度累积步数：判断「加大 batch」到底值不值，看的就是这两项
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
    else:
        peak = None  # m2：不要写 float('nan') —— json.dumps 会产出裸 NaN，不是严格 JSON
    tr = getattr(model, "trainer", None)
    accumulate = int(getattr(tr, "accumulate", 0) or 0)
    # M1：记**实际用的** batch。ultralytics 在首个 epoch OOM 时会把 batch 减半并重建
    # 优化器/scheduler/loss scale；照抄 args.batch 会写下一个从未执行过的配方。
    batch_used = int(getattr(tr, "batch_size", args.batch) or args.batch)
    print(f"显存峰值 = {peak if peak is None else round(peak, 2)} GiB   "
          f"梯度累积 = {accumulate} 步 (等效 batch = {batch_used * max(accumulate, 1)})")
    if batch_used != args.batch:
        print(f"[audit] ⚠️ trainer 把 batch 从 {args.batch} 改成了 {batch_used}"
              f"（通常是首个 epoch OOM 自动减半）—— 该 run 的配方与命令行不一致")

    save_dir = Path(model.trainer.save_dir)

    # m3：真实轮数与**稳态** per-epoch。results.csv 的 time 列是累计秒，做差分即得每轮净耗时
    # （不含一次性启动）。文档 §9.8.4 记录过这个坑：1-epoch 墙钟 173s，稳态其实只有 93.2s。
    epochs_ran, steady, per_epoch_list = 0, None, []
    try:
        import csv as _csv

        _rc = save_dir / "results.csv"
        if _rc.exists():
            _rows = [{k.strip(): v for k, v in r.items()} for r in _csv.DictReader(_rc.open(encoding="utf-8"))]
            _prev = 0.0
            for _r in _rows:
                _t = float(_r.get("time", 0.0))
                per_epoch_list.append(round(_t - _prev, 2))
                _prev = _t
            epochs_ran = len(per_epoch_list)
            if len(per_epoch_list) > 1:          # 第 1 轮含编译/缓存填充，取其后均值
                steady = round(sum(per_epoch_list[1:]) / len(per_epoch_list[1:]), 2)
            else:
                steady = per_epoch_list[0] if per_epoch_list else None
    except Exception as _e:  # noqa: BLE001
        print(f"[audit] results.csv 解析失败（稳态耗时不可得）: {type(_e).__name__}: {_e}")

    # 从建好的模型里读 DSCN 的实际实现（不信任任何默认字符串）
    _impls = sorted({mm.impl for mm in model.model.modules()
                     if type(mm).__name__ == "DSCN1D"})
    dscn_impl_actual = ",".join(_impls) if _impls else "n/a"
    print(f"DSCN 实际实现 = {dscn_impl_actual}（env DSCN_IMPL={os.environ.get('DSCN_IMPL', '(未设)')}）")

    log = {
        "arm": args.arm, "recipe": args.recipe, "model_src": model_src,
        "cfg": cfg, "data": args.data, "epochs": args.epochs,
        "imgsz": args.imgsz, "batch_requested": args.batch, "batch_used": batch_used,
        "seed": args.seed, "workers": args.workers, "cache": not args.no_cache,
        "channels_last": args.channels_last, "compile": args.compile,
        # ⚠️ 这里**必须**记录模型的实际实现，不能记 env 的默认字符串：
        #    之前写的是 os.environ.get("DSCN_IMPL", "grid")，那个 "grid" 是硬编码的，
        #    而模块默认已改成 "gather" —— 于是模型用 gather 跑、审计却记 grid，
        #    溯源信息直接说谎。教训：溯源要从**产物**里读，不要从**输入**里抄。
        "dscn_impl": dscn_impl_actual,
        "detect_head": (M.detect_head_name() if args.arm == "ours" else "official"),
        "peak_gib": peak, "accumulate": accumulate, "per_epoch_seconds": per_epoch_list,
        # m4：把配方完整记下来，否则 JSON 无法自证两臂配方相同
        "recipe_args": {k: getattr(args, k) for k in
                        ("mosaic", "hsv_h", "hsv_s", "hsv_v", "translate", "scale", "fliplr",
                         "erasing", "close_mosaic", "patience", "plots", "pretrained")},
        "torch": torch.__version__, "ultralytics": _ULTRA_VER, "cfg_sha256": _sha256(cfg),
        "amp": not args.no_amp, "params_build": n_build, "params_trainable": n_train,
        "wall_seconds": wall, "save_dir": str(save_dir),
        # m3：wall/epochs 含一次性启动，且除的是**请求**轮数而非实际轮数。
        # 文档 §9.8.4 就是这个指标的受害者（1-epoch 墙钟 173s vs 稳态 93.2s）。
        # 这里给出真实稳态值（读 results.csv 的累计 time 列做差分），并保留旧键以便对照。
        "per_epoch_seconds_incl_setup": wall / max(args.epochs, 1),
        "epochs_ran": epochs_ran,
        "steady_epoch_seconds": steady,
    }
    print(f"\n训练结束：{wall/60:.1f} min（{wall/max(args.epochs,1):.1f} s/epoch）")

    # ---- 训后审计 ----
    # ⚠️ 参照参数量：**分臂**取值，不能一律拿 trainer 重建后的模型自比。
    #    审计演示过的失败模式：在污染进程里，基线臂的 trainer 重建会变成
    #    C3k2DSA/DetectP2Half（2,536,636 参数），而「建模态 vs 重建后」都等于它，
    #    于是审计打印 ✅ —— 训练出来的根本不是官方基线，却毫无察觉。
    #    ours : 重建必须与建模态**逐位一致**（同为 cfg + nc=4），不一致就是缺陷 -> 判失败。
    #    yolo11n: 与**独立推导**的参照值比对（yolo11n.yaml + nc=4 -> 2,590,620），
    #             并断言头部是官方 Detect。
    tr = getattr(model, "trainer", None)
    tr_model = getattr(tr, "model", None)
    n_ref = sum(p.numel() for p in tr_model.parameters()) if tr_model is not None else n_build
    log["params_trainer"] = n_ref

    ok = True
    if args.arm == "ours":
        check_zero = tr_model is None
        if check_zero:
            print("[audit] ❌ 拿不到 trainer.model，审计无从谈起（旧版这里会静默返回 True）")
        ok &= not check_zero
        if n_ref != n_build:
            print(f"[audit] ❌ ours 臂 trainer 重建后 {n_ref:,} != 建模态 {n_build:,}"
                  f"（差 {n_ref - n_build:+,}）—— 训练用的不是我们构建的架构")
            ok = False
        else:
            print(f"[audit] trainer 重建与建模态一致：{n_ref:,} 参数 -> OK")
    else:
        # 独立推导参照：官方 yolo11n.yaml 按数据集 nc 重建
        try:
            from ultralytics.nn.tasks import DetectionModel as _DM, yaml_model_load as _yml
            import yaml as _y

            _d = _y.safe_load(open(str(ROOT / "experiments" / "baseline_e5"
                                        / "tea_bud_4cls_v2k_detect.yaml"), encoding="utf-8"))
            _nc = int(_d["nc"])
            _ref_model = _DM(_yml("yolo11n.yaml"), ch=3, nc=_nc, verbose=False)
            _n_expect = sum(p.numel() for p in _ref_model.parameters())
            _h_expect = type(_ref_model.model[-1]).__name__
            log["baseline_ref_params"] = _n_expect
            same = (n_ref == _n_expect)
            print(f"[audit] yolo11n 臂：trainer 重建 {n_ref:,} vs 独立参照 {_n_expect:,} -> "
                  f"{'OK' if same else '❌ 不一致'}")
            ok &= same
        except Exception as _e:  # noqa: BLE001
            print(f"[audit] ❌ 无法独立推导基线参照值：{type(_e).__name__}: {_e}")
            ok = False

    # 审计条数：0 项检查=通过 是最隐蔽的 fail-open
    n_checks = 0

    def _ck(cond: bool, msg: str) -> bool:
        nonlocal n_checks
        n_checks += 1
        print(f"[audit] {msg} -> {'OK' if cond else '❌'}")
        return bool(cond)

    try:
        if args.arm == "ours":
            ok &= _audit_in_process(model, snap_fixed, log)

        if args.arm == "yolo11n":
            # M4：头部类型必须按臂断言（旧版只在 isinstance(DetectP2Half) 时才检查，
            # 于是一个被污染的 DetectP2Half 基线头会被静默跳过）
            _h = tr_model.model[-1] if tr_model is not None else None
            ok &= _ck(_h is not None and type(_h).__name__ == "Detect",
                      f"yolo11n 臂头部类型 == Detect（实际 {type(_h).__name__ if _h else 'None'}）")

        for nm in ("best.pt", "last.pt"):
            ck = save_dir / "weights" / nm
            if ck.exists():
                n_checks += 1
                ok &= audit_checkpoint(ck, args.arm, n_ref, log)
            else:
                print(f"[audit] ❌ 缺少 {ck}")
                ok = False

        ok &= _ck(n_checks > 0, f"审计实际执行了 {n_checks} 项检查（0 项=通过 是最隐蔽的 fail-open）")
    except Exception as e:  # noqa: BLE001
        # m9：审计中途抛异常以前会让 train_audit.json **完全不落盘**，什么证据都不剩
        import traceback

        traceback.print_exc()
        log["audit_exception"] = f"{type(e).__name__}: {e}"
        ok = False
    finally:
        log["audit_ok"] = bool(ok)
        log["audit_checks"] = n_checks
        (save_dir / "train_audit.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n审计结果：{'✅ 通过' if ok else '❌ 未通过'}（{n_checks} 项检查）  "
              f"已写入 {save_dir/'train_audit.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
