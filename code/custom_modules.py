# -*- coding: utf-8 -*-
"""自定义模块注册中心：把 paper/ 里的 8 个论文模块 + 若干主干部件接进 ultralytics。

============================ 机制（先读这段） ============================

ultralytics 的模型由 ultralytics/nn/tasks.py 的 parse_model() 按 YAML 逐行搭建：

    m = getattr(torch.nn, m[3:]) if m.startswith("nn.") else globals()[m]      # tasks.py:2082-2088

YAML 里写的模块名必须能在 **ultralytics.nn.tasks 的命名空间**里找到（tasks.py 用显式导入清单，
比 nn.modules.__all__ 窄，CBAM / C3k 之类都没导出）。找到类之后还要看两份白名单：

  - base_modules   （tasks.py:2024）  args -> [c1, c2, *其余]，c2 按 width 缩放（YAML 写"名义通道"）
  - repeat_modules （tasks.py:2062）  args 里插入重复次数 n
两份都不在的模块走 else 分支（tasks.py:2179）：c2 = ch[f]，args 原样传入、不缩放。

本文件用三种方式把自定义模块接进去：

  1) register_base_module(name, cls) —— **本文件的通用做法**
     base_modules 是一个"类对象"的 frozenset，而 parse_model 每次调用时才按名字从 globals 取值组装。
     所以只要把自定义类**临时绑到一个白名单名字**上（借壳；_BORROW_SLOTS 里都是 base_modules 成员、
     parse_model 里没有身份判断、官方 cfg/models 里没人用的遗留名字），调用结束后立刻还原——
     这样自定义类就获得完整的 base_modules 语义：通道自动缩放、c1/c2 记账正确、跨 n/s/m/l/x 都安全，
     而且不给进程留下全局副作用。

  2) tasks.X = X（新名字注入）—— 只适合"通道保持且参数与原通道无关"的模块，YAML 里要写缩放后的实际通道。

  3) tasks.C3k2 = C3k2CBAM（同名替换）—— 复用白名单，但会覆盖所有同名层。

============================ YAML 用法速查 ============================

    activation: NeLU()                          # 全局把 Conv 的激活换成 NeLU（Conv.default_act 是进程级）

    backbone:
      - [-1, 1, PCDM,          [512]]           # 下采样+加宽，c2 必须 > c1，H/W 需偶数
      - [-1, 1, IDConv,        [512]]           # 通道保持；可写 [512, 3, 11, 0.125]
      - [-1, 1, MSA,           [512, 8, True]]  # 通道保持；heads 必须整除通道
      - [-1, 1, DCAFE,         [512]]           # 通道保持（论文签名是 inp/oup，实际必须相等）
      - [-1, 1, IIA,           [512]]           # 通道保持
      - [-1, 1, GatedCNNBlock, [512, 7]]        # 通道保持；7 = 深度卷积核
      - [-1, 1, LRSA,          [512, 4]]        # 通道保持；4 = heads
      - [-1, 1, NeLU,          [512]]           # 通道保持（也可以只当激活函数用）

  要点：这些模块都不在 repeat_modules 里，YAML 第 2 列（repeats）请写 1；
        YAML 里的通道号是"名义通道"，实际 = make_divisible(min(名义值, max_channels) * width, 8)。

必须先 import 本模块再构建模型；用 best.pt 推理时同样要 import，否则反序列化会 ModuleNotFoundError。
"""

import importlib.util
import os
import sys
import types
import warnings

import torch
from torch import nn

import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules import CBAM, C3k2, Conv

PAPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper")

REGISTERED = {}  # YAML 名字 -> 说明
SKIPPED = {}  # YAML 名字 -> 跳过原因


# ======================================================================================
# 1. paper/ 里那两个第三方依赖：没装的话用等价实现顶一下
#    （只在加载 paper 文件那一刻注入 sys.modules，加载完立刻移除，不污染全局）
# ======================================================================================
class _DropPath(nn.Module):
    """timm.models.layers.DropPath 的等价实现（paper/28 只用到这一个 timm 符号）。"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x.div(keep) * mask

    def __repr__(self):
        return f"DropPath(p={self.drop_prob})"


def _trunc_normal_(*args, **kwargs):
    raise NotImplementedError("paper/28 并未真正使用 trunc_normal_，内置替身只用于满足 import")


def _rearrange(x, pattern, **kw):
    """einops.rearrange 的最小等价实现，只覆盖 paper/62 用到的两个 pattern。"""
    if pattern == "b (head c) h w -> b head c (h w)":
        b, c, h, w = x.shape
        head = kw["head"]
        return x.reshape(b, head, c // head, h, w).reshape(b, head, c // head, h * w)
    if pattern == "b head c (h w) -> b (head c) h w":
        b, head, c, _ = x.shape
        h, w = kw["h"], kw["w"]
        return x.reshape(b, head, c, h, w).reshape(b, head * c, h, w)
    raise NotImplementedError(f"内置 rearrange 不支持 pattern: {pattern}（pip install einops 可用完整实现）")


def _timm_stub():
    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers")
    layers.trunc_normal_ = _trunc_normal_
    layers.DropPath = _DropPath
    timm.models = models
    models.layers = layers
    return {"timm": timm, "timm.models": models, "timm.models.layers": layers}


def _einops_stub():
    einops = types.ModuleType("einops")
    einops.rearrange = _rearrange
    return {"einops": einops}


def _load_paper(filename, alias, stubs=None):
    """按文件名加载 paper/ 下的模块（文件名带空格和括号，不能用普通 import）。"""
    path = os.path.join(PAPER_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 {path}")
    injected = []
    for name, mod in (stubs or {}).items():
        if name not in sys.modules:  # 装了真包就用真包
            sys.modules[name] = mod
            injected.append(name)
    try:
        spec = importlib.util.spec_from_file_location(alias, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module  # 反序列化 best.pt 时要能按名字找到
        with warnings.catch_warnings():
            # timm 1.x 里 paper/28 用的老路径 timm.models.layers 只是弃用、仍然可用，屏蔽这条噪音
            warnings.filterwarnings("ignore", message=".*timm.models.layers is deprecated.*")
            spec.loader.exec_module(module)
    finally:
        for name in injected:
            sys.modules.pop(name, None)
    return module


def _load_with_fallback(filename, alias, stub_factory, dep, what):
    """先按原样加载（用真实第三方包），失败了再注入内置替身，并记录用了替身。

    只有"缺的正是 dep 本身或它的子模块"时才退化成替身（例如装的是老版 timm，
    但 timm.models.layers 这个子模块已经没了），其它错误照常抛出。
    """
    try:
        return _load_paper(filename, alias), None
    except (ModuleNotFoundError, ImportError) as e:
        missing = getattr(e, "name", None)
        if missing is not None and missing != dep and not str(missing).startswith(dep + "."):
            raise
        module = _load_paper(filename, alias, stub_factory())
        return module, f"缺少 {missing or dep}，已用内置等价实现（{what}）；pip install {dep} 可切换官方实现"


# ======================================================================================
# 2. base_modules 语义注册（借壳机制）
# ======================================================================================
# 借壳用的白名单名字。三个条件缺一不可：
#   a) 在 base_modules 里              —— 否则拿不到 [c1, c2, *args] 语义
#   b) 不在 repeat_modules 里          —— 否则 args 会被插入重复次数 n，构造签名直接对不上
#   c) parse_model 里没有身份判断       —— 否则借用期间会把相关分支也改掉
# 下面这 8 个都是"base_modules 有、repeat_modules 没有、无身份判断、官方 cfg/models 里也没人用"的遗留名字。
# 借壳只在 parse_model 调用期间生效，调用结束立刻还原。导入时会自动自检（见 _verify_borrow_slots）。
_BORROW_SLOTS = (
    "Focus",
    "DWConvTranspose2d",
    "ConvTranspose",
    "GhostBottleneck",
    "ELAN1",
    "SPPELAN",
    "RepNCSPELAN4",
    "AConv",
    "DWConv",
    "Bottleneck",
)
_BASE_MODULES = {}  # YAML 名字 -> 适配类
_SLOT_FOR = {}  # YAML 名字 -> 借壳用的白名单名字（注册时固定下来，见下方说明）


def register_base_module(name, cls, note=""):
    """让 YAML 里的 name 享有 base_modules 语义：args 自动变成 [c1, c2_scaled, *其余]。

    ⚠️ 借壳槽位是**在注册时按名字固定**的，不能等到 parse_model 时靠 zip 按顺序配。
       原来写的是 `zip(_BASE_MODULES.items(), _BORROW_SLOTS)` —— 位置配对。于是只要在源码
       中间插入一个新模块（比如 SCPP），它后面所有模块的槽位就整体错位一位：
       NeLU 会被绑到 "DWConv" 这个名字上，于是 YAML 里的 `DWConv` 解析成了 NeLU，
       报 "NeLU.__init__() takes from 1 to 4 positional arguments but 5 were given"。
       （之前没炸纯属巧合：DWConv 恰好是最后一个注册的，正好对上同名的槽位。）
       现在改成：**优先用同名槽位**（DWConv -> "DWConv"），没有同名才取第一个未占用的。
    """
    if len(_BASE_MODULES) >= len(_BORROW_SLOTS):
        raise RuntimeError(f"借壳槽位不够（{len(_BORROW_SLOTS)} 个），需要扩充 _BORROW_SLOTS")
    _BASE_MODULES[name] = cls
    setattr(tasks, name, cls)
    REGISTERED[name] = note
    return cls


def _finalize_slots():
    """全部模块注册完之后，再统一分配借壳槽位（幂等）。

    两轮分配：
      1) 名字本身就是白名单槽位的（目前只有 DWConv）**优先占同名槽位** ——
         否则 YAML 里的 `DWConv` 会解析成别的类；
      2) 其余按注册顺序取剩下的槽位。
    ⚠️ 不能边注册边分配：NeLU 注册早于 DWConv，若那时就取"第一个空闲槽位"，
       它会抢走 "DWConv" 这个名字，DWConv 只能退到 "Bottleneck"。
    """
    if _SLOT_FOR:
        return
    remaining = list(_BORROW_SLOTS)
    for name in _BASE_MODULES:
        if name in remaining:
            _SLOT_FOR[name] = name
            remaining.remove(name)
    for name in _BASE_MODULES:
        if name not in _SLOT_FOR:
            _SLOT_FOR[name] = remaining.pop(0)
    _verify_borrow_slots()  # 定完槽位立刻自检（导入期那次 _BASE_MODULES 还是空的）


def _verify_borrow_slots():
    """自检借壳名字：必须"在 base_modules 且不在 repeat_modules"。源码读不到就跳过。"""
    try:
        import inspect
        import re

        src = inspect.getsource(tasks.parse_model)

        def _names(block):
            i = src.index(f"{block} = frozenset(")
            j = src.index("\n    )", i)
            return set(re.findall(r"\b([A-Z][A-Za-z0-9_]*)\b", src[i:j]))

        base, repeat = _names("base_modules"), _names("repeat_modules")
    except Exception:
        return
    bad = [s for s in _BORROW_SLOTS if s not in base or s in repeat]
    if bad:
        raise RuntimeError(
            f"借壳槽位 {bad} 不再安全（当前 ultralytics 里必须满足：在 base_modules、不在 repeat_modules）。"
            f"请修改 custom_modules._BORROW_SLOTS。"
        )
    # 自检二：槽位不能重复占用，且同名的模块必须用到同名槽位（否则 YAML 名字会解析成别的类）
    used = list(_SLOT_FOR.values())
    dup = {s for s in used if used.count(s) > 1}
    if dup:
        raise RuntimeError(f"借壳槽位被重复占用：{dup}（YAML 里的模块名会解析成错误的类）")
    wrong = {n: s for n, s in _SLOT_FOR.items() if n in _BORROW_SLOTS and s != n}
    if wrong:
        raise RuntimeError(f"同名槽位没有优先使用：{wrong} —— YAML 里的 {list(wrong)} 会解析不到自己的类")


def _install_base_module_patch():
    """包一层 parse_model：调用期间把自定义类临时绑到借壳名字上，使它们进入 base_modules。"""
    original = tasks.parse_model
    if getattr(original, "_custom_base_module_patch", False):
        return

    def parse_model(d, ch, verbose=True):
        _finalize_slots()  # 懒分配：确保全部注册完成后再定槽位
        saved = []
        for name, cls in _BASE_MODULES.items():
            slot = _SLOT_FOR[name]  # 注册时固定，不靠位置配对（见 register_base_module 的说明）
            if slot in tasks.__dict__:
                saved.append((slot, tasks.__dict__[slot]))
                tasks.__dict__[slot] = cls
        try:
            return original(d, ch, verbose)
        finally:
            for slot, old in saved:
                tasks.__dict__[slot] = old

    parse_model._custom_base_module_patch = True
    parse_model.__doc__ = original.__doc__
    tasks.parse_model = parse_model


_finalize_slots()
_verify_borrow_slots()
_install_base_module_patch()


def _same_channels(name, c1, c2):
    if c1 != c2:
        raise ValueError(
            f"{name} 是通道保持模块，但 YAML 声明的输出通道 {c2} != 输入通道 {c1}；"
            f"请把 args[0]（名义通道）改成与 {c1} 对应的值"
        )


# ======================================================================================
# 3. 八个论文模块的适配层 + 注册
# ======================================================================================
# ---- 108 (ASOC 2025) PCDM / Pool-Conv-Down：下采样算子 ----
try:
    _pcdm = _load_paper("108.0 (ASOC 2025) PCDM.py", "_paper_pcdm")

    class PCDM(nn.Module):
        """池化(保留原通道) + 卷积(只算新增通道) 的双分支下采样。约束：c2 > c1；H/W 都是偶数。"""

        def __init__(self, c1, c2):
            super().__init__()
            if c2 <= c1:
                raise ValueError(f"PCDM 要求输出通道 > 输入通道，收到 c1={c1}, c2={c2}")
            self.m = _pcdm.PoolConvDownSampling(c1, c2)

        def forward(self, x):
            if x.shape[-1] % 2 or x.shape[-2] % 2:
                raise ValueError(f"PCDM 要求输入 H/W 都是偶数，收到 {tuple(x.shape[-2:])}")
            return self.m(x)

    register_base_module("PCDM", PCDM, "108 ASOC2025 池化卷积下采样（c2>c1，H/W 需偶数）")
except Exception as e:  # pragma: no cover
    SKIPPED["PCDM"] = f"{type(e).__name__}: {e}"

# ---- 19 (CVPR 2024) InceptionNeXt / InceptionDWConv2d：卷积特征块 ----
try:
    _idconv = _load_paper("19.0 (CVPR 2024) IDConv.py", "_paper_idconv")

    class IDConv(nn.Module):
        """Inception 深度卷积：通道四切分（直通 + 3x3 深度 + 1x11 + 11x1 带状）。通道保持。"""

        def __init__(self, c1, c2, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
            super().__init__()
            _same_channels("IDConv", c1, c2)
            if int(c1 * branch_ratio) < 1:
                raise ValueError(f"IDConv 通道太少（c1={c1}, branch_ratio={branch_ratio}），请调大通道或 branch_ratio")
            self.m = _idconv.InceptionDWConv2d(c1, square_kernel_size, band_kernel_size, branch_ratio)

        def forward(self, x):
            return self.m(x)

    register_base_module("IDConv", IDConv, "19 CVPR2024 InceptionNeXt 深度卷积")
except Exception as e:  # pragma: no cover
    SKIPPED["IDConv"] = f"{type(e).__name__}: {e}"

# ---- 28 (CVPR 2025) MambaOut / GatedCNNBlock：卷积特征块（原版 NHWC） ----
try:
    _gated, _gated_note = _load_with_fallback(
        "28.0 (CVPR 2025) GatedCNNBlock.py", "_paper_gated", _timm_stub, "timm", "DropPath"
    )

    class GatedCNNBlock(nn.Module):
        """门控 CNN 块（MambaOut）。原版输入 NHWC，这里包了 NCHW <-> NHWC 转换。通道保持。"""

        def __init__(self, c1, c2, kernel_size=7, expansion_ratio=8 / 3, conv_ratio=1.0, drop_path=0.0):
            super().__init__()
            _same_channels("GatedCNNBlock", c1, c2)
            self.m = _gated.GatedCNNBlock(
                dim=c1,
                expansion_ratio=expansion_ratio,
                kernel_size=kernel_size,
                conv_ratio=conv_ratio,
                drop_path=drop_path,
            )

        def forward(self, x):
            x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
            x = self.m(x)
            return x.permute(0, 3, 1, 2)  # NHWC -> NCHW

    register_base_module("GatedCNNBlock", GatedCNNBlock, "28 CVPR2025 MambaOut 门控块")
    if _gated_note:
        REGISTERED["GatedCNNBlock"] += f" [{_gated_note}]"
except Exception as e:  # pragma: no cover
    SKIPPED["GatedCNNBlock"] = f"{type(e).__name__}: {e}"

# ---- 62 (TPAMI 2024) CamoFormer / MaskedSeparableAttention：通道注意力 ----
try:
    _msa, _msa_note = _load_with_fallback(
        "62.0 (TPAMI 2024) MaskedSeparableAttention.py", "_paper_msa", _einops_stub, "einops", "rearrange"
    )

    class MSA(nn.Module):
        """掩码可分离注意力：注意力图是 CxC（对分辨率线性），不是空间注意力。通道保持。"""

        def __init__(self, c1, c2, num_heads=8, bias=True):
            super().__init__()
            _same_channels("MSA", c1, c2)
            if c1 % num_heads:
                raise ValueError(f"MSA 要求通道数能被 num_heads 整除，收到 c1={c1}, num_heads={num_heads}")
            self.m = _msa.MaskedSeparableAttention(c1, num_heads, bias)

        def forward(self, x):
            return self.m(x)

    register_base_module("MSA", MSA, "62 TPAMI2024 CamoFormer 掩码可分离注意力")
    if _msa_note:
        REGISTERED["MSA"] += f" [{_msa_note}]"
except Exception as e:  # pragma: no cover
    SKIPPED["MSA"] = f"{type(e).__name__}: {e}"

# ---- 65 (Elsevier 2025) Flora-NET / CoordAttMeanMax：坐标注意力 ----
# ⚠️ 原版内部用 adaptive pool 实现「沿 H 或 W 的归约」：
#       AdaptiveAvgPool2d((None, 1)) / AdaptiveMaxPool2d((None, 1))   -> 沿 W 归约，保留 H
#       AdaptiveAvgPool2d((1, None)) / AdaptiveMaxPool2d((1, None))   -> 沿 H 归约，保留 W
#   而 adaptive_avg_pool2d_backward_cuda / adaptive_max_pool2d_backward_cuda 在 torch 里
#   **没有确定性实现**，开了 torch.use_deterministic_algorithms(True, warn_only=True) 之后会刷：
#       UserWarning: adaptive_max_pool2d_backward_cuda does not have a deterministic implementation...
#   实测它的"非确定性"只出现在不参与反传的 inds（int64 argmax 索引）上，grad_input 逐位可复现，
#   所以是无害提示。**保持原版实现不动**，警告由 experiments/common.py 的
#   silence_adaptive_pool_warning() 过滤（train(suppress_adpool_warning=True)，默认开）。
#
#   曾经试过把这 4 个 pool 换成等价的 dim-reduce（mean / amax）来消灭警告，已按原样回退：
#   收益只是日志干净，代价是 fp32 下前向不再逐位相同（相对差 ~1.5e-7）、以及并列极值时
#   max 分支的梯度分布从"单点"变成"均分"（前向一致、行内总和一致，但确实是语义变化）。
#   不划算，所以回到"原版实现 + 过滤日志"。
try:
    _dcafe = _load_paper("65.0 (Elsevier 2025) DCAFE.py", "_paper_dcafe")

    class DCAFE(nn.Module):
        """双坐标注意力（mean/max 双分支）。原版签名 (inp, oup) 有误导，实际必须相等。通道保持。"""

        def __init__(self, c1, c2, groups=32):
            super().__init__()
            _same_channels("DCAFE", c1, c2)
            self.m = _dcafe.CoordAttMeanMax(c1, c2, groups)

        def forward(self, x):
            return self.m(x)

    register_base_module("DCAFE", DCAFE, "65 Elsevier2025 Flora-NET 坐标注意力")
except Exception as e:  # pragma: no cover
    SKIPPED["DCAFE"] = f"{type(e).__name__}: {e}"

# ---- 86 (TGRS 2025) IIA：方向注意力 ----
try:
    _iia = _load_paper("86.0 (TGRS 2025) IIA.py", "_paper_iia")

    class IIA(nn.Module):
        """信息集成注意力：H/W 两方向各做一次 1D 注意力后相加。通道保持、成本最低。

        实现是 x + x_h + x_w（论文版带 1/2 系数，原文件里被注释掉了），输出幅度约放大 2 倍。
        """

        def __init__(self, c1, c2):
            super().__init__()
            _same_channels("IIA", c1, c2)
            self.m = _iia.IIA(c1)

        def forward(self, x):
            # paper/86 内部的 depthwise Conv1d(c,c,k=7,groups=c)，在 autocast(fp16) + cuDNN
            # 启发式（cudnn.benchmark=False，训练默认值）下会挑到坏 kernel：
            #   "GET was unable to find an engine to execute this computation" / illegal memory access。
            # 所以这里只在 IIA 内部关掉 autocast，让它按权重的本征 dtype 计算。
            # ⚠️ 只关 autocast 是不够的：上层 autocast 产出的激活是 fp16，而 AMP 下权重仍是 fp32，
            #    两者混用会直接抛 "Input type (struct c10::Half) and bias type (float) should be the same"。
            #    必须先按权重 dtype 对齐输入，算完再换回原 dtype，对外部完全透明。
            orig_dtype = x.dtype
            w_dtype = next(self.m.parameters(), torch.empty(0)).dtype
            with torch.autocast(device_type=x.device.type, enabled=False):
                out = self.m(x.to(w_dtype))
            return out.to(orig_dtype)

    register_base_module("IIA", IIA, "86 TGRS2025 信息集成注意力（内部按权重 dtype 计算，避开 cuDNN fp16 坏 kernel）")
except Exception as e:  # pragma: no cover
    SKIPPED["IIA"] = f"{type(e).__name__}: {e}"

# ---- 87 (TPAMI 2025) LRSA：低分辨率自注意力（原版接口 (B,N,C) + 外部 H/W/d_convs） ----
try:
    _lrsa = _load_paper("87.0 (TPAMI 2025) LRSA.py", "_paper_lrsa")

    class LRSA(nn.Module):
        """低分辨率自注意力。原版 forward 需要 (B,N,C) 和外部 H/W/d_convs，这里包成 NCHW 层。

        pooled_sizes 是 K/V 金字塔池化的目标边长（论文场景是 32x32 特征图）。
        """

        def __init__(self, c1, c2, num_heads=4, pooled_sizes=(11, 8, 6, 4), q_pooled_size=1):
            super().__init__()
            _same_channels("LRSA", c1, c2)
            if c1 % num_heads:
                raise ValueError(f"LRSA 要求通道数能被 num_heads 整除，收到 c1={c1}, num_heads={num_heads}")
            self.pooled_sizes = tuple(pooled_sizes)
            self.m = _lrsa.LRSA(
                dim=c1,
                num_heads=num_heads,
                qkv_bias=True,
                pooled_sizes=list(self.pooled_sizes),
                q_pooled_size=q_pooled_size,
                q_conv=False,
            )
            # 原版要求调用方给每个池化尺度准备一个深度卷积
            self.d_convs = nn.ModuleList(nn.Conv2d(c1, c1, 3, padding=1, groups=c1) for _ in self.pooled_sizes)

        def forward(self, x):
            b, c, h, w = x.shape
            y = self.m(x.flatten(2).transpose(1, 2).contiguous(), h, w, self.d_convs)
            return y.transpose(1, 2).reshape(b, c, h, w)

    register_base_module("LRSA", LRSA, "87 TPAMI2025 低分辨率自注意力")
except Exception as e:  # pragma: no cover
    SKIPPED["LRSA"] = f"{type(e).__name__}: {e}"

# ---- 131 (TGRS 2025) GLVMamba / SCPP：尺度感知金字塔池化（替代 SPPF） ----
# ⚠️ 论文版 forward 返回 **(out, meta_dict) 元组**：
#       return out, {"scale_fused": ..., "local": ..., "global": ..., "attn": attn}
#    而 ultralytics 的 _predict_once 是 `x = m(x)` 把返回值直接往下传 —— 返回 tuple 会让
#    下游的 Concat/Conv 全部报错（实测就是 "Sizes of tensors must match except in dimension 1"）。
#    所以适配层**只取第 0 个元素**，meta 丢弃。
#
# 通道语义与 SPPF 相同：args[0] 是输出名义通道（走 width 缩放），所以可以一对一替换 SPPF。
# YAML 写法：
#     [-1, 1, SCPP, [512]]                # c2 = make_divisible(min(512,max_ch)*width, 8)
#     [-1, 1, SCPP, [512, 16, [1, 3, 5]]] # 显式给 se_reduction 与 dilation_rates
#
# 成本提示（实测，n 档）：SCPP 比 SPPF 贵得多 —— 3 个全通道 3x3 膨胀卷积 + 3 个 1x1 +
#     SE + 全局分支 + 3C->C 的输出 1x1。放 P5（20x20）时参数量是 SPPF 的 4 倍量级，见运行日志。
try:
    _scpp = _load_paper("131.0 (TGRS 2025) SCPP.py", "_paper_scpp")

    class SCPP(nn.Module):
        """尺度感知金字塔池化：多尺度膨胀卷积 + SE + 逐像素 softmax 权重融合。输出通道可缩放。

        ⚠️ 内置「先压通道」：SCPP 的成本是 **通道平方级** 的（3 个全通道 3x3 膨胀卷积 + 3 个 1x1
           + SE + 全局分支 + 3C->C 输出）。实测在 256 通道上单层就是 **2,224,899 参数**
           —— 是它替换掉的 SPPF(98,816) 的 22.5 倍、占官方基线整个模型的 93%。
           所以在 c1 > mid 时先用一个 1x1 把通道压到 mid(默认 64)，再进论文版 SCPP：
           同样位置从 2,224,899 降到 ~175K，而多尺度融合那套机制完全保留。
           压完再进，等于把「通道变换」折叠进了这一层（YAML 里可以少写一层）。
           想让它吃原始通道数就把 mid 设大（或传 mid=0 关闭）。
        """

        def __init__(self, c1, c2, se_reduction=16, dilation_rates=(1, 3, 5), mid=64):
            super().__init__()
            self.pre = None
            inner_c1 = c1
            if mid and c1 > mid:
                self.pre = nn.Sequential(
                    nn.Conv2d(c1, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU()
                )
                inner_c1 = mid
            self.m = _scpp.SCPP(inner_c1, c2, se_reduction=se_reduction,
                                dilation_rates=tuple(dilation_rates))

        def forward(self, x):
            if self.pre is not None:
                x = self.pre(x)
            return self.m(x)[0]  # 论文版返回 (out, meta)，只取张量

    register_base_module("SCPP", SCPP, "131 TGRS2025 GLVMamba 尺度感知金字塔池化（替代 SPPF，内置 1x1 压通道）")
except Exception as e:  # pragma: no cover
    SKIPPED["SCPP"] = f"{type(e).__name__}: {e}"

# ---- 79 (arXiv 2025) NeLU：激活函数 ----
try:
    _nelu = _load_paper("79.0 (arXiv 2025) NeLU.py", "_paper_nelu")

    class NeLU(nn.Module):
        """负半轴有界响应 −α/(1+x²) 替代 ReLU 的 0。0 参数、通道保持。

        两种用法：
          - 全局激活：模型 YAML 顶层写 activation: NeLU()
          - 当层用：  [-1, 1, NeLU, [512]]
        """

        def __init__(self, c1=None, c2=None, alpha=0.2):
            super().__init__()
            if c1 is not None and c2 is not None:
                _same_channels("NeLU", c1, c2)
            self.m = _nelu.NeLU(alpha=alpha)

        def forward(self, x):
            return self.m(x)

    tasks.NeLU = NeLU  # 让 YAML 的 activation: NeLU() 能被 eval 到
    register_base_module("NeLU", NeLU, "79 arXiv2025 激活函数（也可 activation: NeLU()）")
except Exception as e:  # pragma: no cover
    SKIPPED["NeLU"] = f"{type(e).__name__}: {e}"


# ======================================================================================
# 4. 之前的三个：CBAM（新名字注入）、C3k2CBAM（同名替换）、MobileNetV3 主干
# ======================================================================================
tasks.CBAM = CBAM
REGISTERED["CBAM"] = "官方 CBAM（新名字注入：通道写缩放后的实际值）"


class C3k2CBAM(C3k2):
    """C3k2 + CBAM：结构与 C3k2 一致，只在输出上叠一层通道/空间注意力。"""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.attention = CBAM(c2)

    def forward(self, x):
        return self.attention(super().forward(x))


# 需要"带注意力的 C3k2"时打开：同名替换（主干+neck 里所有 C3k2 都会变），白名单语义自动缩放通道。
# tasks.C3k2 = C3k2CBAM
REGISTERED.setdefault("C3k2CBAM", "同名替换写法（默认未启用，取消注释 tasks.C3k2 = C3k2CBAM）")


class MobileNetV3Backbone(nn.Module):
    """torchvision MobileNetV3 主干，一次输出 (P3/8, P4/16, P5/32)，通道统一投影到 c3/c4/c5。

    YAML 写法（见 cfg/yolo11n-seg-tea-mnv3.yaml）：

        - [-1, 1, MobileNetV3Backbone, [large, DEFAULT, 64, 128, 256]]
        - [0, 1, Index, [64, 0]]     # P3（from 写 0，指向主干那一行）
        - [0, 1, Index, [128, 1]]    # P4
        - [0, 1, Index, [256, 2]]    # P5

    stride 8/16/32 对应的层号在构造时用假图探出来，换 torchvision 版本 / variant 都不用改代码。
    """

    def __init__(self, variant="large", weights="DEFAULT", c3=64, c4=128, c5=256):
        super().__init__()
        import torchvision

        if weights in ("", "None", "none", None):
            weights = None
        net = torchvision.models.get_model(f"mobilenet_v3_{variant}", weights=weights)
        self.features = net.features
        self.variant = variant
        self.out_indices, in_chs = self._probe_strides()
        self.proj = nn.ModuleList(Conv(c, o, 1, 1) for c, o in zip(in_chs, (c3, c4, c5)))
        self.in_channels = tuple(in_chs)
        self.out_channels = (c3, c4, c5)

    @torch.no_grad()
    def _probe_strides(self, size=128):
        """探出 stride=8/16/32 各自对应的最后一个 features 下标与通道数。"""
        was_training = self.training
        self.eval()  # 避免 BN 的 running stats 被假数据污染
        x = torch.zeros(1, 3, size, size)
        picks = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            for s in (8, 16, 32):
                if x.shape[-1] * s == size:
                    picks[s] = (i, x.shape[1])
        self.train(was_training)
        missing = [s for s in (8, 16, 32) if s not in picks]
        if missing:
            raise ValueError(f"MobileNetV3-{self.variant} 没有 stride={missing} 的特征层，无法接 YOLO head")
        return [picks[s][0] for s in (8, 16, 32)], [picks[s][1] for s in (8, 16, 32)]

    def forward(self, x):
        want = set(self.out_indices)
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in want:
                feats.append(x)
        return tuple(p(o) for p, o in zip(self.proj, feats))


tasks.MobileNetV3Backbone = MobileNetV3Backbone
REGISTERED["MobileNetV3Backbone"] = "torchvision MobileNetV3 主干 -> (P3,P4,P5)"


# ======================================================================================
# 4.1 DWConv：深度可分离卷积（1x1 点卷积 + kxk 深度卷积），base_modules 语义
# ======================================================================================
# 用途：YAML 里的「通道变换」层。C3k2 允许 c1 != c2（主干 2x 加宽、neck 压缩通道），
# 而 IDConv 是通道保持的，所以把 C3k2 换成 IDConv 时必须先补一层通道变换。
# 用普通 1x1 Conv 的话，在 P2/4 这种高分辨率上很贵（例如 256->128 @80x80 单层就 ~0.4 GFLOPs），
# 换成「1x1 点卷积 + 3x3 深度卷积」在同样通道数下便宜得多，且多一层空间混合。
#
# YAML 写法（args[0] 与 Conv 一致，走 width 缩放，同一份配置跨 n/s/m/l/x 安全）：
#     [-1, 1, DWConv, [512, 3, 2]]   # 名义 c2=512、k=3、stride=2
#     [-1, 1, DWConv, [256, 1, 1]]   # 纯通道变换（等价 1x1 Conv，k=1 时深度分支退化）
#
# 实测（n 档，名义 512 -> 实际 128）：
#     Conv1x1[512,1,1]  vs  DWConv[512,3,2]  在 160x160 输入上的 FLOPs 差约 3 倍
class DWConv(nn.Module):
    """深度可分离卷积：1x1 点卷积（先降/升通道）+ kxk 深度卷积。

    顺序是「先点卷积再深度卷积」，为的是让深度卷积的输入通道数 = c2（与 Conv 的输出语义一致）。
    实现上用 native nn.Conv2d，避免引入 ultralytics 内部 Conv 的耦合。

    args 语义与白名单 Conv 完全一致：c1 是真实输入通道（c2 缩放后），c2 由 parse_model 传进来。
    """

    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        p = k // 2
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.dw = nn.Conv2d(c2, c2, k, s, p, groups=c2, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn2(self.dw(self.bn1(self.pw(x)))))


register_base_module("DWConv", DWConv, "深度可分离卷积 (1x1 点卷积 + kxk 深度)，args[0] 走 width 缩放")


# ======================================================================================
# 4.2 把 Detect / Segment 头里的 3x3 卷积换成深度可分离卷积
# ======================================================================================
# 动机（实测，n 档 640 输入）：
#     head 占全网 FLOPs 的 80%（body 只有 20%），其中 cv2（框回归分支）= 6.50 GFLOPs，
#     占全网 35%。而 cv2/cv4 里的两个 3x3 Conv 都是「先扩到 c2、再 3x3」的写法：
#         Conv(x, c2, 3) -> Conv(c2, c2, 3) -> Conv2d(c2, out, 1)
#     第二层 3x3 同理。按 in/out 通道数相同时的 3x3 FLOPs 核算：
#         x=64 -> c2=64：3x3 = 0.59 G，DW(1x1 64->64 + 3x3 dw) = 0.07 G，省 8x
#         x=64 -> c2=64（第二层）同上
#     所以把这两个 3x3 换成 DWConv，cv2 大约能从 6.50 G 降到 3 G 左右。
#
# 实现方式：**结构级重建**，不是事后改模块。
#   - 只重建 cv2 / cv3 / cv4（以及 end2end 的 one2one 副本），proto 与 dfl 不动；
#   - 复刻官方 head 的通道推导：c2 = max(16, ch[0]//4, reg_max*4)，c3 = max(ch[0], min(nc,100))，
#     c4 = max(ch[0]//4, nm)，最后一个 1x1 输出维度保持不变（4*reg_max / nc / nm）；
#   - 重建后重新初始化权重，尽量贴近「一开始就这么建」的分布。
# 官方本来就有这种轻量 head（ultralytics 的 Detect 在 legacy=False 时 cv3 就用 DWConv），
# 所以这不是"魔改"，只是把这个思路推到 cv2/cv4 上。
import copy as _copy

from ultralytics.nn.modules import Conv as _UltraConv
from ultralytics.nn.modules.head import DFL as _UltraDFL
from ultralytics.utils.torch_utils import initialize_weights as _init_weights


def _dw_seq(c1: int, c2: int, k: int, out_ch: int, out_k: int = 1):
    """3x3 换成 DWConv，再拼上原来的 1x1 输出层（结构与官方 head 一一对应）。"""
    return nn.Sequential(DWConv(c1, c2, k), DWConv(c2, c2, k), nn.Conv2d(c2, out_ch, out_k))


def _last_conv_of(seq) -> nn.Conv2d:
    """取出某个分支里最后一个 Conv2d（用来读输出维度）。"""
    for m in reversed(list(seq.modules())):
        if isinstance(m, nn.Conv2d):
            return m
    raise RuntimeError("分支里没有 Conv2d")


def make_head_depthwise(model) -> dict:
    """把 model 里所有 Detect/Segment 头的 3x3 分支换成深度可分离，返回改动统计。

    用法：建好模型后调用一次，之后再训练/验证。对已经改过的模型重复调用是幂等的
    （重建出来的分支里已经没有 ultralytics 的 Conv 了，检测不到就跳过）。
    """
    stats = {"heads": 0, "branches": 0, "params_before": 0, "params_after": 0}
    for layer in model.modules():
        # 只处理带 cv2 的头（Detect / Segment / OBB / Pose 都是这个套路）
        if not (hasattr(layer, "cv2") and hasattr(layer, "nl") and hasattr(layer, "reg_max")):
            continue
        # 已经处理过就跳过：重建后的分支第一个模块是 DWConv（我们自己注册的那个）
        first = layer.cv2[0][0]
        if isinstance(first, DWConv):
            continue

        ch = [blk[0].conv.in_channels for blk in layer.cv2]
        reg_max = layer.reg_max
        c2 = max(16, ch[0] // 4, reg_max * 4)
        stats["params_before"] += sum(p.numel() for p in layer.parameters())

        # 框回归分支 cv2
        layer.cv2 = nn.ModuleList(
            _dw_seq(x, c2, 3, 4 * reg_max) for x in ch
        )
        stats["branches"] += 1

        # 分类分支 cv3（legacy=True 时是普通 3x3；legacy=False 时本来就是 DWConv，跳过）
        if hasattr(layer, "cv3") and layer.cv3 is not None and getattr(layer, "legacy", False):
            nc = layer.nc
            c3 = max(ch[0], min(nc, 100))
            layer.cv3 = nn.ModuleList(nn.Sequential(DWConv(x, c3, 3), DWConv(c3, c3, 3), nn.Conv2d(c3, nc, 1)) for x in ch)
            stats["branches"] += 1
        if hasattr(layer, "one2one_cv3") and getattr(layer, "legacy", False):
            layer.one2one_cv3 = _copy.deepcopy(layer.cv3)

        # 掩码系数分支 cv4（Segment 专有）
        if hasattr(layer, "cv4") and layer.cv4 is not None:
            nm = layer.nm
            c4 = max(ch[0] // 4, nm)
            layer.cv4 = nn.ModuleList(nn.Sequential(DWConv(x, c4, 3), DWConv(c4, c4, 3), nn.Conv2d(c4, nm, 1)) for x in ch)
            stats["branches"] += 1

        if hasattr(layer, "one2one_cv2"):
            layer.one2one_cv2 = _copy.deepcopy(layer.cv2)
        if hasattr(layer, "one2one_cv4") and hasattr(layer, "cv4"):
            layer.one2one_cv4 = _copy.deepcopy(layer.cv4)

        # 重建出来的分支需要重新初始化（官方在模型 build 完会跑 initialize_weights）
        layer.apply(_init_weights)
        # ⚠️ 还必须把官方的 bias_init() 重做一遍 —— 漏了它 cls_loss 会直接爆三个数量级。
        #    依据：
        #      tasks.py:498        m.bias_init()   # "only run once"，只在 SegmentationModel.__init__ 里调
        #      head.py:218-224     cv2[-1].bias[:] = 2.0
        #                          cv3[-1].bias[:nc] = log(5 / nc / (640 / stride) ** 2)
        #                                              （nc=4 / stride=8 时 = -8.54）
        #      torch_utils.py:562  initialize_weights() 里 `if t is nn.Conv2d: pass`
        #                          （kaiming 那行是注释掉的）—— 它**不碰 Conv2d 的 bias**，补不回来。
        #    我们是在 __init__ 之后重建分支的，新建 Conv2d 的 bias 是 PyTorch 默认值(≈0)：
        #      cls 头初始 sigmoid 从 2e-4 变成 0.5 -> BCE 每项从 ~2e-4 涨到 0.693（~3400 倍）
        #      实测症状：epoch 1 的 cls_loss 从正常的 ~5.5 变成 ~6850
        #      box 头 bias 从 2.0 变成 0 -> 初始框尺度全错，指派器前期几乎找不到正样本
        #    bias_init 只写 bias，幂等，重复调用无副作用。
        if hasattr(layer, "bias_init"):
            layer.bias_init()
        stats["heads"] += 1
        stats["params_after"] += sum(p.numel() for p in layer.parameters())
    return stats


# ======================================================================================
# 5. 多检测头（多尺度输出）时修正每个 head 的 stride
# ======================================================================================
# 起因：tasks.py 里 DetectionModel.__init__ 是这样推算 stride 的
#     m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])
# 其中 _forward 返回「最后一个 head 记录的每一层特征图」。只要 YAML 里出现两个 head
# （例如本仓库 TeaSeg-S 的「P2 辅助 Detect + P3/P4/P5 主 Segment」），hook 会把两个 head
# 的输入混在一起返回，于是：
#   1) 个数对不上（实测 [0.0, 8.0, 8.0, 16.0, 32.0]），多出来的 0.0 会让
#      make_anchors / TaskAlignedAssigner 拿到 stride=0，anchor 全变 0；
#   2) 前面那个 head（P2 辅助头）压根没进 _forward，它的 stride 一直是初值 [32]，而它
#      实际是 stride=4 —— 训练时 anchor 网格会错 8 倍。
#
# 另外：当这个傻推算过程本身抛异常（size 太小、显存不够等）时，tasks.py 会留下
#     self.stride, m.stride = 8, 32   # 与 head 个数无关的常量默认值
# 这也是多 head 场景下要兜底的情形，所以判定条件要同时看「长度」和「全为正」。
#
# 修法（只在真的不对时才做，单 head 模型完全不受影响）：
#   a) 从「最后一个记录到特征图的 head」的所有卷积层开 forward hook，记录每个下采样层
#      (stride, 输出边长)，用 h * s == size 这个不变式反推所有存在的 stride（探测边长取 1024：
#      P2/4 的特征图此时是 256x256，不会像 size=256 那样把 stride=2 的浅层也算进来）；
#   b) 再开一次 hook 记录「每个 head 的输入特征图通道数」，通过「第 i 层输出正好是第 i+1 层
#      输入」把通道数映射到 stride；
#   c) 给每个 head 单独写它自己那几个尺度的 stride。
def _patch_stride_inference():
    model = tasks.DetectionModel
    original = model.__init__
    if getattr(original, "_multi_head_stride_patch", False):
        return

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
        original(self, cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        net = self.model
        heads = [x for x in net if hasattr(x, "stride") and hasattr(x, "nl")]
        if not heads:
            return
        if all(len(h.stride) == h.nl and bool((h.stride > 0).all()) for h in heads):
            return  # 单 head（含官方 yolo11-seg）走这条，什么都不做

        # 探测边长：优先用 tasks.py 已经探过的尺寸（可从遗留的 stride 反推），否则用 1024
        size = 0
        for h in heads:
            s = torch.as_tensor(h.stride).flatten()
            if len(s) == 1 and 8 <= float(s[0]) <= 64:
                size = int(float(s[0]) * 16)
        size = size or 1024

        downs = []  # (stride, 输出边长, 输出通道)
        head_sizes = []  # 每个 head 的输入特征图边长（可能多个尺度）

        def _conv_hook(mod, inp, out):
            if mod.stride[0] > 1:
                downs.append((int(mod.stride[0]), int(out.shape[-2]), int(out.shape[1])))

        def _head_pre_hook(mod, inp):
            feats = inp[0]
            head_sizes.append([int(t.shape[-2]) for t in feats] if isinstance(feats, (list, tuple)) else [int(feats.shape[-2])])

        handles = [
            mod.register_forward_hook(_conv_hook) for mod in net.modules() if isinstance(mod, torch.nn.Conv2d)
        ]
        handles += [layer.register_forward_pre_hook(_head_pre_hook) for layer in net if hasattr(layer, "nl")]

        was_training = net.training
        try:
            net.eval()
            with torch.no_grad():
                self.forward(torch.zeros(1, ch, size, size))
        except Exception as e:  # 显存不够等：放弃修正，但明确告知
            warnings.warn(f"custom_modules: 多 head stride 修正失败（{type(e).__name__}: {e}），请检查模型结构")
            return
        finally:
            net.train(was_training)
            for hd in handles:
                hd.remove()

        # 每个特征图边长 -> stride。用「该 stride 真的被某个下采样层用到过」来过滤：
        # 下采样 strided Conv 的 stride 就是 s，所以 {size // h : h 是所有 stride>=2 卷积的输出边长}
        # 正好是网络里存在的 stride 集合；FPN 里上采样出来的边长也算得对。
        # 这样 strided Conv 之前那一层原图尺寸（size 本身）不会被误当成 stride=1 的金字塔层。
        real_strides = {size // h for _, h, _ in downs}
        for h, szs in zip(heads, head_sizes):
            # head 内部特征图是从小到大排列的（P3 -> P5），stride 按同样顺序写
            strides = [size // sz for sz in szs if (size // sz) in real_strides]
            h.stride = torch.tensor([float(x) for x in strides])
        self.stride = heads[-1].stride
        if not all(len(h.stride) == h.nl and bool((h.stride > 0).all()) for h in heads):
            warnings.warn(
                f"custom_modules: 多 head stride 修正结果仍不合法（head_sizes={head_sizes}, downs={downs}），请检查模型结构"
            )

    __init__._multi_head_stride_patch = True
    model.__init__ = __init__


_patch_stride_inference()


# ======================================================================================
# 5.5 训练期重建钩子：model.train() 会按 yaml 把模型整个重建，必须把结构改动重做上去
# ======================================================================================
# 起因（实测，这个坑吞掉了三次训练的改动，是全仓库代价最大的一个 bug）：
#
#   engine/model.py:862-867 在 Model.train() 里做
#       if not args.get("resume") and self.ckpt:
#           weights = None if pretrained is False else self.model
#           self.trainer.model = self.trainer.get_model(weights=weights, cfg=self.model.yaml)
#   而 trainer.get_model() 是**按 yaml 字典重新建一个全新模型**，再 load(weights) 按 **key 名**
#   把旧模型的权重搬过去（实测输出 "Transferred 275/455 items from pretrained weights"）。于是：
#       - 权重类改动（按模块语义迁移出来的预训练权重）能过去，因为键名相同；
#       - **结构类改动过不去** —— 重建 head 的 cv3、把 cv2/cv4 换成深度可分离，全部被丢掉。
#
#   更糟的一条：用 `YOLO("xxx.yaml")` 起步时 `self.ckpt == {}`（falsy，见 model.py:113/255），
#   上面那个 if 根本不进，`trainer.model` 保持 BaseTrainer.__init__ 设的 **yaml 路径字符串**；
#   接着 _setup_train -> setup_model() 看到 self.model 既不是 nn.Module 也不是 .pt，
#   weights 就是 None —— **连预训练权重一起丢光，等于纯随机初始化在训**。
#   （setup_model 在 engine/trainer.py:814，第一行就是
#    `if isinstance(self.model, torch.nn.Module): return`。）
#
#   实测后果：experiments/runs/*/weights/best.pt 里保存的全是「没改过、也没热启动」的结构：
#       baseline_yolo11n_seg  2,843,388 参数 / head   684,220  cv2首层=Conv（期望 2,382,780 / 223,612）
#       teaseg_s              2,683,623 参数 / head   918,404  cv2首层=Conv（期望 2,047,015 / 281,796）
#       teaseg_s_idconv       2,232,119 参数 / head 1,367,300  cv2首层=Conv（期望 1,146,615 / 281,796）
#   日志里的 [legacy] / [head] / [transfer] 全是真的（当时确实改上了），几分钟后被 train() 重建掉。
#
# 修法（三步，缺一不可）：
#   a) 本函数：给每个 trainer 实例包一层 get_model，建完模型后重放登记过的「构建后处理」。
#      不在类级别打补丁：ultralytics 里 detect / segment / pose / obb / classify / ... **每个 task 的
#      trainer 都自己重写了 get_model**（SegmentationTrainer 就不继承 DetectionTrainer 的实现），
#      只补基类会漏；在 BaseTrainer.__init__ 末尾装**实例属性**最省事且覆盖所有子类。
#   b) experiments/common.py 里设 `model.ckpt = model.model`，让 Model.train() 走进上面那个 if，
#      用我们改好的模型当 weights（键名匹配 -> 权重保留），再由 (a) 把结构改动补回去。
#   c) 再配一个 on_train_start 校验：trainer 真正要训的模型参数量必须等于我们建模后的参数量，
#      对不上直接抛错 —— 免得又白跑两小时才发现改动没生效。
def _log(*args, **kwargs):
    """诊断输出（默认静默；把 common.VERBOSE 设 True 或直接改这里的 _LOG 打开）。"""
    if _LOG:
        print(*args, **kwargs)


_LOG = False
_BUILD_HOOKS: list = []


def add_build_hook(fn) -> None:
    """登记一个「模型构建完成后」要执行的处理函数（同一个函数只登记一次）。"""
    if fn not in _BUILD_HOOKS:
        _BUILD_HOOKS.append(fn)


def clear_build_hooks() -> None:
    """清空登记表。train() 每个 run 开始前调一次，避免上一个 run 的钩子残留。"""
    _BUILD_HOOKS.clear()


def apply_build_hooks(model) -> None:
    for fn in _BUILD_HOOKS:
        fn(model)


def _patch_trainer_get_model() -> None:
    from ultralytics.engine.trainer import BaseTrainer

    original = BaseTrainer.__init__
    if getattr(original, "_custom_build_hook_patch", False):
        return

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if not _BUILD_HOOKS:
            return
        inner = self.get_model  # 绑定到具体子类的实现（detect / segment / ...）

        def get_model(cfg=None, weights=None, verbose=True):
            model = inner(cfg=cfg, weights=weights, verbose=verbose)
            before = sum(p.numel() for p in model.parameters())
            apply_build_hooks(model)
            after = sum(p.numel() for p in model.parameters())
            _log(f"[rebuild] ↑ 上面那张 parse_model 表是 trainer 按 yaml 重建出来的**中间态**"
                  f"（{before:,} 参数，legacy head 未修 / cv2/cv4 未深度化）；"
                  f"已重放 {len(_BUILD_HOOKS)} 个构建后处理 -> 真正训练的是 {after:,} 参数")
            return model

        self.get_model = get_model  # 实例属性优先于类方法

    __init__._custom_build_hook_patch = True
    BaseTrainer.__init__ = __init__


_patch_trainer_get_model()


# ======================================================================================
# 6. 导入时打印注册结果
# ======================================================================================
def _report():
    from ultralytics.utils import LOGGER

    # 注册清单不打了（训练日志里是噪音）。但 SKIPPED 一定要报 —— 那意味着某个论文模块
    # 没加载成功，YAML 里用到它就会 KeyError，属于必须立刻看见的问题。
    for name, why in SKIPPED.items():
        LOGGER.warning(f"custom_modules: SKIPPED {name} -> {why}")


_report()

__all__ = ("C3k2CBAM", "MobileNetV3Backbone", "REGISTERED", "SKIPPED", "register_base_module",
           "add_build_hook", "clear_build_hooks", "apply_build_hooks")
