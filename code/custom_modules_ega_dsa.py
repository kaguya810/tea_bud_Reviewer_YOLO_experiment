# -*- coding: utf-8 -*-
"""YOLOv11n + P2 检测分支 + EGA-Stem + DSA + LightSAFFM 的模块实现与 YAML 注册。

本文件服务于一次「结构 + 数值自检」实验，**不训练**。配套：
  modules YAML : cfg/yolo11n-tea-ega-dsa-saffm-p2.yaml
  自检脚本     : _ega_dsa_verify.py

与参考实现（C:\\WorkSpace-DL\\teabud\\modules\\code\\）的溯源关系
---------------------------------------------------------------------------
`52.0 (TCSVT 2025) EdgeGaussianAggregation.py`  -> EGA / EGAStem
    参考实现只有「固定高斯 + Conv_Extra + 残差」，且上游把 5x5 高斯核当 nn.Conv2d
    权重（requires_grad 仍为 True，冻结意图失效）。本文件：
      * 保留「冻结核必须写在 __init__ 内」这一约束（ultralytics 从 YAML/.pt 重建模型时
        会重新执行 __init__，写成外部后处理会在重载后静默解冻）；
      * **进一步改用 `register_buffer`**：实测发现 ultralytics 的
        `BaseTrainer._setup_train` 会把**所有** requires_grad=False 的浮点 Parameter
        强制改回 True（只有 `.dfl` 与 `freeze` 名单豁免），所以「冻结参数」写法在
        真实训练回路里必然失效。buffer 不在 `named_parameters()` 里，结构上无法被解冻。
        详见 `_fixed_buffer()` 的 docstring（含实测日志）。
      * 补上用户口径里的 **Scharr 边缘** 与 **ECA 通道注意力** 两个分支；
      * 去掉 mmcv 依赖（参考实现用 mmcv.cnn.build_norm_layer），改用内置 norm 工厂，
        语义等价（见 _build_norm 的说明）。

`3.0 (TNNLS 2025) Deformable Spatial Attention (DSA).py` -> DSA / DSCNPair / DSCN1D
    ⛔ 参考实现首行 `from ops_dscn.modules import DSCNX, DSCNY`。ops_dscn 是上游随仓库
    分发的自编译 C++/CUDA 扩展，PyPI 无 wheel、GitHub 在本机不可达、也没有现成的纯
    PyTorch 等价物。**本文件按 DSAN 预印本 Eqn.4a/4b 重新实现了一个纯 PyTorch 的
    DSCN**，语义对齐论文，但**不是**上游那个 CUDA 算子 —— 二者的数值不会逐位相同。
    见本文件末尾 `__main__` 里与「逐点 for 循环朴素实现」的对拍。
    另外参考实现里的 `permute(0,2,3,1)` 是上游 CUDA 算子要求 NHWC 的产物，
    纯 PyTorch 实现在 NCHW 上直接算，故本文件不作 permute（数值等价，少两次搬运）。

`88.0 (AAAI 2025) HFP.py` -> HFP / DctSpatialInteraction / DctChannelInteraction
    ⛔ 参考实现依赖 `torch_dct`（本机未装）。torch_dct 内部走 torch.fft，其 dct_2d 在
    norm='ortho' 下与「余弦变换矩阵 Dh @ x @ Dw.T」数值等价（float64 最大误差 ~3e-14）。
    本文件默认用余弦矩阵实现，避免新增依赖；装了 torch_dct 则自动切换（见 _dct_2d）。
    ⚠️ HFP 末尾是 3x3 卷积 + GroupNorm(32, C)，**要求 C % 32 == 0**。

`132.0 (2026 本项目自研) LightSAFFM.py` -> LightSAFFM / LightSAFFMFusion
    ✅ 纯 PyTorch，无外部依赖，本文件按原件移植（只改 docstring 与 import）。
    `LightSAFFMFusion` 是本文件新增的 **YAML 适配层**，原因见下。

YAML 集成的三个硬约束（读 ultralytics 8.4.142 的 parse_model 得到）
---------------------------------------------------------------------------
1. 模块名解析走 `globals()[m]`，所以 `setattr(ultralytics.nn.tasks, 名字, 类)` 即可让
   YAML 用上新名字。
2. **非白名单模块**（不在 parse_model 的 base_modules 里）走 `else: c2 = ch[f]`：
   * `f` 必须是 **int**（list 会让 `ch[f]` 直接 TypeError）；
   * 于是「输出通道 == ch[f]」——即模块必须**通道保持**，否则下游白名单模块拿到的
     c1 会与实际张量不符，运行期才炸。
3. 需要「通道变换 + width 缩放」的模块必须拿到 base_modules 语义。本文件用
   `_install_slot_patch()` 在 parse_model 调用期间把一个闲置白名单名字临时绑到
   EGAStem 上（与 custom_modules.py 的「借壳」机制同一手法，但用独立槽位，不冲突）。

由此得到本文件的设计取舍：
  * EGAStem      —— 借壳 "SPP" 槽，拿到 [c1, c2_scaled, *args] 语义（3 -> 16，stride 2）。
  * C3k2DSA      —— 同名替换 `tasks.C3k2`，完整复用 base_modules + repeat_modules 语义。
  * DetectP2Half —— 同名替换 `tasks.Detect`，复用 Detect 那一支的身份判断。
  * LightSAFFMFusion —— 通道保持（输出 = c_x + c_y = 前一层 Concat 的通道数），
                    因此不需要借壳。**这是本文件对「两输入融合」的折中**：
                    parse_model 无法把两个索引喂给一个非 Concat 模块，
                    所以写成 `Concat -> LightSAFFMFusion`（模块内部再 split 回两路）。
  * HFP          —— 通道保持，不需要借壳。
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C3k2, Detect, DFL, DWConv
import ultralytics.nn.tasks as tasks

__all__ = (
    "ECA",
    "EGAStem",
    "DSCN1D",
    "DSCNPair",
    "DSA",
    "LightSAFFM",
    "LightSAFFMFusion",
    "HFP",
    "DctSpatialInteraction",
    "DctChannelInteraction",
    "C3k2DSA",
    "DetectP2Half",
)


# ======================================================================================
# 0. 小工具
# ======================================================================================
def _build_norm(cfg, num_features: int) -> nn.Module:
    """`mmcv.cnn.build_norm_layer(cfg, num_features)[1]` 的最小等价实现。

    参考实现依赖 mmcv。本仓库的 .venv 里确实装了 mmcv 2.2.0（实测
    `build_norm_layer(dict(type='BN'), 8)` 可用），但 **C:\\WorkSpace-DL\\teabud 的
    .venv 没有**，而且完整版 mmcv 在 py3.12 + torch 2.14 下只能源码编译。
    这里只实现实际会用到的三种，避免给模块引入一个沉重的可选依赖。

    mmcv 语义对齐点：接受 dict；`requires_grad` 是弹出参数并作用于该层每个参数；
    `eps` 默认 1e-5；`GN` 需要 `num_groups` 且用 `num_channels=` 调用。
    """
    if isinstance(cfg, nn.Module):
        return cfg
    if not isinstance(cfg, dict):
        raise TypeError(f"norm_layer 必须是 dict（mmcv 语义），收到 {type(cfg).__name__}")
    cfg = dict(cfg)  # 不污染调用方的 dict
    kind = cfg.pop("type")
    requires_grad = cfg.pop("requires_grad", True)
    layer = {
        "BN": lambda: nn.BatchNorm2d(num_features, **cfg),
        "BN2d": lambda: nn.BatchNorm2d(num_features, **cfg),
        "LN": lambda: nn.LayerNorm(num_features, **cfg),
        "GN": lambda: nn.GroupNorm(cfg.pop("num_groups"), num_features, **cfg),
    }.get(kind)
    if layer is None:
        raise NotImplementedError(f"_build_norm 未实现 norm type={kind!r}（只支持 BN / LN / GN）")
    mod = layer()
    if not requires_grad:
        mod.requires_grad_(False)
    return mod


def _fixed_buffer(module: nn.Module, name: str, weight: torch.Tensor, note: str) -> None:
    """把解析核注册为 **buffer**（而不是 requires_grad=False 的 Parameter）。

    ⚠️ 这里踩过一个真实的坑，结论是「冻结参数」这套写法在 ultralytics 里**根本活不下来**：

    ultralytics 的 `BaseTrainer._setup_train`（engine/trainer.py）里有这么一段：

        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                v.requires_grad = False          # 只冻 freeze 名单里的（默认只有 .dfl）
            elif not v.requires_grad and v.dtype.is_floating_point:
                LOGGER.warning(f"setting 'requires_grad=True' for frozen layer '{k}'...")
                v.requires_grad = True           # ← 其余一切「冻结参数」都被强制解冻

    也就是说：**任何 requires_grad=False 的浮点 Parameter，只要名字不在 freeze 名单里，
    训练一开始就会被改回 True**。实测日志：
        WARNING setting 'requires_grad=True' for frozen layer 'model.0.scharr.weight'
        WARNING setting 'requires_grad=True' for frozen layer 'model.0.gauss.weight'

    之前 `_ega_dsa_verify.py` 的 F 段全绿，是因为它**只验证了建模态**，没有经过 trainer
    —— 这正说明了「离线自检通过 ≠ 训练回路正确」。

    几条出路里只有 buffer 是结构性安全的：
      * 用 `freeze=[0]` 按名字冻 —— 会把 EGAStem 里**可学习的** fuse/norm/eca/res 一起冻掉，❌
      * 挂 on_train_start 回调重新冻 —— ultralytics 升级或换入口就失效，且仍然刷警告，❌
      * `register_buffer` —— buffer 压根不在 `named_parameters()` 里，那段循环永远看不到它，
        也就没有任何东西能解冻它；同时它天然不进优化器、不被 weight_decay 碰、
        随 `.to()/.half()` 迁移、随 `state_dict` 存取（persistent=True）。✅

    代价：核从 `parameters()` 移到 `buffers()`，**参数量口径少 129**
    （Scharr 54 + Gaussian 75）—— 与上游 LEGNet 的参数量对比时要记住这一点。
    """
    module.register_buffer(name, weight.detach().clone(), persistent=True)
    module.__dict__.setdefault("_fixed_notes", {})[name] = note


def _fixed_conv2d(x: torch.Tensor, w: torch.Tensor, padding: int, groups: int) -> torch.Tensor:
    """用固定核做 depthwise 卷积。必要时把核转成 x 的 dtype（AMP 下 x 可能是 fp16）。"""
    if w.dtype != x.dtype:
        w = w.to(dtype=x.dtype)
    return F.conv2d(x, w, padding=padding, groups=groups)


def _to_2tuple(v):
    """标量（int/float）或长度 2 的序列 -> 二元组。"""
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


# ======================================================================================
# 1. EGA-Stem：Scharr 边缘 + 高斯抑噪（均为冻结先验）+ ECA + 残差
# ======================================================================================
def scharr_kernels(device=None, dtype=torch.float32) -> torch.Tensor:
    """返回 (2, 3, 3) 的 Scharr 核：`[gx, gy]`。

    Scharr 是 Sobel 的等角旋转改进版（8 邻域权重 3/10/3），对细边缘响应更各向同性，
    适合茶芽这种细长、低对比度的目标。
    归一化除以 16：sum(|gx|) = 3+10+3+3+10+3 = 32，除以 16 后为 2；
    这样边缘幅值的量纲与输入梯度同阶，避免把后续 BN 推到饱和。
    注意这里刻意不做 /2 之类的「半归一化」以外的调整，保持解析核可复现。
    """
    gx = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]], device=device, dtype=dtype)
    gy = torch.tensor([[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]], device=device, dtype=dtype)
    return torch.stack([gx, gy], 0) / 16.0


def scharr_depthwise_weight(c1: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """构造 depthwise Scharr 卷积的权重，形状 `(2*c1, 1, 3, 3)`。

    通道顺序刻意排成 **[gx_0..gx_{c1-1}, gy_0..gy_{c1-1}]**（而不是逐通道交错），
    这样 forward 里 `out.chunk(2, dim=1)` 拿到的前一半就正好是全部 gx、后一半是全部 gy。
    交错排布会让 chunk 拿到混合通道 —— 这是个很容易写错且不会报错的地方
    （形状全对、数值全错），所以单独抽成函数并在自检里逐通道核对。
    """
    k = scharr_kernels(device=device, dtype=dtype)  # (2, 3, 3)
    w = torch.cat([k[0:1].repeat(c1, 1, 1), k[1:2].repeat(c1, 1, 1)], dim=0)  # (2*c1, 3, 3)
    return w.unsqueeze(1)  # (2*c1, 1, 3, 3)


def gaussian_kernel(size: int, sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """(1, 1, size, size) 的归一化高斯核。坐标口径与参考实现一致（-size//2+1 ..= size//2）。"""
    if size % 2 == 0:
        raise ValueError(f"高斯核边长必须为奇数（偶数核关于中心不对称），收到 size={size}")
    if not (sigma > 0):
        # sigma=0 时 exp(-r²/(2σ²)) = exp(-inf·0) = 0/0 -> 全 NaN 核；
        # 它会被注册成 buffer，之后每次前向、每次 loss 都是 NaN，却**不抛任何异常** ——
        # 一个 YAML 手误就能让整次训练白跑。这里直接拦掉。
        raise ValueError(
            f"sigma 必须 > 0（sigma=0 会产生全 NaN 的高斯核且全程不报错），收到 sigma={sigma}"
        )
    ax = torch.arange(-size // 2 + 1, size // 2 + 1, device=device, dtype=torch.float64)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    k = k / k.sum()
    return k.to(dtype).unsqueeze(0).unsqueeze(0)


class ECA(nn.Module):
    """Efficient Channel Attention（Wang et al., CVPR 2020）的条带实现。

    与 SENet 的区别：不做降维，直接用一个长度为 k 的 1-D 卷积在 GAP 后的通道序列上
    做跨通道交互。k 由通道数自适应：k = |log2(C) + 1| / 2，再向上取到奇数。
    """

    def __init__(self, channels: int, k: int | None = None):
        super().__init__()
        if k is None:
            t = int(abs((math.log2(channels) + 1) / 2))
            k = t if t % 2 else t + 1
        k = max(3, k | 1)  # 至少 3 且强制奇数
        self.k = k
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.adaptive_avg_pool2d(x, 1)                       # B,C,1,1
        y = self.conv(y.squeeze(-1).transpose(-1, -2))        # B,1,C
        y = torch.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # B,C,1,1
        return x * y


class EGAStem(nn.Module):
    """EGA-Stem：浅层边缘提取 + 抑噪 + 通道注意力 + 残差，输出 stride=2。

    数据流（c1=3 -> c2=16 @ n 档）：

        x ──┬─ Scharr(BUFFER, 固定) ─ |·| ───┐
            ├─ Gaussian(BUFFER, 固定) ───────┤ cat -> Conv(3*c1 -> c2, k=3, s=stride) -> ECA ─┐
            └────────────────────────────────┘                                                ├─ + -> out
            └─ Conv(c1 -> c2, k=3, s=stride)  ─────────────────────────── 残差（可学习）───────┘

    设计说明（相对参考实现的偏离，均为**有意**且可核对）：
      * 参考实现 `EdgeGaussianAggregation` 只有高斯一路；上游 LEGNet 的 `legacy.py` 里
        另有 `Scharr` 类，搬运时被重新打包丢掉了。本文件按用户口径把 Scharr 补回来。
      * 参考实现用 `x + gaussian` 再进 `Conv_Extra`；本文件把三条路 concat 后一次融合
        （Scharr 幅值、高斯抑噪、原始输入），因为 P2 小目标既需要边缘也需要不过度平滑的
        原始细节。
      * **固定核用 buffer，不用冻结 Parameter** —— 见 `_fixed_buffer()`：
        ultralytics 训练器会强制解冻一切 requires_grad=False 的浮点参数。
      * 核形状取自论文（LEGNet §3.3.2 用 5x5 / sigma=1.0 高斯），Scharr 用标准 3x3。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        stride: int = 2,
        gauss_k: int = 5,
        sigma: float = 1.0,
        eca_k: int | None = None,
        norm_layer: dict | None = None,
        act_layer=nn.SiLU,
    ):
        super().__init__()
        self.c1, self.c2, self.stride = c1, c2, stride
        self.gauss_k, self.sigma = gauss_k, sigma

        # ---------------------------------------------------------------------------
        # ⚠️ 子模块的**创建顺序是有承载作用的，不要重排**：
        #   ultralytics 的 get_flops() 用 `p = next(model.parameters())` 推输入通道数，
        #   再 `torch.empty((1, p.shape[1], H, W))` 造探针输入。
        #   本模块的固定核是 buffer（不在 parameters() 里），所以第一个参数是下面这个
        #   残差 3x3 卷积，形状 (c2, c1, 3, 3)，shape[1] == c1 == 3 ✔。
        #   自检脚本的 S9d/S4 就是钉这个不变量的。
        # ---------------------------------------------------------------------------
        self.res = Conv(c1, c2, 3, stride, 1)  # 必须最先创建，见上

        # ---------------- 固定先验（buffer，非参数；原因见 _fixed_buffer 的注释） ----------------
        _fixed_buffer(self, "scharr_w", scharr_depthwise_weight(c1), note="scharr gx/gy")
        _fixed_buffer(self, "gauss_w", gaussian_kernel(gauss_k, sigma).repeat(c1, 1, 1, 1),
                      note=f"gaussian {gauss_k}x{gauss_k} sigma={sigma}")

        # ---------------- 可学习分支 ----------------
        # 注意 Conv 的 stride 参数：(c1, c2, k, s, p)
        self.fuse = Conv(3 * c1, c2, 3, stride, 1)  # cat([x, |Scharr|, Gaussian]) -> c2
        self.norm = _build_norm(norm_layer or dict(type="BN"), c2)
        self.act = act_layer()
        self.eca = ECA(c2, eca_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 逐通道 depthwise 用固定核；scharr_w 排布是 [gx*c1, gy*c1]，故 chunk(2,1) 前后两半各是一整组
        gx, gy = _fixed_conv2d(x, self.scharr_w, 1, self.c1).chunk(2, dim=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)  # 梯度幅值；eps 防 sqrt 在 0 处 NaN 梯度
        denoised = _fixed_conv2d(x, self.gauss_w, self.gauss_k // 2, self.c1)
        z = self.fuse(torch.cat([x, edge, denoised], dim=1))
        z = self.act(self.norm(z))
        z = self.eca(z)
        return z + self.res(x)

    def extra_repr(self) -> str:
        return f"c1={self.c1}, c2={self.c2}, stride={self.stride}, gauss={self.gauss_k}/sigma={self.sigma}"


# ======================================================================================
# 2. DSA：Deformable Strip Convolution（纯 PyTorch 等价实现）+ Deformable Spatial Attention
# ======================================================================================
class DSCN1D(nn.Module):
    """沿单一轴的 1-D 可变形条带卷积（DSCN）。

    对应 DSAN 预印本 Eqn. 4a（x 轴）/ Eqn. 4b（y 轴）：

        y(i0, j0) = Σ_g Σ_k  w[g, k] · x( i0 + δ_y ,  j0 + k·d + Δp[g,k] )      (x 轴)
                                  ↑ 只有轴向坐标带偏移，另一轴保持 i0

    与 `torchvision.ops.deform_conv2d` 的区别（重要，不能用后者冒充）：
      * deform_conv2d 做的是 **双线性** 插值，两个轴同时偏移；
      * DSCN 只在一个轴上偏移，因此退化成 **1-D 线性插值**（论文 Fig.3 / Eqn.5-6 的
        核心论点：同样的偏移采样，两次 1-D 线性插值只需双线性 63.2% 的浮点运算）。

    实现要点：
      * 偏移量由一个极小的卷积分支预测（dw 5x5 -> GELU -> 1x1 -> G*K），
        **末层零初始化** —— 于是训练起点严格等于一个规则的 K×1 条带卷积，
        这对可训练性和自检都很关键（见 _ega_dsa_verify.py 的 **N2** 与 **N2c**）。
      * **两种采样实现，数学等价，性能差很多**（`impl=` 可选）：
          - `"gather"`：`x.gather(dim, idx)` + `lerp`。**实测慢得多**，但原因要说清：
            `idx.unsqueeze(1).expand(-1, C, -1, -1)` 得到的是 stride (0,0,0,1) 的**视图**；
            实测 `gather` 峰值显存 27.3 MB，与输出 26.2 MB 相当 ——
            **并没有物化 52 MB 的 int64 索引**（显式 `.contiguous()` 才会，那一步是 52.4 MB）。
            真实训练回路的实测差异是 113.0 -> 60.8 s/epoch（1.86×，见报告 §9.8.2）。
            机制尚未定位；「stride-0 索引会走通用 gather kernel」只是猜测，
            **不要当结论引用**。
          - `"grid"`（默认）：`F.grid_sample`，一次 kernel 完成整轴插值，无 int64 索引；
            `align_corners=True` + `padding_mode='border'`，与 gather 版的
            `clamp` 边界行为一致（见 _ega_dsa_verify.py 的 **N1c**，含 fp16 用例）。
            但两者都不与 fp64 参照逐位相等：fp32 下 grid 差 ~6e-5，所以 N1c 的容差
            按精度分档（fp64 1e-10 / fp32 1e-4 / fp16 1e-3），不是 0。
       实测（真实训练回路，batch 16 / 640，channels_last 关闭）：
        DSA 占单 epoch 的 56%（119.4 s vs 关闭后 52.1 s），换 grid 实现后显著下降。
    """

    def __init__(self, dim: int, kernel_size: int = 11, axis: str = "x", group: int = 1,
                 dw_kernel_size: int = 5, dilation: int = 1, act=nn.GELU, impl: str | None = None):
        super().__init__()
        if axis not in ("x", "y"):
            raise ValueError(f"axis 只能是 'x' 或 'y'，收到 {axis!r}")
        if kernel_size < 1:
            raise ValueError("kernel_size 必须 >= 1")
        # impl=None -> 取模块级默认。
    # 默认是 "gather"（确定性），不是 "grid" —— 理由见本文件 DSCN1D 的 _grid_sample 注释与
    # 报告 §13.7：grid 的 CUDA 反向没有确定性实现，会让 deterministic=True 名不副实。
    # DSCN_IMPL=grid 可切回快 2.5× 但非确定的那条路（A/B 基准用）。
        impl = impl or os.environ.get("DSCN_IMPL", "gather")
        if impl not in ("grid", "gather"):
            raise ValueError(f"impl 只能是 'grid' 或 'gather'，收到 {impl!r}")
        self.dim, self.K, self.axis, self.group, self.dilation = dim, kernel_size, axis, group, dilation
        self.impl = impl

        self.dw = nn.Conv2d(dim, dim, dw_kernel_size, padding=dw_kernel_size // 2, groups=dim)
        self.act = act()
        self.offset = nn.Conv2d(dim, group * kernel_size, 1)
        # 零初始化 -> 起点是规则条带卷积。这一步不能用默认初始化。
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

        # 逐通道的抽头权重（DCNv3 的 depthwise 权重在 DSCN 里的对应物）。
        # 若不加权重，K 个采样点直接相加会把激活放大 ~sqrt(K) 倍，训练初期不稳定。
        self.tap = nn.Parameter(torch.full((dim, group, kernel_size), 1.0 / kernel_size))
        # 规则网格的基准位置，居中：K=11 -> [-5..5]，与 pad=K//2 的 "same" 口径一致
        self.register_buffer("base", (torch.arange(kernel_size) - (kernel_size - 1) // 2).float() * dilation,
                             persistent=False)

    @staticmethod
    def _acc_dtype(x: torch.Tensor) -> torch.dtype:
        """半精度一律升到 fp32 做坐标与采样运算；其它精度原样。

        ⚠️ 这不是「保险起见」，而是在修一个实测 ~4% 量级的**静默**数值偏差：
            AMP 下 x 是 fp16，坐标 `pos + base + off` 也在 fp16 里算。
            归一化坐标 n = 2p/(size-1) - 1 在 ±1 附近的 fp16 ulp/2 ≈ 4.9e-4，
            被 (size-1)/2 放大回像素空间（P2 的 160 -> 79.5）≈ 0.06 px，
            双线性插值于是混入邻行/邻列约 6%。
            实测（B=4, C=16, 160x160, K=11, 偏移 ±2.5px, 与 fp64 参照比）：
              fp32: grid 6.4e-5 / gather 1.3e-5          -> 1e-5 量级，合格
              fp16: grid 1.2e-1 / gather 9.3e-2，grid-gather = 1.1e-1（相对 4.07%）
            升 fp32 后两者都回到 ~1e-3 量级。
            注意 fp64 输入必须**保持 fp64** —— 自检的 N1/N1b/N1c 都在 fp64 下比对。
        """
        return torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype

    @staticmethod
    def _linear_sample(x: torch.Tensor, coord: torch.Tensor, axis: str) -> torch.Tensor:
        """在 `axis` 轴上按浮点坐标做 1-D 线性插值（边界 clamp）。

        `frac` 必须转回 `x.dtype`：`torch.lerp` 要求 weight 与输入同 dtype。
        第 3 批把坐标升到 fp32 之后，若不转就会抛
        `RuntimeError: expected dtype struct c10::Half for weight but got dtype float`
        —— 这正是「坐标只能是 fp16」这个隐式依赖的暴露点。
        """
        size = x.shape[2] if axis == "y" else x.shape[3]
        coord = coord.clamp(0.0, size - 1.0)
        lo = coord.floor()
        frac = (coord - lo).unsqueeze(1).to(x.dtype)   # B,1,H,W  <= lerp 的 dtype 约束
        i0 = lo.long().clamp(0, size - 1)
        i1 = (i0 + 1).clamp(0, size - 1)
        dim = 2 if axis == "y" else 3
        idx0 = i0.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        idx1 = i1.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        return torch.lerp(x.gather(dim, idx0), x.gather(dim, idx1), frac)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        off = self.offset(self.act(self.dw(x))).view(b, self.group, self.K, h, w)
        return self.sample_sum(x, off)

    def sample_sum(self, x: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
        """给定偏移量张量 `off (B, G, K, H, W)`，做条带采样加权求和。

        从 forward 里抽出来是为了让自检脚本能**显式喂入偏移量**，
        与「逐点 for 循环的朴素实现」逐元素对拍（见 _ega_dsa_verify.py 的 **N1**）。

        ⚠️ 坐标口径（这里踩过一次坑，务必看清）：
            Eqn.4a 的采样位置是 `j0 + j + Δp_gj` —— **绝对**坐标，
            即「输出位置 j0」+「抽头基准」+「学习到的偏移」三者之和。
            本实现用居中的 base（= j - (K-1)//2，配合 pad=(K-1)//2 得到 "same" 卷积，
            与参考实现 DSCNPair 传 pad=5/kernel=11 的口径一致）。
            如果漏掉 `pos` 这一项，coord 就变成相对量，所有输出位置会采样同一个点，
            算子退化成「与位置无关的逐通道缩放 + 常数」——
            **形状全对、能训、loss 也会降，但形变能力完全丧失**。
            `_ega_dsa_verify.py` 的 N2 就是钉这个不变量的（零偏移时必须等于规则条带卷积）。
        """
        b, c, h, w = x.shape
        # 坐标运算统一在 acc_dtype（fp16 -> fp32）。见 _acc_dtype：
        # fp16 下坐标本身会被量化（160 附近 fp16 ulp = 0.125 -> 1/8 px），
        # 前向变阶梯、梯度仍平滑 -> 更新方向有偏、可学到的形变分辨率受限。
        adt = self._acc_dtype(x)
        # 输出位置本身：沿 axis 的整数坐标
        if self.axis == "y":
            pos = torch.arange(h, device=x.device, dtype=adt).view(1, h, 1)
        else:
            pos = torch.arange(w, device=x.device, dtype=adt).view(1, 1, w)
        base = self.base.to(adt)
        # 累加也在 acc_dtype（fp32 累加更准，tap 本来也是 fp32）；最后转回 x.dtype，
        # 维持「输出 dtype == 输入 dtype」这个干净契约。
        out = torch.zeros(b, c, h, w, device=x.device, dtype=adt)
        for g in range(self.group):
            for k in range(self.K):
                coord = pos + base[k] + off[:, g, k].to(adt)  # B,H,W 绝对像素坐标
                if self.impl == "grid":
                    sampled = self._grid_sample(x, coord, self.axis)
                else:
                    sampled = self._linear_sample(x, coord, self.axis)
                out = out + sampled.to(adt) * self.tap[:, g, k].to(adt).view(1, c, 1, 1)
        return out.to(x.dtype)

    @staticmethod
    def _grid_sample(x: torch.Tensor, coord: torch.Tensor, axis: str) -> torch.Tensor:
        """用 `F.grid_sample` 做同一条 1-D 插值 —— 免掉 int64 索引。

        与 `_linear_sample` 的等价性：
          * `align_corners=True` 时归一化坐标 `-1 / +1` 正好对应像素 `0 / size-1`，
            所以 `norm = 2*px/(size-1) - 1`；
          * `padding_mode='border'` 把越界坐标夹到边缘，正对应 gather 版的
            `coord.clamp(0, size-1)`；
          * 不偏移的那一轴给的是**整数像素坐标**，双线性插值在该轴上权重恰好落在
            单个像素上，于是整体退化成纯 1-D 线性插值。
        """
        b, c, h, w = x.shape
        # 采样与归一化都在 _acc_dtype（fp16 -> fp32）里做；grid 与 input 的 dtype 必须一致，
        # 所以 x 也要跟着升（只转 grid 会抛 "expected scalar type Half but found Float"）。
        dt = DSCN1D._acc_dtype(x)
        x_s = x.to(dt)
        coord = coord.to(dt)
        ar = lambda n: torch.arange(n, device=x.device, dtype=dt)
        if axis == "y":
            n_axis = 2.0 * coord.clamp(0.0, h - 1.0) / max(h - 1, 1) - 1.0          # 高度（被采样轴）
            n_other = (2.0 * ar(w).view(1, 1, w) / max(w - 1, 1)) - 1.0
            grid = torch.stack([n_other.expand(b, h, w), n_axis.expand(b, h, w)], dim=-1)
        else:
            n_axis = 2.0 * coord.clamp(0.0, w - 1.0) / max(w - 1, 1) - 1.0          # 宽度（被采样轴）
            n_other = (2.0 * ar(h).view(1, h, 1) / max(h - 1, 1)) - 1.0
            grid = torch.stack([n_axis.expand(b, h, w), n_other.expand(b, h, w)], dim=-1)
        return F.grid_sample(x_s, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, K={self.K}, axis={self.axis}, group={self.group}, dilation={self.dilation}"


class DSCNPair(nn.Module):
    """先沿 x 再沿 y 做两次条带采样，最后与输入逐元素相乘（门控）。

    结构完全对齐参考实现的 `DSCNPair.forward`：
        u = x; h = conv0(x); a = dscn_y(dscn_x(h)); return u * conv1x1(a)
    差别只有一点：参考实现在两次 DSCN 之间做了 `permute(0,2,3,1)`（NHWC），
    那是上游 CUDA 算子的布局要求；纯 PyTorch 版在 NCHW 上直接算，数值等价。
    """

    def __init__(self, dim: int, kernel_size: int = 11, dw_kernel_size: int = 5,
                 group: int = 1, dilation: int = 1):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, dw_kernel_size, padding=dw_kernel_size // 2, groups=dim)
        self.dscn_x = DSCN1D(dim, kernel_size, axis="x", group=group, dw_kernel_size=dw_kernel_size, dilation=dilation)
        self.dscn_y = DSCN1D(dim, kernel_size, axis="y", group=group, dw_kernel_size=dw_kernel_size, dilation=dilation)
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x
        a = self.dscn_y(self.dscn_x(self.conv0(x)))
        return u * self.conv(a)


class DSA(nn.Module):
    """Deformable Spatial Attention：`x + proj_2(SGU(GELU(proj_1(x))))`。

    论文的核心观察是 DCNv3 里的 modulation mask 分支与空间注意力同构，
    于是用「一对正交的 DSCN + 门控」替代 DCNv3。**通道保持**，可原地替换 C3k2 的
    Bottleneck。
    """

    def __init__(self, d_model: int, kernel_size: int = 11, dw_kernel_size: int = 5,
                 group: int = 1, dilation: int = 1):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = DSCNPair(d_model, kernel_size, dw_kernel_size, group, dilation)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj_2(self.spatial_gating_unit(self.activation(self.proj_1(x))))


# ======================================================================================
# 3. LightSAFFM（本项目自研，按原件移植）+ YAML 适配层
# ======================================================================================
class LightSAFFM(nn.Module):
    """轻量双路特征融合：通道注意力（来自 x）+ 空间注意力（来自 y）联合门控。

    Args:
        in_ch_x: 第一路输入通道数，同时决定**输出**通道数。
        in_ch_y: 第二路输入通道数，默认与 `in_ch_x` 相同。
        mid_ch:  融合瓶颈通道数。默认 `max(32, min(in_ch_x, in_ch_y) // 4)`。
        spatial_k: 空间注意力卷积核，**必须为奇数**（默认 7）。
        use_dw:  是否在空间注意力前插入两层深度可分离卷积。

    形状:
        x: (B, in_ch_x, H, W)   y: (B, in_ch_y, H, W)   两路 H/W 必须一致
        返回: (B, in_ch_x, H, W)
    """

    def __init__(self, in_ch_x, in_ch_y=None, mid_ch=None, spatial_k=7, use_dw=False):
        super().__init__()
        if in_ch_y is None:
            in_ch_y = in_ch_x
        if mid_ch is None:
            mid_ch = max(32, min(in_ch_x, in_ch_y) // 4)
        if spatial_k % 2 == 0:
            raise ValueError(
                f"spatial_k 必须为奇数（偶数会使空间注意力图变为 H+1 并导致广播失败），收到 spatial_k={spatial_k}"
            )

        self.use_dw = use_dw
        self.proj_x = nn.Conv2d(in_ch_x, mid_ch, 1)
        self.proj_y = nn.Conv2d(in_ch_y, mid_ch, 1)

        if use_dw:
            self.dw = nn.Sequential(
                nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(),
                nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(),
            )
        else:
            self.dw = nn.Identity()

        self.spa = nn.Conv2d(2, 1, spatial_k, padding=spatial_k // 2, bias=False)
        # 一维通道卷积核长：按通道数取「floor(log2(C)) 向上取到奇数」。
        # ⚠️ 这与标准 ECA 的 k=int((log2(C)+1)/2) 不同，约宽 2 倍，是有意放宽感受野的选择。
        # mid_ch >= 1 时 bit_length()-1 >= 0，abs() 冗余（上游原文如此，保留以便对拍）
        eca_k = max(3, (mid_ch.bit_length() - 1) // 2 * 2 + 1)
        self.eca = nn.Conv1d(1, 1, kernel_size=eca_k, padding=eca_k // 2, bias=False)
        self.fuse = nn.Conv2d(mid_ch * 2, in_ch_x, 1)
        self.bn = nn.BatchNorm2d(in_ch_x)

    def forward(self, x, y):
        x = self.proj_x(x)
        y = self.proj_y(y)

        gap = x.mean((2, 3)).unsqueeze(1)                                       # B,1,C
        att_c = torch.sigmoid(self.eca(gap)).transpose(-1, -2).unsqueeze(-1)    # B,C,1,1

        f = self.dw(y)
        avg = f.mean(1, keepdim=True)
        maxv, _ = f.max(1, keepdim=True)
        att_s = torch.sigmoid(self.spa(torch.cat([avg, maxv], 1)))              # B,1,H,W

        # 加法而非乘法：两个 sigmoid 相乘会把特征系统性压到 ~1/4（原件注释里的实测）
        att = 0.5 * (att_c + att_s)

        y_feat = f if self.use_dw else y
        z = torch.cat([x * att, y_feat * att], 1)
        return F.relu(self.bn(self.fuse(z)))


class LightSAFFMFusion(nn.Module):
    """LightSAFFM 的 **YAML 适配层**：把「两输入融合」塞进 ultralytics 的单输入模块约束。

    为什么需要它（读 parse_model 源码得到，非猜测）：
        非白名单模块走 `else: c2 = ch[f]`，`f` 必须是 int；把两个索引写成 list
        会让 `ch[f]` 抛 TypeError。想支持 list `f` 只有 Concat 那一支，而 Concat 分支
        硬编码 `c2 = sum(ch[x])`。因此「LightSAFFM 直接替换 Concat」在这套 YAML DSL 里
        做不到 —— 除非 monkey-patch parse_model 或按输出通道=求和来对齐账本。

    本类的做法：保持**输出通道 = c_x + c_y**（与它前面的 Concat 行完全一致），
    于是 `c2 = ch[f]` 天然正确，下游 C3k2 拿到的 c1 也是对的，无需任何 patch。

        z(Concat) -> split -> LightSAFFM(x, y) -> cat([gated, y]) -> out   # (c_x + c_y)

    语义上是「用 LightSAFFM 的门控结果替换掉 x 那一路，再与 y 拼接」——
    对标准 PAN 节点（Concat -> C3k2）而言是严格的 drop-in 替换。
    """

    def __init__(self, c_x: int, c_y: int, mid_ch: int | None = None, spatial_k: int = 7, use_dw: bool = False):
        super().__init__()
        self.c_x, self.c_y = c_x, c_y
        self.inner = LightSAFFM(c_x, c_y, mid_ch, spatial_k, use_dw)

    @property
    def out_channels(self) -> int:
        return self.c_x + self.c_y

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x, y = z.split([self.c_x, self.c_y], dim=1)
        return torch.cat([self.inner(x, y), y], dim=1)

    def extra_repr(self) -> str:
        return f"c_x={self.c_x}, c_y={self.c_y}, out={self.out_channels}"


# ======================================================================================
# 4. HFP：High Frequency Perception（AAAI 2025 / HS-FPN）—— 可选消融项
# ======================================================================================
_TORCH_DCT = None
try:  # pragma: no cover - 取决于环境
    import torch_dct as _TORCH_DCT  # type: ignore
except Exception:
    _TORCH_DCT = None


def _dct_matrix(n: int, device, dtype) -> torch.Tensor:
    """正交归一 DCT-II 矩阵 M，满足 `dct(x) = M @ x @ M.T`。"""
    k = torch.arange(n, device=device, dtype=torch.float64).view(-1, 1)
    i = torch.arange(n, device=device, dtype=torch.float64).view(1, -1)
    m = torch.cos(math.pi * (2.0 * i + 1.0) * k / (2.0 * n))
    m[0] *= math.sqrt(1.0 / n)
    m[1:] *= math.sqrt(2.0 / n)
    return m.to(dtype)


def _dct_2d(x: torch.Tensor) -> torch.Tensor:
    """2-D DCT-II（norm='ortho'）。装了 torch_dct 就用它，否则走余弦矩阵。"""
    if _TORCH_DCT is not None:
        return _TORCH_DCT.dct_2d(x, norm="ortho")
    mh = _dct_matrix(x.shape[-2], x.device, x.dtype)
    mw = _dct_matrix(x.shape[-1], x.device, x.dtype)
    return mh @ x @ mw.transpose(-1, -2)


def _idct_2d(x: torch.Tensor) -> torch.Tensor:
    """2-D 逆 DCT（DCT-III，norm='ortho'），即 `_dct_2d` 的转置。"""
    if _TORCH_DCT is not None:
        return _TORCH_DCT.idct_2d(x, norm="ortho")
    mh = _dct_matrix(x.shape[-2], x.device, x.dtype)
    mw = _dct_matrix(x.shape[-1], x.device, x.dtype)
    return mh.transpose(-1, -2) @ x @ mw


class DctSpatialInteraction(nn.Module):
    """空间交互：DCT 后把左上角低频块置零（保留高频），再 IDCT 得到空间掩码。"""

    def __init__(self, in_channels: int, ratio, isdct: bool = True):
        super().__init__()
        self.ratio = _to_2tuple(ratio) if not isinstance(ratio, (tuple, list)) else tuple(ratio)
        self.isdct = isdct
        if not isdct:
            self.spatial1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        h, w = x.shape[-2:]
        dct = _dct_2d(x)
        weight = self._compute_weight(h, w, self.ratio).to(dct.device, dct.dtype)
        weight = weight.view(1, h, w).expand_as(dct)
        return x * _idct_2d(dct * weight)

    def _compute_weight(self, h: int, w: int, ratio) -> torch.Tensor:
        h0, w0 = int(h * ratio[0]), int(w * ratio[1])
        weight = torch.ones((h, w))
        weight[:h0, :w0] = 0
        return weight


class DctChannelInteraction(nn.Module):
    """通道交互：DCT 高频分量 -> 池化 -> 两条 1x1 分组卷积 -> 通道门控。"""

    def __init__(self, in_channels: int, patch, ratio, isdct: bool = True, groups: int = 32):
        super().__init__()
        if in_channels % groups:
            raise ValueError(f"DctChannelInteraction 要求 in_channels({in_channels}) 能被 groups({groups}) 整除")
        self.in_channels = in_channels
        self.h, self.w = _to_2tuple(patch) if not isinstance(patch, (tuple, list)) else tuple(patch)
        self.ratio = _to_2tuple(ratio) if not isinstance(ratio, (tuple, list)) else tuple(ratio)
        self.isdct = isdct
        self.channel1x1 = nn.Conv2d(in_channels, in_channels, 1, groups=groups, bias=False)
        self.channel2x1 = nn.Conv2d(in_channels, in_channels, 1, groups=groups, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        if not self.isdct:
            amaxp = F.adaptive_max_pool2d(x, 1)
            aavgp = F.adaptive_avg_pool2d(x, 1)
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))
            return x * torch.sigmoid(self.channel2x1(channel))

        dct = _dct_2d(x)
        weight = self._compute_weight(h, w, self.ratio).to(dct.device, dct.dtype)
        dct = dct * weight.view(1, h, w).expand_as(dct)
        dct_ = _idct_2d(dct)

        amaxp = F.adaptive_max_pool2d(dct_, (self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, (self.h, self.w))
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)
        # 加法而非 concat：两条通路的量纲不同（原件 2025-03-15 的注释也指出了这点）
        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(channel))

    def _compute_weight(self, h, w, ratio) -> torch.Tensor:
        h0, w0 = int(h * ratio[0]), int(w * ratio[1])
        weight = torch.ones((h, w))
        weight[:h0, :w0] = 0
        return weight


class HFP(nn.Module):
    """High Frequency Perception：DCT 空域 + 通道双路交互，**通道保持**。

    ⚠️ 约束（原件实测）：
      * `ratio` 必填，无默认值；
      * 末尾 `GroupNorm(32, C)` 与 `groups=32` 的分组卷积都要求 **C % 32 == 0**；
      * FLOPs 读数会**低估**它的真实开销 —— thop 不计 torch.fft / matmul，
        而 HFP 的核心正是 DCT。评估成本要看延迟，不能只看 GFLOPs。
    """

    def __init__(self, in_channels: int, ratio, patch=(8, 8), is_dct: bool = True, groups: int = 32):
        super().__init__()
        if in_channels % groups:
            raise ValueError(f"HFP 要求 in_channels({in_channels}) 能被 {groups} 整除（GroupNorm/分组卷积约束）")
        # ratio 允许写成标量（0.25）或二元组（(0.25, 0.125)）；在这里统一成二元组，
        # 避免下传到 DctChannelInteraction 时 tuple(0.25) 抛 TypeError。
        ratio = _to_2tuple(ratio) if not isinstance(ratio, (tuple, list)) else tuple(ratio)
        if len(ratio) != 2:
            raise ValueError(f"ratio 必须是标量或长度 2 的序列，收到 {ratio!r}")
        self.ratio = ratio
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=is_dct)
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=is_dct, groups=groups)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.spatial(x) + self.channel(x))


# ======================================================================================
# 5. C3k2DSA：把 C3k2 内部的 Bottleneck 按比例换成 DSA
# ======================================================================================
class C3k2DSA(C3k2):
    """C3k2 的变体：把内部 `self.m` 里的前 `dsa_n` 个单元换成 DSA（通道保持）。

    参数（与 ultralytics 8.4.142 的 C3k2 签名保持一致，只在**末尾**追加新参数，
    这样同名替换后既有的 YAML 写法全部继续有效）：

        C3k2(c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True,
             dsa_n=0, dsa_ratio=0.0, dsa_kernel=11, dsa_group=1)

    * `dsa_n` 优先；为 0 时用 `dsa_ratio` 按 `ceil(n * ratio)` 推。
    * 两者都为 0 -> 行为与官方 C3k2 **完全相同**（本类的 no-op 路径，自检里有对拍）。
    * ⚠️ `dsa_ratio` 用 `ceil` 且至少 1：因为 n 档 depth=0.5 时 n 常被压到 1，
      而 Python 的 `round(0.5) == 0`（银行家舍入），直接用 round 会让 50% 变成 0 个 DSA。
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True,
                 dsa_n=0, dsa_ratio=0.0, dsa_kernel=11, dsa_group=1):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.dsa_n = self._resolve_n(n, dsa_n, dsa_ratio)
        self.dsa_ratio = dsa_ratio
        if self.dsa_n <= 0:
            return  # no-op：与官方 C3k2 逐位一致

        c = self.c
        blocks = list(self.m)
        for i in range(min(self.dsa_n, len(blocks))):
            # DSA 的 d_model 必须等于 C3k2 内部通道 self.c（C2f 的 hidden = int(c2 * e)）
            blocks[i] = DSA(c, kernel_size=dsa_kernel, group=dsa_group)
        self.m = nn.ModuleList(blocks)

    @staticmethod
    def _resolve_n(n: int, dsa_n: int, dsa_ratio: float) -> int:
        if dsa_n and dsa_n > 0:
            return min(int(dsa_n), int(n))
        if dsa_ratio and dsa_ratio > 0:
            return min(int(n), max(1, math.ceil(n * float(dsa_ratio) - 1e-9)))
        return 0

    def extra_repr(self) -> str:
        return f"n={len(self.m)}, dsa_n={self.dsa_n}, ratio={self.dsa_ratio}"


# ======================================================================================
# 6. DetectP2Half：4 尺度 Detect，P2 那一支的隐藏通道减半
# ======================================================================================
class DetectP2Half(Detect):
    """4 尺度 anchor-free + DFL 检测头，P2 分支的隐藏通道按 `p2_scale` 减半。

    官方 `Detect.__init__` 里隐藏通道是**全局共享**的：
        c2 = max(16, ch[0] // 4, reg_max * 4);  c3 = max(ch[0], min(nc, 100))
    两个都从 ch[0]（第一个尺度）推出。四尺度时 ch[0] 是 P2，通道数最小（n 档 64），
    于是 P2 与 P5 共用同一套隐藏通道 —— 对 160x160 的 P2 特征图来说这笔开销最贵。
    本类把每个尺度的隐藏通道拆开，P2 那一支乘 `p2_scale`。

    注意：`legacy` 由 parse_model 在**构造之后**赋给模块，所以这里必须在 `__init__`
    里就复刻 Detect 的分支逻辑（用实例属性 self.legacy，默认类属性 False = v11 口径）。
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=(), p2_scale=0.5, p2_index=0, min_ch=8):
        if end2end:
            raise NotImplementedError("DetectP2Half 暂不支持 end2end 双头（one2one_* 未做同样的通道拆分）")
        super().__init__(nc, reg_max, end2end, ch)
        if not 0.0 < p2_scale <= 1.0:
            raise ValueError(f"p2_scale 必须在 (0, 1] 内，收到 {p2_scale}")
        self.p2_scale, self.p2_index = p2_scale, p2_index

        ch = list(ch)
        c2, c3 = max(16, ch[0] // 4, reg_max * 4), max(ch[0], min(self.nc, 100))
        hid2, hid3 = [], []
        for i in range(self.nl):
            s = p2_scale if i == p2_index else 1.0
            # s == 1.0 必须**逐位**等于官方 Detect（自检 N7 断言的就是这条）。
            # 第一版对所有层级都做 round-to-8 + min_ch 下限，于是 hidden 宽度不是 8 的倍数时
            # 就不再等价：实测 nc=91/ch0=264 时 cv2 hidden 66 -> 64、nc=4/ch0=100 时 100 -> 96。
            # 本仓库的配置（nc=4, ch=[32,64,128,256]）恰好都是 8 的倍数，所以是潜在缺陷。
            hid2.append(c2 if s == 1.0 else int(max(min_ch, round(c2 * s / 8) * 8)))
            hid3.append(c3 if s == 1.0 else int(max(min_ch, round(c3 * s / 8) * 8)))

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, h, 3), Conv(h, h, 3), nn.Conv2d(h, 4 * self.reg_max, 1))
            for x, h in zip(ch, hid2)
        )
        if self.legacy:
            self.cv3 = nn.ModuleList(
                nn.Sequential(Conv(x, h, 3), Conv(h, h, 3), nn.Conv2d(h, self.nc, 1))
                for x, h in zip(ch, hid3)
            )
        else:
            self.cv3 = nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, h, 1)),
                    nn.Sequential(DWConv(h, h, 3), Conv(h, h, 1)),
                    nn.Conv2d(h, self.nc, 1),
                )
                for x, h in zip(ch, hid3)
            )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.head_ch = {"cv2": hid2, "cv3": hid3}  # 供自检脚本核对

    def extra_repr(self) -> str:
        return f"nc={self.nc}, nl={self.nl}, reg_max={self.reg_max}, p2_scale={self.p2_scale}"


# ======================================================================================
# 7. 注册进 ultralytics（YAML 名字 -> 类）+ EGAStem 的借壳补丁
# ======================================================================================
# 借壳槽位：必须在 base_modules 里、不在 repeat_modules 里、parse_model 无身份判断，
# 且**本实验的 YAML 里不使用这个名字**（借用期间它会被临时换成 EGAStem）。
# custom_modules.py 已占用 Focus/DWConvTranspose2d/ConvTranspose/GhostBottleneck/
# ELAN1/SPPELAN/RepNCSPELAN4/AConv/DWConv/Bottleneck 十个，故这里另取一个空闲名字。
_EGA_SLOT = "SPP"

tasks.EGAStem = EGAStem
tasks.LightSAFFMFusion = LightSAFFMFusion
tasks.HFP = HFP
# 同名替换：拿到 base_modules + repeat_modules + 身份判断的全套语义
tasks.C3k2 = C3k2DSA

# 官方的 Detect 原类，留给 configure_detect_head(False) 还原用。
_DetectOfficial = tasks.Detect
tasks.DetectP2Half = DetectP2Half  # 让 YAML 里能直接写 DetectP2Half（调试用）


def configure_detect_head(p2_half: bool) -> None:
    """**显式**选择检测头。必须在构建模型之前调用，且一次进程只选一种。

    ⚠️ 为什么不能无条件替换 `tasks.Detect`：
        parse_model 的头部判断是 `elif m in frozenset({Detect, WorldDetect, ...})`，
        这个 frozenset 是在**函数体内**用 tasks 模块的 globals 现算的 ——
        也就是说它比对的是「`task.Detect` 这个名字当前指向谁」。
        一旦全局替换成 `DetectP2Half`，任何写 `Detect` 的配置都会被解析成
        `DetectP2Half(nc, 16, None, ch)`，**默认 p2_index=0 会把第 0 个尺度（P3）的头减半** ——
        3 尺度配置会静默变成一个 S2/P3/P4 头，而且不报错。

    Args:
        p2_half: True  -> `tasks.Detect = DetectP2Half`（4 尺度 P2 变体，P2 头隐藏通道 x0.5）
                 False -> `tasks.Detect = ultralytics 官方 Detect`（3 尺度，默认）
    """
    tasks.Detect = DetectP2Half if p2_half else _DetectOfficial
    tasks.__dict__["_ega_dsa_detect_head"] = "p2half" if p2_half else "official"


def detect_head_name() -> str:
    """当前 Detect 指向的是官方类还是 P2 变体（写进审计 JSON 用）。"""
    return "p2half" if tasks.Detect is DetectP2Half else "official"


# 默认：官方 Detect（3 尺度）。P2 变体必须显式调用 configure_detect_head(True)。
configure_detect_head(False)


def _install_slot_patch() -> None:
    """在 parse_model 调用期间把 `_EGA_SLOT` 临时绑到 EGAStem，让它落进 base_modules。

    base_modules 是 parse_model **函数体内**用 globals() 现算的 frozenset，
    所以只要在调用期间把某个白名单名字指向 EGAStem，`m in base_modules` 就成立，
    EGAStem 便拿到 `args = [c1, c2_scaled, *args[1:]]` 的语义（含 width 缩放）。
    调用结束后立刻还原 —— 与 custom_modules.py 的借壳机制同一手法，槽位互不重叠。
    """
    original = tasks.parse_model
    if getattr(original, "_ega_dsa_slot_patch", False):
        return

    def parse_model(d, ch, verbose=True):
        saved = tasks.__dict__.get(_EGA_SLOT, None)
        tasks.__dict__[_EGA_SLOT] = EGAStem
        try:
            return original(d, ch, verbose)
        finally:
            if saved is None:
                tasks.__dict__.pop(_EGA_SLOT, None)
            else:
                tasks.__dict__[_EGA_SLOT] = saved

    parse_model._ega_dsa_slot_patch = True
    parse_model.__doc__ = original.__doc__
    tasks.parse_model = parse_model


def _verify_slot(_slot: str = _EGA_SLOT) -> None:
    """借壳槽位自检。

    借用期间 `tasks.<_EGA_SLOT>` 会被指向 EGAStem，所以该名字必须满足：
      1) 在 base_modules 里（否则拿不到 [c1, c2, *args] 语义）；
      2) 不在 repeat_modules 里（否则 args 会被插入重复次数，构造签名对不上）；
      3) **本仓库没有任何 YAML 真的使用这个名字** —— 否则那段 YAML 会被静默改语义，
         例如 `[1024, 5]` 经 base_modules 语义变成 EGAStem(c1, 256, 5)，
         第 3 个参数成了 stride=5，一个 5 倍下采样而不是池化。
    与 custom_modules.py::_verify_borrow_slots 同一手法。
    """
    try:
        import inspect
        import re

        src = inspect.getsource(tasks.parse_model)

        def _names(block: str) -> set:
            i = src.index(f"{block} = frozenset(")
            j = src.index("\n    )", i)
            return set(re.findall(r"\b([A-Z][A-Za-z0-9_]*)\b", src[i:j]))

        base, repeat = _names("base_modules"), _names("repeat_modules")
    except Exception:
        return
    if _slot not in base or _slot in repeat:
        raise RuntimeError(
            f"借壳槽位 {_slot!r} 不再安全（必须：在 base_modules、不在 repeat_modules）。"
            f"请修改 custom_modules_ega_dsa._EGA_SLOT。"
        )
    try:
        import glob

        for f in glob.glob("cfg/*.yaml") + glob.glob("experiments/**/*.yaml", recursive=True):
            txt = open(f, encoding="utf-8", errors="ignore").read()
            if re.search(rf"^\s*-\s*\[[^\]]*\b{_slot}\b", txt, re.M):
                raise RuntimeError(
                    f"cfg {f} 里使用了 {_slot!r}，但该名字被 EGAStem 借壳 —— "
                    f"借用期间它会被解析成 EGAStem。请换槽位或改配置。"
                )
    except RuntimeError:
        raise
    except Exception:
        pass


_install_slot_patch()
_verify_slot()

#: 供自检/文档引用：本文件替换掉的 ultralytics 名字
REPLACED = {
    "C3k2": "C3k2DSA（dsa_n / dsa_ratio 默认 0 -> 与官方逐位一致）",
    "Detect": "DetectP2Half（P2 分支隐藏通道 x p2_scale）",
    _EGA_SLOT: f"借壳槽位，仅供 EGAStem 在 parse_model 期间进入 base_modules",
}
REGISTERED = {
    "EGAStem": "EGA-Stem：Scharr(冻结) + Gaussian(冻结) + ECA + 残差，stride=2",
    "DSA": "Deformable Spatial Attention（DSCN 为纯 PyTorch 等价实现）",
    "LightSAFFMFusion": "LightSAFFM 的 YAML 适配层，输出 = c_x + c_y，drop-in 替换 Concat",
    "HFP": "High Frequency Perception（DCT 双路），可选消融项",
}
