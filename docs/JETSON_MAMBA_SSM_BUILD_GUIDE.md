# Jetson 上 Mamba-SSM 编译与安装指南

![](https://img.shields.io/badge/Platform-Jetson%20Orin-32B3E6?logo=nvidia)
![](https://img.shields.io/badge/Architecture-ARM64-E96479?logo=arm)
![](https://img.shields.io/badge/OS-JetPack%205.15.148-06C?logo=nvidia)
![](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)
![](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia)
![](https://img.shields.io/badge/PyTorch-2.5.0a0-EE4C2C?logo=pytorch)
![](https://img.shields.io/badge/Status-Working-00C853)

## 目录

- [概述](#概述)
- [为什么需要在 Jetson 上编译](#为什么需要在-jetson-上编译)
- [编译环境](#编译环境)
- [编译方法](#编译方法)
- [编译过程中的错误与修正](#编译过程中的错误与修正)
- [运行时修复 (libc10.so 依赖问题)](#运行时修复-libc10so-依赖问题)
- [验证效果](#验证效果)
- [使用预编译的 wheel 文件](#使用预编译的-wheel-文件)
- [Mamba 对 YOLO 的作用](#mamba-对-yolo-的作用)

---

## 概述

本文档记录了在 NVIDIA Jetson Orin (ARM64 架构) 上编译和安装 Mamba-SSM 的完整过程，包括编译方法、遇到的错误及解决方案，以及最终的验证结果。

**编译状态**: ✓ 成功
**测试状态**: ✓ 通过

---

## 为什么需要在 Jetson 上编译

### 架构差异

| 平台 | 架构 | 说明 |
|------|------|------|
| x86_64 (PC/服务器) | AMD64 | Mamba-SSM 官方提供的预编译包 |
| Jetson Orin | aarch64 (ARM64) | 需要本地从源码编译 |

Mamba-SSM 官方只提供 x86_64 架构的预编译包，而 Jetson 使用的是 ARM64 架构，因此必须在 Jetson 设备上本地编译。

---

## 编译环境

```
设备: NVIDIA Jetson Orin
系统: Linux 5.15.148-tegra (JetPack)
Python: 3.10
CUDA: 12.6 (/usr/local/cuda-12.6)
PyTorch: 2.5.0a0+872d972e41.nv24.8 (JetPack 版本)
pip: 22.0.2
```

---

## 编译方法

### 1. 设置环境变量

```bash
# 必须设置 CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 2. 编译 causal_conv1d (v1.6.0)

```bash
# 方法 1: 从源码编译 (约 20-40 分钟)
cd causal_conv1d
pip install .

# 方法 2: 使用预编译的 wheel 文件 (快速安装)
pip install causal_conv1d-1.6.0-cp310-cp310-linux_aarch64.whl
```

**编译时间**: 约 20-40 分钟 (取决于 Jetson 性能模式)

### 3. 编译 mamba-ssm (v2.2.4)

```bash
# 方法 1: 从源码编译
cd mamba-main  # 或 mamba-ssm 源码目录
pip install .

# 方法 2: 使用预编译的 wheel 文件 (推荐)
pip install mamba_ssm-2.2.4-cp310-cp310-linux_aarch64.whl
```

**编译时间**: 约 1-2 分钟

### 4. 编译 selective_scan_cuda

selective_scan_cuda 通常作为 mamba-ssm 的依赖自动编译。编译后会生成 `.so` 文件:

```
/home/jetson/.local/lib/python3.10/site-packages/selective_scan_cuda.cpython-310-aarch64-linux-gnu.so
```

---

## 编译过程中的错误与修正

### 错误 1: CUDA_HOME 未设置

**错误信息**:
```
RuntimeError: CUDA unavailable or invalid cuda_HOME
```

**解决方案**:
```bash
export CUDA_HOME=/usr/local/cuda-12.6
```

---

### 错误 2: editable 模式不支持

**错误信息**:
```
ERROR: Project uses a build backend that is missing the 'build_editable' hook
```

**解决方案**: 使用普通安装模式而非 editable 模式
```bash
# 错误方式
pip install -e .

# 正确方式
pip install .
```

---

### 错误 3: torch.distributed API 缺失

**错误信息**:
```
AttributeError: module 'torch.distributed' has no attribute '_all_gather_base'
```

**解决方案**: 创建 `/home/jetson/.local/lib/python3.10/site-packages/sitecustomize.py`

```python
# sitecustomize to patch missing torch.distributed APIs
try:
    import torch
    if not hasattr(torch.distributed, 'all_gather_into_tensor'):
        if hasattr(torch.distributed, '_all_gather_base'):
            torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base
        else:
            def _dummy_all_gather(*args, **kwargs):
                raise RuntimeError("API not available")
            torch.distributed.all_gather_into_tensor = _dummy_all_gather
    # 类似处理 _all_gather_base, reduce_scatter_tensor, _reduce_scatter_base
except Exception:
    pass
```

---

### 错误 4: selective_scan_cuda_core 未找到

**错误信息**:
```
ImportError: cannot import name 'selective_scan_cuda_core'
```

**解决方案**: 在 `mamba_yolo.py` 中创建 wrapper 类

```python
try:
    import selective_scan_cuda_core
except ImportError:
    import selective_scan_cuda
    class _SelectiveScanCudaCore:
        @staticmethod
        def fwd(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, ...):
            result = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
            return result[0], result[1], *result[2:]
```

---

### 错误 5: SS2D 空间维度不匹配

**错误信息**:
```
RuntimeError: The size of tensor a (31) must match the size of tensor b (32)
```

**原因**: conv2d 使用偶数卷积核 (如 kernel_size=4) 导致输出尺寸减少

**解决方案**: 在 `mamba_yolo.py` forward 中添加中心裁剪

```python
# 处理 conv2d 导致的尺寸不匹配
if y.shape[2] != z1.shape[2] or y.shape[3] != z1.shape[3]:
    h_diff = z1.shape[2] - y.shape[2]
    w_diff = z1.shape[3] - y.shape[3]
    if h_diff > 0 or w_diff > 0:
        h_start = h_diff // 2
        h_end = h_start + y.shape[2]
        w_start = w_diff // 2
        w_end = w_start + y.shape[3]
        z1 = z1[:, :, h_start:h_end, w_start:w_end]
```

---

## 运行时修复 (libc10.so 依赖问题)

### 问题描述

`causal_conv1d_cuda` 模块依赖于 `libc10.so`，但 PyTorch on Jetson 不提供这个独立库文件。

**错误信息**:
```
ImportError: libc10.so: cannot open shared object file: No such file or directory
```

### 解决方案

修改 mamba_ssm 源码，使用 `causal_conv1d_fn` 直接调用，绕过 `causal_conv1d_cuda`。

#### 修改文件 1: selective_scan_interface.py

**位置**: `/home/jetson/.local/lib/python3.10/site-packages/mamba_ssm/ops/selective_scan_interface.py`

**修改 1 (line 309)**: 修改 assert 语句
```python
# 修改前
assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available..."

# 修改后
assert causal_conv1d_fn is not None or causal_conv1d_cuda is not None, "causal_conv1d_fn or causal_conv1d_cuda is not available..."
```

**修改 2 (line 211-213)**: forward pass
```python
# 修改前
conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
    x, conv1d_weight, conv1d_bias, None, None, None, True
)

# 修改后
if causal_conv1d_fn is not None:
    conv1d_out = causal_conv1d_fn(
        x, conv1d_weight, conv1d_bias,
        seq_idx=None, initial_states=None, final_states_out=None,
        activation="silu"
    )
elif causal_conv1d_cuda is not None:
    conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
        x, conv1d_weight, conv1d_bias, None, None, None, True
    )
```

**修改 3 (line 318-321)**: backward pass (类似修改)

#### 修改文件 2: ssd_combined.py

**位置**: `/home/jetson/.local/lib/python3.10/site-packages/mamba_ssm/ops/triton/ssd_combined.py`

**修改 1 (line 779-782)**: forward pass
```python
# 修改前
xBC_conv = rearrange(
    causal_conv1d_cuda.causal_conv1d_fwd(rearrange(xBC, "b s d -> b d s"),
                                         conv1d_weight, conv1d_bias, seq_idx, None, None, activation in ["silu", "swish"]),
    "b d s -> b s d"
)

# 修改后
if causal_conv1d_fn is not None:
    conv_out = causal_conv1d_fn(
        rearrange(xBC, "b s d -> b d s"),
        conv1d_weight, conv1d_bias,
        seq_idx=seq_idx, initial_states=None, final_states_out=None,
        activation=activation if activation in ["silu", "swish"] else None
    )
else:
    conv_out = causal_conv1d_cuda.causal_conv1d_fwd(...)
xBC_conv = rearrange(conv_out, "b d s -> b s d")
```

**修改 2 (line 862-864)**: backward pass (类似修改)

---

## 验证效果

### 测试脚本

```python
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './yolov10_main')

import torch
import yolov10_main.ultralytics.nn.AddModules.Structure.mamba_yolo as mamba_yolo

# 1. 测试 causal_conv1d
from causal_conv1d import causal_conv1d_fn
x = torch.randn(2, 32, 64, device='cuda')
weight = torch.randn(32, 4, device='cuda')
y = causal_conv1d_fn(x, weight, None, None, None, None, 'silu')
print("✓ causal_conv1d: {} -> {}".format(x.shape, y.shape))

# 2. 测试 Mamba 模块
from mamba_ssm.modules.mamba_simple import Mamba
mamba = Mamba(d_model=64, d_state=8).cuda().half()
x = torch.randn(2, 32, 64, device='cuda', dtype=torch.float16)
y = mamba(x)
print("✓ Mamba: {} -> {}".format(x.shape, y.shape))

# 3. 测试 SS2D
ss2d = mamba_yolo.SS2D(d_model=64, d_state=8).cuda().float()
x = torch.randn(2, 64, 32, 32, device='cuda', dtype=torch.float32)
y = ss2d(x)
print("✓ SS2D: {} -> {}".format(x.shape, y.shape))

# 4. 测试 VSSBlock_YOLO
vss = mamba_yolo.VSSBlock_YOLO(in_channels=64, hidden_dim=64, ssm_d_state=8).cuda().float()
y = vss(x)
print("✓ VSSBlock_YOLO: {} -> {}".format(x.shape, y.shape))
```

### 测试结果

| 组件 | 状态 | 输入形状 | 输出形状 |
|------|------|----------|----------|
| causal_conv1d_fn | ✓ Pass | (2, 32, 64) | (2, 32, 64) |
| Mamba 模块 | ✓ Pass | (2, 32, 64) | (2, 32, 64) |
| selective_scan_cuda | ✓ Pass | - | - |
| SS2D | ✓ Pass | (2, 64, 32, 32) | (2, 64, 32, 32) |
| VSSBlock_YOLO | ✓ Pass | (2, 64, 32, 32) | (2, 64, 32, 32) |

---

## 使用预编译的 wheel 文件

项目已将编译好的 wheel 文件存档在 `/home/jetson/pythonProject/` 目录下:

### 文件列表

```
/home/jetson/pythonProject/
├── causal_conv1d-1.6.0-cp310-cp310-linux_aarch64.whl  (约 XX MB)
└── mamba_ssm-2.2.4-cp310-cp310-linux_aarch64.whl     (约 324 MB)
```

### 安装方法

```bash
# 设置 CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.6

# 安装 causal_conv1d
pip install /home/jetson/pythonProject/causal_conv1d-1.6.0-cp310-cp310-linux_aarch64.whl

# 安装 mamba-ssm
pip install /home/jetson/pythonProject/mamba_ssm-2.2.4-cp310-cp310-linux_aarch64.whl
```

### 兼容性说明

这些 wheel 文件仅在以下环境兼容:
- **架构**: ARM64 (aarch64)
- **Python**: 3.10
- **CUDA**: 12.x
- **设备**: NVIDIA Jetson (Orin/Xavier/Nano 系列)

---

## Mamba 对 YOLO 的作用

### 什么是 Mamba?

Mamba 是一种基于**状态空间模型 (State Space Model, SSM)** 的新架构，可以作为 Transformer 的替代方案。

### 为什么在 YOLO 中使用 Mamba?

| 特性 | CNN | Transformer | Mamba (SSM) |
|------|-----|-------------|-------------|
| 感受野 | 局部 | 全局 | 全局 |
| 计算复杂度 | O(N) | O(N²) | **O(N)** |
| 显存占用 | 低 | 高 | **低** |
| 长序列建模 | 弱 | 强 | **强** |

### Mamba + TensorRT 效率提升

在 Jetson 上使用 Mamba + TensorRT 的优势:

1. **更低的显存占用**: O(N) 复杂度 vs Transformer 的 O(N²)
2. **更快的推理速度**: 无需注意力矩阵计算
3. **TensorRT 友好**: Mamba 的操作更容易被 TensorRT 优化
4. **适合边缘设备**: 在 Jetson 这类资源受限设备上性能更佳

### SS2D 和 VSSBlock_YOLO

- **SS2D (Selective Scan 2D)**: 将 2D 图像展平为 1D 序列，使用四个方向的 selective scan (上下左右) 捕获多方向特征
- **VSSBlock_YOLO**: 视觉状态空间块，专门为 YOLO 设计的 Mamba 模块

---

## 编译时间记录

| 模块 | 编译时间 | 备注 |
|------|----------|------|
| causal_conv1d v1.6.0 | ~20-40 分钟 | 首次编译较慢 |
| mamba-ssm v2.2.4 | ~1-2 分钟 | 依赖已安装后较快 |
| selective_scan_cuda | 自动编译 | 随 mamba-ssm 一起 |

---

## 常见问题

### Q1: 为什么不用 pip 直接安装?

A: PyPI 上的预编译包是 x86_64 架构，Jetson 是 ARM64 架构，不兼容。

### Q2: 每次都要重新编译吗?

A: 不需要。使用本项目存档的 wheel 文件可以直接安装。

### Q3: dtype 相关错误怎么办?

A: YOLO 模块 (SS2D, VSSBlock_YOLO) 需要 float32，使用 `.float()` 或 autocast。

### Q4: 如何确认安装成功?

A: 运行验证脚本，所有组件应该输出 ✓ Pass。

---

## 参考资料

- [Mamba-SSM GitHub](https://github.com/state-spaces/mamba)
- [causal_conv1d GitHub](https://github.com/Dao-AILab/causal-conv1d)
- [Jetson下载](https://developer.nvidia.com/embedded/jetson-platform)

---

**文档版本**: 1.0
**更新日期**: 2026-02-01
**维护者**: jetson@pythonProject
