# causal_conv1d 在 Jetson 上的编译

## 概述

`causal_conv1d` 是 Mamba-SSM 的依赖项，提供 CUDA 加速的因果卷积操作。

## 编译方法

### 前置条件

```bash
# 必须设置 CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.6
```

### 从源码编译

```bash
# 克隆仓库
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal_conv1d

# 安装 (约 20-40 分钟)
pip install .
```

### 使用预编译 wheel

```bash
pip install causal_conv1d-1.6.0-cp310-cp310-linux_aarch64.whl
```

## 已知问题

### 1. libc10.so 依赖

**错误**:
```
ImportError: libc10.so: cannot open shared object file: No such file or directory
```

**原因**: `causal_conv1d_cuda` C++ 扩展链接了 `libc10.so`，但 JetPack 的 PyTorch 不提供此独立库。

**解决方案**: 使用 `causal_conv1d_fn` (Python wrapper) 替代 `causal_conv1d_cuda`。

### 2. 函数签名不匹配

**错误**:
```
TypeError: causal_conv1d_fwd() takes 7 positional arguments but 8 were given
```

**原因**: mamba_ssm 的 cpp_functions 调用 `causal_conv1d_fwd` 时传递 8 个参数，但 wrapper 只接受 7 个。

**解决方案**: 在 `mamba_ssm/ops/selective_scan_interface.py` 中使用 `causal_conv1d_fn` 替代直接调用。

## 编译时间

| 设备 | 时间 |
|------|------|
| Jetson Orin (15W) | ~40 分钟 |
| Jetson Orin (MAXN) | ~20 分钟 |

## API 说明

### causal_conv1d_fn

```python
from causal_conv1d import causal_conv1d_fn

def causal_conv1d_fn(
    x,              # (batch, dim, seqlen)
    weight,         # (dim, kernel_size)
    bias=None,      # (dim,)
    seq_idx=None,   # (batch, seqlen) or None
    initial_states=None,  # (batch, dim, state_size)
    final_states_out=None,  # (batch, dim, state_size)
    activation=None, # "silu", "swish", or None
):
    """
    因果 1D 卷积

    返回: (batch, dim, seqlen)
    """
```

## 验证安装

```bash
python3 -c "
from causal_conv1d import causal_conv1d_fn
import torch
x = torch.randn(2, 32, 64, device='cuda')
weight = torch.randn(32, 4, device='cuda')
y = causal_conv1d_fn(x, weight, None, None, None, None, 'silu')
print('causal_conv1d working:', x.shape, '->', y.shape)
"
```

## 版本信息

- **当前版本**: 1.6.0
- **仓库**: https://github.com/Dao-AILab/causal-conv1d
- **作者**: Tri Dao

## 与 mamba-ssm 的关系

```
mamba-ssm
    └── 依赖 causal_conv1d
            ├── causal_conv1d_fn (Python, 可用)
            └── causal_conv1d_cuda (C++, libc10.so 依赖问题)
```

在 Jetson 上，我们使用 `causal_conv1d_fn` 替代 `causal_conv1d_cuda`。
