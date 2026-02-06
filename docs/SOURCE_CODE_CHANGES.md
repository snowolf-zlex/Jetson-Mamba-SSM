# Mamba-SSM Jetson 源代码修改文档

## 修改概述

| 项目 | 内容 |
|------|------|
| **修改文件** | 2 个 |
| **修改函数** | 4 个 |
| **修改策略** | 调用替换 |

---

## 修改的文件

```
mamba_ssm/ops/selective_scan_interface.py
mamba_ssm/ops/triton/ssd_combined.py
```

---

## 修改的函数

| # | 函数 | 文件 |
|---|------|------|
| 1 | `MambaInnerFn.forward()` | selective_scan_interface.py |
| 2 | `MambaInnerFn.backward()` | selective_scan_interface.py |
| 3 | `MambaSplitConv1dScanCombinedFn.forward()` | ssd_combined.py |
| 4 | `MambaSplitConv1dScanCombinedFn.backward()` | ssd_combined.py |

---

## 替换说明

### 替换内容

将 `causal_conv1d_cuda.causal_conv1d_fwd()` 调用替换为 `causal_conv1d_fn()`

### 替换位置

所有卷积操作调用处：
- 前向传播 (forward)
- 反向传播重计算 (backward)

---

## 详细修改

### 文件 1: mamba_ssm/ops/selective_scan_interface.py

#### 修改 1: 断言放宽

**位置:** 第 309 行

```python
# 原始
assert causal_conv1d_cuda is not None, "..."

# 修改后
assert causal_conv1d_fn is not None or causal_conv1d_cuda is not None, "..."
```

---

#### 修改 2: MambaInnerFn.forward() - 卷积调用替换

**位置:** 第 208-224 行

```python
# 原始
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

---

#### 修改 3: MambaInnerFn.backward() - 重计算卷积替换

**位置:** 第 318-329 行

```python
# 原始
if ctx.checkpoint_lvl == 1:
    delta = rearrange(...)

# 修改后
if ctx.checkpoint_lvl == 1:
    if causal_conv1d_fn is not None:
        conv1d_out = causal_conv1d_fn(
            x, conv1d_weight, conv1d_bias,
            seq_idx=None, initial_states=None, final_states_out=None,
            activation="silu"
        )
    else:
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
    delta = rearrange(...)
```

---

### 文件 2: mamba_ssm/ops/triton/ssd_combined.py

#### 修改 1: MambaSplitConv1dScanCombinedFn.forward() - 卷积调用替换

**位置:** 第 776-791 行

```python
# 原始
xBC_conv = rearrange(
    causal_conv1d_cuda.causal_conv1d_fwd(
        rearrange(xBC, "b s d -> b d s"),
        conv1d_weight, conv1d_bias, seq_idx, None, None,
        activation in ["silu", "swish"]
    ),
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
    conv_out = causal_conv1d_cuda.causal_conv1d_fwd(
        rearrange(xBC, "b s d -> b d s"),
        conv1d_weight, conv1d_bias, seq_idx, None, None,
        activation in ["silu", "swish"]
    )
xBC_conv = rearrange(conv_out, "b d s -> b s d")
```

---

#### 修改 2: MambaSplitConv1dScanCombinedFn.backward() - 重计算卷积替换

**位置:** 第 859-874 行

```python
# 原始
xBC_conv = rearrange(
    causal_conv1d_cuda.causal_conv1d_fwd(
        rearrange(xBC, "b s d -> b d s"),
        conv1d_weight, conv1d_bias, seq_idx, None, None,
        ctx.activation in ["silu", "swish"]
    ),
    "b d s -> b s d"
)

# 修改后
if causal_conv1d_fn is not None:
    conv_out = causal_conv1d_fn(
        rearrange(xBC, "b s d -> b d s"),
        conv1d_weight, conv1d_bias,
        seq_idx=seq_idx, initial_states=None, final_states_out=None,
        activation=ctx.activation if ctx.activation in ["silu", "swish"] else None
    )
else:
    conv_out = causal_conv1d_cuda.causal_conv1d_fwd(
        rearrange(xBC, "b s d -> b d s"),
        conv1d_weight, conv1d_bias, seq_idx, None, None,
        ctx.activation in ["silu", "swish"]
    )
xBC_conv = rearrange(conv_out, "b d s -> b s d")
```

---

## 替换对照表

| 原始调用 | 替换为 |
|----------|--------|
| `causal_conv1d_cuda.causal_conv1d_fwd(x, w, b, None, None, None, True)` | `causal_conv1d_fn(x, w, b, seq_idx=None, initial_states=None, final_states_out=None, activation="silu")` |
| `causal_conv1d_cuda.causal_conv1d_fwd(x, w, b, seq_idx, None, None, flag)` | `causal_conv1d_fn(x, w, b, seq_idx=seq_idx, initial_states=None, final_states_out=None, activation=activation)` |

---

## 未修改的内容

| 内容 | 说明 |
|------|------|
| `causal_conv1d_fn` | 来自 `causal_conv1d` 包，未修改 |
| `causal_conv1d_cuda` | 来自 `causal_conv1d` 包，未修改 |
| CUDA 内核 | 未修改 |
| 其他函数 | 未修改 |

---

## 修改原理

```
原始调用链:
mamba_ssm → causal_conv1d_cuda.causal_conv1d_fwd() → CUDA 内核
           └─ 直接依赖 C++ 扩展 ─┘

修改后调用链:
mamba_ssm → causal_conv1d_fn() → CUDA 内核
           └─ Python 包装器 ──┘
```

`causal_conv1d_fn` 是 `causal_conv1d_cuda` 的 Python 包装器，两者底层调用相同的 CUDA 内核。使用 Python 包装器避免了某些环境下的依赖问题。
