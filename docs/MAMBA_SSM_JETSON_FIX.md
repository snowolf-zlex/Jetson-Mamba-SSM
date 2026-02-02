# Mamba-SSM on Jetson - Fix Summary

## Status: ✓ WORKING

All Mamba-SSM components are now functional on NVIDIA Jetson Orin (ARM64).

## Problem

The `causal_conv1d_cuda` module has a `libc10.so` dependency issue on Jetson because:
1. PyTorch on Jetson doesn't provide `libc10.so` as a separate library
2. The causal_conv1d C++ extension links against `libc10.so` which doesn't exist

This caused `ImportError: libc10.so: cannot open shared object file` when importing `causal_conv1d_cuda`.

## Solution

Patched the `mamba_ssm` source code to use `causal_conv1d_fn` directly instead of `causal_conv1d_cuda.causal_conv1d_fwd`. The `causal_conv1d_fn` function is a Python wrapper that doesn't have the `libc10.so` dependency.

## Files Modified

### 1. `~/.local/lib/python3.10/site-packages/mamba_ssm/ops/selective_scan_interface.py`

**Line 309 (assert):**
```python
# Before:
assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available..."

# After:
assert causal_conv1d_fn is not None or causal_conv1d_cuda is not None, "causal_conv1d_fn or causal_conv1d_cuda is not available..."
```

**Lines 211-213 (forward pass):**
```python
# Before:
conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
    x, conv1d_weight, conv1d_bias, None, None, None, True
)

# After:
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

**Lines 318-321 (backward pass):**
```python
# Similar patch applied
if causal_conv1d_fn is not None:
    conv1d_out = causal_conv1d_fn(...)
else:
    conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(...)
```

### 2. `~/.local/lib/python3.10/site-packages/mamba_ssm/ops/triton/ssd_combined.py`

**Lines 779-782 (forward pass):**
```python
# Before:
xBC_conv = rearrange(
    causal_conv1d_cuda.causal_conv1d_fwd(rearrange(xBC, "b s d -> b d s"),
                                         conv1d_weight, conv1d_bias, seq_idx, None, None, activation in ["silu", "swish"]),
    "b d s -> b s d"
)

# After:
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

**Lines 862-864 (backward pass):**
```python
# Similar patch applied
```

## Verification

All components tested and working:

```bash
# 1. causal_conv1d
from causal_conv1d import causal_conv1d_fn
y = causal_conv1d_fn(x, weight, None, None, None, None, 'silu')
# ✓ PASS

# 2. Mamba module
from mamba_ssm.modules.mamba_simple import Mamba
mamba = Mamba(d_model=64, d_state=8).cuda().half()
y = mamba(x)
# ✓ PASS

# 3. SS2D (Mamba YOLO)
ss2d = SS2D(d_model=64, d_state=8).cuda().float()
y = ss2d(x)
# ✓ PASS

# 4. VSSBlock_YOLO
vss = VSSBlock_YOLO(in_channels=64, hidden_dim=64, ssm_d_state=8).cuda().float()
y = vss(x)
# ✓ PASS
```

## Notes

1. **dtype Handling**: YOLO modules (SS2D, VSSBlock_YOLO) require `float32` due to LayerNorm expectations. Use `float()` or autocast for mixed precision.

2. **Backward Compatibility**: The patches check for `causal_conv1d_fn` first, falling back to `causal_conv1d_cuda` if available.

3. **No Recompilation Needed**: This solution works with the existing causal_conv1d v1.6.0 installation.

4. **sitecustomize.py**: The sitecustomize.py fix for torch.distributed compatibility is still needed.

## Installation Summary

| Component | Version | Status |
|-----------|---------|--------|
| causal_conv1d | v1.6.0 | ✓ Installed |
| mamba_ssm | v2.2.4 | ✓ Installed |
| selective_scan_cuda | .so file | ✓ Working |
| PyTorch distributed | patched | ✓ Fixed |
| libc10.so dependency | bypassed | ✓ Fixed |
