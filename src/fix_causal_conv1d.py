#!/usr/bin/env python3
"""
创建 causal_conv1d_cuda 兼容层

IMPORTANT: This module must be imported BEFORE any mamba_ssm imports

This version forces reload of causal_conv1d modules
"""
import sys

# Set sys.modules['causal_conv1d_cuda'] BEFORE importing anything else
sys.modules['causal_conv1d_cuda'] = type(sys)('causal_conv1d_cuda')

# Force remove any cached causal_conv1d modules
for key in list(sys.modules.keys()):
    if 'causal_conv1d' in key:
        del sys.modules[key]

# Now import causal_conv1d to get causal_conv1d_fn
import causal_conv1d

# Get causal_conv1d_fn
from causal_conv1d import causal_conv1d_fn

# Create wrapper
def causal_conv1d_fwd(*args):
    if len(args) == 7:
        x, weight, bias, seq_idx, initial_states, final_states_out, silu_activation = args
        return causal_conv1d_fn(
            x=x, weight=weight, bias=bias, seq_idx=seq_idx,
            initial_states=initial_states, final_states_out=final_states_out,
            activation="silu" if silu_activation else None
        )
    else:
        # For 8 arguments (from cpp_functions), call causal_conv1d_fn directly
        # The cpp_function signature is: x, weight, bias, seq_idx, initial_states, out, final_states_out, silu_activation
        # But causal_conv1d_fn creates its own 'out', so we can ignore the 'out' parameter
        x, weight, bias, seq_idx, initial_states, out, final_states_out, silu_activation = args
        return causal_conv1d_fn(
            x=x, weight=weight, bias=bias, seq_idx=seq_idx,
            initial_states=initial_states, final_states_out=final_states_out,
            activation="silu" if silu_activation else None
        )

def causal_conv1d_bwd(*args, **kwargs):
    raise NotImplementedError("Backward pass not available in wrapper")

def causal_conv1d_update(*args, **kwargs):
    raise NotImplementedError("Update not available in wrapper")

# Set the functions on the module
sys.modules['causal_conv1d_cuda'].causal_conv1d_fwd = causal_conv1d_fwd
sys.modules['causal_conv1d_cuda'].causal_conv1d_bwd = causal_conv1d_bwd
sys.modules['causal_conv1d_cuda'].causal_conv1d_update = causal_conv1d_update

print(f"✓ 创建 causal_conv1d_cuda 兼容层")
print(f"  使用 causal_conv1d_fn 实现 (支持7和8参数)")
