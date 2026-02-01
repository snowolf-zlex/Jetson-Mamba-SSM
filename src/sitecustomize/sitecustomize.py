# sitecustomize to patch missing torch.distributed APIs at import time for compatibility
try:
    import torch
    try:
        # For JetPack/PyTorch on Jetson, some distributed APIs may be missing
        # Provide dummy implementations to prevent import errors

        # all_gather_into_tensor / _all_gather_base
        if not hasattr(torch.distributed, 'all_gather_into_tensor'):
            if hasattr(torch.distributed, '_all_gather_base'):
                torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base
            else:
                # Create a dummy function that raises a clear error when actually used
                def _dummy_all_gather(*args, **kwargs):
                    raise RuntimeError("torch.distributed.all_gather_into_tensor is not available on this PyTorch build")
                torch.distributed.all_gather_into_tensor = _dummy_all_gather

        if not hasattr(torch.distributed, '_all_gather_base'):
            if hasattr(torch.distributed, 'all_gather_into_tensor'):
                torch.distributed._all_gather_base = torch.distributed.all_gather_into_tensor
            else:
                def _dummy_all_gather_base(*args, **kwargs):
                    raise RuntimeError("torch.distributed._all_gather_base is not available on this PyTorch build")
                torch.distributed._all_gather_base = _dummy_all_gather_base

        # reduce_scatter_tensor / _reduce_scatter_base
        if not hasattr(torch.distributed, 'reduce_scatter_tensor'):
            if hasattr(torch.distributed, '_reduce_scatter_base'):
                torch.distributed.reduce_scatter_tensor = torch.distributed._reduce_scatter_base
            else:
                def _dummy_reduce_scatter(*args, **kwargs):
                    raise RuntimeError("torch.distributed.reduce_scatter_tensor is not available on this PyTorch build")
                torch.distributed.reduce_scatter_tensor = _dummy_reduce_scatter

        if not hasattr(torch.distributed, '_reduce_scatter_base'):
            if hasattr(torch.distributed, 'reduce_scatter_tensor'):
                torch.distributed._reduce_scatter_base = torch.distributed.reduce_scatter_tensor
            else:
                def _dummy_reduce_scatter_base(*args, **kwargs):
                    raise RuntimeError("torch.distributed._reduce_scatter_base is not available on this PyTorch build")
                torch.distributed._reduce_scatter_base = _dummy_reduce_scatter_base

    except Exception as e:
        # Silently fail to avoid breaking startup
        pass
except Exception:
    pass

# Load Mamba-SSM compatibility fixes
# This must be loaded BEFORE any mamba_ssm imports
try:
    import sys
    import os
    import importlib.util

    # Use absolute path - don't rely on sys.path
    project_dir = '/home/jetson/pythonProject'
    fix_path = os.path.join(project_dir, 'fix_causal_conv1d.py')

    if os.path.exists(fix_path):
        spec = importlib.util.spec_from_file_location('fix_causal_conv1d', fix_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules['fix_causal_conv1d'] = module
            spec.loader.exec_module(module)
except Exception as e:
    # Print error for debugging
    import traceback
    print(f"sitecustomize.py error: {e}")
    traceback.print_exc()
