# Top-level shim to re-export selective_scan_cuda functions

import sys
import os

# Save the original selective_scan_cuda module if it exists
_original_module = sys.modules.get('selective_scan_cuda')

# Find the .so file
_so_path = None

# Check both system and user site-packages
_paths_to_check = []
_paths_to_check.extend(__import__('site').getsitepackages())
_user_site = __import__('site').getusersitepackages()
if _user_site and os.path.exists(_user_site):
    _paths_to_check.append(_user_site)

for site_dir in _paths_to_check:
    so_file = os.path.join(site_dir, 'selective_scan_cuda.cpython-310-aarch64-linux-gnu.so')
    if os.path.exists(so_file):
        _so_path = so_file
        break

# Also check direct path as fallback
if not _so_path:
    _direct_path = '/home/jetson/.local/lib/python3.10/site-packages/selective_scan_cuda.cpython-310-aarch64-linux-gnu.so'
    if os.path.exists(_direct_path):
        _so_path = _direct_path

if _so_path:
    # Remove from sys.modules if already imported
    if 'selective_scan_cuda' in sys.modules:
        del sys.modules['selective_scan_cuda']

    # Load the .so file as a module
    # IMPORTANT: Use 'selective_scan_cuda' as module name to match the PyInit function
    import importlib.util
    spec = importlib.util.spec_from_file_location('selective_scan_cuda', _so_path)
    if spec and spec.loader:
        _so_module = importlib.util.module_from_spec(spec)
        sys.modules['selective_scan_cuda'] = _so_module
        spec.loader.exec_module(_so_module)

        # Export functions
        fwd = _so_module.fwd
        bwd = _so_module.bwd
        print(f"✓ Loaded selective_scan_cuda from {_so_path}")
    else:
        raise ImportError(f"Could not load selective_scan_cuda from {_so_path}")
else:
    def fwd(*args, **kwargs):
        raise ImportError('selective_scan_cuda.so not found')
    def bwd(*args, **kwargs):
        raise ImportError('selective_scan_cuda.so not found')

__all__ = ['fwd', 'bwd']
