#!/usr/bin/env python3
"""
重新打包 causal_conv1d wheel，包含 libc10.so 兼容层

将 fix_causal_conv1d.py 直接集成到 causal_conv1d 包中，
无需依赖 sitecustomize.py 自动加载。

作者: jetson-mamba-ssm 项目
日期: 2026-02-02
"""

import os
import sys
import shutil
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime

# 颜色输出
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"


def print_header(msg):
    print(f"\n{CYAN}{'=' * 60}{RESET}")
    print(f"{BLUE}{msg}{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}\n")


def print_info(msg):
    print(f"{BLUE}[INFO]{RESET} {msg}")


def print_success(msg):
    print(f"{GREEN}[✓]{RESET} {msg}")


def print_error(msg):
    print(f"{RED}[✗]{RESET} {msg}")


def print_warning(msg):
    print(f"{YELLOW}[!]{RESET} {msg}")


def rebuild_causal_conv1d_wheel(original_wheel_path, output_dir=None):
    """
    重新打包 causal_conv1d wheel，包含 libc10.so 兼容层

    Args:
        original_wheel_path: 原始 wheel 路径
        output_dir: 输出目录

    Returns:
        新 wheel 文件路径，失败返回 None
    """
    print_header("重新打包 causal_conv1d wheel (含 libc10.so 兼容层)")

    original_wheel = Path(original_wheel_path)
    if not original_wheel.exists():
        print_error(f"找不到原始 wheel: {original_wheel}")
        return None

    if output_dir is None:
        output_dir = original_wheel.parent
    else:
        output_dir = Path(output_dir)

    print_info(f"原始 wheel: {original_wheel}")
    print_info(f"原始大小: {original_wheel.stat().st_size / (1024**2):.1f} MB")

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        extract_dir = temp_path / "extracted"
        extract_dir.mkdir()

        # 1. 解压原始 wheel
        print_info("\n[1/5] 解压原始 wheel...")
        try:
            with zipfile.ZipFile(original_wheel, 'r') as zf:
                zf.extractall(extract_dir)
            print_success(f"解压到: {extract_dir}")
        except Exception as e:
            print_error(f"解压失败: {e}")
            return None

        # 2. 添加兼容层到 causal_conv1d 包
        print_info("\n[2/5] 添加 libc10.so 兼容层...")

        causal_conv1d_dir = extract_dir / "causal_conv1d"
        if not causal_conv1d_dir.exists():
            print_error(f"找不到 causal_conv1d 目录: {causal_conv1d_dir}")
            return None

        # 创建兼容层文件
        compat_code = '''"""
causal_conv1d_cuda compatibility layer for Jetson

This module provides a compatibility layer for causal_conv1d_cuda
to avoid libc10.so dependency issues on NVIDIA Jetson.

It wraps causal_conv1d_fn to provide the causal_conv1d_fwd interface
expected by mamba_ssm.
"""

import sys
from causal_conv1d import causal_conv1d_fn

# Create the compatibility wrapper
def causal_conv1d_fwd(*args):
    """
    Wrapper for causal_conv1d_fn that matches causal_conv1d_cuda.causal_conv1d_fwd interface.

    Args:
        x: Input tensor (B, C, L)
        weight: Weight tensor (D, C)
        bias: Bias tensor (D,) or None
        seq_idx: Sequence indices or None
        initial_states: Initial states or None
        final_states_out: Final states output or None
        silu_activation: Whether to apply SiLU activation

    Returns:
        Output tensor
    """
    if len(args) == 7:
        x, weight, bias, seq_idx, initial_states, final_states_out, silu_activation = args
        return causal_conv1d_fn(
            x=x,
            weight=weight,
            bias=bias,
            seq_idx=seq_idx,
            initial_states=initial_states,
            final_states_out=final_states_out,
            activation="silu" if silu_activation else None
        )
    elif len(args) == 8:
        # 8 argument version (from cpp_functions)
        x, weight, bias, seq_idx, initial_states, out, final_states_out, silu_activation = args
        return causal_conv1d_fn(
            x=x,
            weight=weight,
            bias=bias,
            seq_idx=seq_idx,
            initial_states=initial_states,
            final_states_out=final_states_out,
            activation="silu" if silu_activation else None
        )
    else:
        raise TypeError(f"causal_conv1d_fwd expects 7 or 8 arguments, got {len(args)}")


def causal_conv1d_bwd(*args, **kwargs):
    """Backward pass - not implemented in wrapper"""
    raise NotImplementedError(
        "Backward pass is not available in the compatibility wrapper. "
        "This is expected on Jetson where the full causal_conv1d_cuda is not available."
    )


def causal_conv1d_update(*args, **kwargs):
    """Update function - not implemented in wrapper"""
    raise NotImplementedError(
        "Update function is not available in the compatibility wrapper. "
        "This is expected on Jetson where the full causal_conv1d_cuda is not available."
    )


# Auto-register this module as causal_conv1d_cuda
# This happens when the package is imported
if 'causal_conv1d_cuda' not in sys.modules:
    # Import causal_conv1d to ensure causal_conv1d_fn is available
    import importlib
    import pkg_resources

    # Get the causal_conv1d package
    try:
        causal_conv1d_spec = pkg_resources.locate_distribution('causal_conv1d')
        if causal_conv1d_spec and hasattr(causal_conv1d_spec, '_path'):
            # Add causal_conv1d to sys.modules as causal_conv1d_cuda
            import types
            module = types.ModuleType('causal_conv1d_cuda')
            module.__file__ = __file__
            module.__package__ = 'causal_conv1d_cuda'
            module.causal_conv1d_fwd = causal_conv1d_fwd
            module.causal_conv1d_bwd = causal_conv1d_bwd
            module.causal_conv1d_update = causal_conv1d_update
            sys.modules['causal_conv1d_cuda'] = module
    except Exception:
        # Fallback: just register in sys.modules directly
        import types
        module = types.ModuleType('causal_conv1d_cuda')
        module.__file__ = __file__
        module.__package__ = 'causal_conv1d_cuda'
        module.causal_conv1d_fwd = causal_conv1d_fwd
        module.causal_conv1d_bwd = causal_conv1d_bwd
        module.causal_conv1d_update = causal_conv1d_update
        sys.modules['causal_conv1d_cuda'] = module


__all__ = ['causal_conv1d_fwd', 'causal_conv1d_bwd', 'causal_conv1d_update']
'''

        # 写入兼容层文件
        compat_file = causal_conv1d_dir / "causal_conv1d_cuda_compat.py"
        compat_file.write_text(compat_code)
        print_success("创建 causal_conv1d_cuda_compat.py")

        # 3. 修改 __init__.py 自动加载兼容层
        print_info("\n[3/5] 修改 __init__.py 自动加载兼容层...")

        init_file = causal_conv1d_dir / "__init__.py"
        if init_file.exists():
            init_content = init_file.read_text()

            # 检查是否已经添加过
            if "causal_conv1d_cuda_compat" not in init_content:
                # 在文件开头添加导入
                auto_import = """
# Auto-load libc10.so compatibility layer for Jetson
# This creates causal_conv1d_cuda module using causal_conv1d_fn
try:
    from causal_conv1d import causal_conv1d_cuda_compat
    _causal_conv1d_cuda_available = True
except ImportError:
    _causal_conv1d_cuda_available = False

"""
                init_content = auto_import + init_content
                init_file.write_text(init_content)
                print_success("__init__.py 已更新")
            else:
                print_warning("__init__.py 已包含兼容层导入")
        else:
            print_error(f"找不到 __init__.py: {init_file}")
            return None

        # 4. 更新 METADATA
        print_info("\n[4/5] 更新 METADATA...")

        build_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        metadata_file = extract_dir / "causal_conv1d-1.6.0.dist-info" / "METADATA"

        patch_info = f"""

-------------------------------------------------------------------------------
Jetson-Mamba-SSM Modifications
-------------------------------------------------------------------------------
This wheel has been patched for libc10.so compatibility on NVIDIA Jetson.

Source Code: https://github.com/Dao-AILab/causal-conv1d
Original Version: 1.6.0
Original Tag: v1.6.0
Patch Version: 1.6.0+jetson
Build Date: {build_date}
Patch Project: https://github.com/snowolf-zlex/Jetson-Mamba-SSM/jetson-mamba-ssm

Build Environment:
  - Platform: Linux-5.15.148-tegra-aarch64-with-glibc2.35
  - Architecture: ARM64 (aarch64)
  - JetPack: R36 (release), REVISION: 4.7
  - GPU: NVIDIA Ampere (Orin)
  - CUDA: 12.6
  - Python: 3.10.12

Modified Files:
  - causal_conv1d/causal_conv1d_cuda_compat.py (libc10.so compatibility layer)
  - causal_conv1d/__init__.py (auto-load compatibility layer)

Features Added:
  - causal_conv1d_cuda compatibility layer using causal_conv1d_fn
  - Automatic registration of causal_conv1d_cuda module
  - No dependency on libc10.so

Installation:
  pip install causal_conv1d-1.6.0+jetson-cp310-cp310-linux_aarch64.whl

The compatibility layer is automatically loaded when you import causal_conv1d.
No need for sitecustomize.py or external scripts.

Documentation: docs/
-------------------------------------------------------------------------------
"""

        if metadata_file.exists():
            original_metadata = metadata_file.read_text()
            # 移除旧的补丁信息
            if "Jetson-Mamba-SSM Modifications" in original_metadata:
                parts = original_metadata.split("Jetson-Mamba-SSM Modifications")
                original_metadata = parts[0]
            metadata_file.write_text(original_metadata + patch_info)
            print_success("METADATA 已更新")

        # 5. 重新打包 wheel
        print_info("\n[5/5] 重新打包 wheel...")

        # 生成新文件名 (符合 PEP 427/440)
        # causal_conv1d-1.6.0+jetson-cp310-cp310-linux_aarch64.whl
        parts = original_wheel.stem.split('-')
        if len(parts) >= 5:
            dist = parts[0]
            version = parts[1]
            python_tag = parts[2]
            abi_tag = parts[3]
            platform_tag = parts[4]
            new_name = f"{dist}-{version}+jetson-{python_tag}-{abi_tag}-{platform_tag}"
        else:
            new_name = f"causal_conv1d-1.6.0+jetson-cp310-cp310-linux_aarch64"

        new_wheel_path = output_dir / f"{new_name}.whl"

        with zipfile.ZipFile(new_wheel_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in extract_dir.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(extract_dir)
                    zf.write(file, arcname)

        new_size = new_wheel_path.stat().st_size / (1024**2)
        print_success(f"新 wheel: {new_wheel_path}")
        print_success(f"新大小: {new_size:.1f} MB")

        return new_wheel_path


def main():
    """主函数"""
    print("=" * 60)
    print("causal_conv1d Wheel 重新打包工具")
    print("=" * 60)

    # 查找原始 wheel
    script_dir = Path(__file__).parent.parent.parent
    wheels_dir = script_dir / "wheels"
    original_wheel = wheels_dir / "causal_conv1d-1.6.0-cp310-cp310-linux_aarch64.whl"

    if not original_wheel.exists():
        print_error(f"找不到原始 wheel: {original_wheel}")
        print_info("请确保以下文件存在:")
        print(f"  {original_wheel}")
        return 1

    # 重新打包
    new_wheel = rebuild_causal_conv1d_wheel(original_wheel, wheels_dir)

    if new_wheel is None:
        print_error("重新打包失败")
        return 1

    print("\n" + "=" * 60)
    print_success("重新打包完成！")
    print_info("\n生成的文件:")
    print(f"  {new_wheel}")
    print_info("\n安装方法:")
    print(f"  pip install {new_wheel.name}")
    print_info("\n使用说明:")
    print("  兼容层会自动加载，无需 sitecustomize.py")
    print("  import causal_conv1d  # 自动注册 causal_conv1d_cuda")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
