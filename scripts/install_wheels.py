#!/usr/bin/env python3
"""
Mamba-SSM Jetson 预编译 Wheel 安装脚本

自动安装预编译的 wheel 文件并应用所需的运行时补丁。
"""
import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_ok(msg):
    print(f"  ✓ {msg}")


def print_fail(msg):
    print(f"  ✗ {msg}")


def print_info(msg):
    print(f"  → {msg}")


def get_project_root():
    """获取项目根目录"""
    script_dir = Path(__file__).parent
    return script_dir.parent


def check_cuda():
    """检查 CUDA 环境"""
    print("\n1. 检查 CUDA 环境...")

    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home:
        print_ok(f"CUDA_HOME: {cuda_home}")
    else:
        # 尝试自动检测
        common_paths = [
            '/usr/local/cuda',
            '/usr/local/cuda-12.6',
            '/usr/local/cuda-12.4',
            '/usr/local/cuda-12.2',
        ]
        for path in common_paths:
            if Path(path).exists():
                os.environ['CUDA_HOME'] = path
                print_ok(f"自动检测到 CUDA: {path}")
                return True
        print_fail("CUDA_HOME 未设置，且无法自动检测 CUDA")
        print_info("请设置: export CUDA_HOME=/path/to/cuda")
        return False
    return True


def install_wheels(project_root):
    """安装 wheel 文件"""
    print("\n2. 安装预编译 Wheel 文件...")

    wheels_dir = project_root / 'wheels'
    if not wheels_dir.exists():
        print_fail(f"wheels 目录不存在: {wheels_dir}")
        return False

    # 查找 wheel 文件
    causal_wheel = list(wheels_dir.glob('causal_conv1d-*.whl'))
    mamba_wheel = list(wheels_dir.glob('mamba_ssm-*.whl'))

    if not causal_wheel:
        print_fail("未找到 causal_conv1d wheel 文件")
        return False
    if not mamba_wheel:
        print_fail("未找到 mamba_ssm wheel 文件")
        return False

    causal_wheel = causal_wheel[0]
    mamba_wheel = mamba_wheel[0]

    print_ok(f"找到 causal_conv1d: {causal_wheel.name}")
    print_ok(f"找到 mamba_ssm: {mamba_wheel.name}")

    # 安装 causal_conv1d
    print_info(f"安装 {causal_wheel.name}...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', str(causal_wheel)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print_ok("causal_conv1d 安装成功")
        else:
            print_fail(f"causal_conv1d 安装失败: {result.stderr}")
            return False
    except Exception as e:
        print_fail(f"安装 causal_conv1d 时出错: {e}")
        return False

    # 安装 mamba_ssm
    print_info(f"安装 {mamba_wheel.name}...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', str(mamba_wheel)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print_ok("mamba_ssm 安装成功")
        else:
            print_fail(f"mamba_ssm 安装失败: {result.stderr}")
            return False
    except Exception as e:
        print_fail(f"安装 mamba_ssm 时出错: {e}")
        return False

    return True


def apply_patches(project_root):
    """应用运行时补丁"""
    print("\n3. 应用运行时补丁...")

    apply_script = project_root / 'scripts' / 'apply_patches.py'
    if not apply_script.exists():
        print_fail(f"apply_patches.py 不存在: {apply_script}")
        print_info("请手动复制 src/ 目录中的文件到 site-packages")
        return False

    print_info("运行 apply_patches.py...")
    try:
        # 使用 subprocess 但不使用 -y 自动确认
        result = subprocess.run(
            [sys.executable, str(apply_script)],
            capture_output=True,
            text=True,
            input='y\n'  # 自动确认
        )
        if result.returncode == 0:
            print_ok("补丁应用成功")
        else:
            print_fail(f"补丁应用失败: {result.stderr}")
            return False
    except Exception as e:
        print_fail(f"应用补丁时出错: {e}")
        return False

    return True


def verify_installation(project_root):
    """验证安装"""
    print("\n4. 验证安装...")

    verify_script = project_root / 'scripts' / 'verify.py'
    if not verify_script.exists():
        print_fail("verify.py 不存在，跳过验证")
        return True

    print_info("运行 verify.py...")
    try:
        result = subprocess.run(
            [sys.executable, str(verify_script)],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return result.returncode == 0
    except Exception as e:
        print_fail(f"验证时出错: {e}")
        return False


def main():
    print_header("Mamba-SSM Jetson 预编译 Wheel 安装程序")

    project_root = get_project_root()
    print(f"\n项目目录: {project_root}")

    # 检查 CUDA
    if not check_cuda():
        print("\n" + "=" * 60)
        print("请先设置 CUDA 环境后重试")
        print("=" * 60)
        return 1

    # 安装 wheels
    if not install_wheels(project_root):
        print("\n" + "=" * 60)
        print("Wheel 安装失败")
        print("=" * 60)
        return 1

    # 应用补丁
    if not apply_patches(project_root):
        print("\n" + "=" * 60)
        print("补丁应用失败")
        print("=" * 60)
        return 1

    # 验证
    success = verify_installation(project_root)

    print_header("安装完成")
    if success:
        print("\n🎉 Mamba-SSM 已成功安装!")
        print("\n您可以:")
        print("  - 运行测试: python scripts/verify.py")
        print("  - 使用 run_with_mamba.sh 运行您的脚本")
        print("  - 查看 docs/WHEELS_ARCHIVE.md 了解更多")
        return 0
    else:
        print("\n⚠️  安装可能未完全成功")
        print("请运行 python scripts/check_mamba_install.py 进行详细检查")
        return 1


if __name__ == '__main__':
    sys.exit(main())
