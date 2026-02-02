#!/usr/bin/env python3
"""
Jetson-Mamba-SSM 安装主脚本

统一入口脚本，提供完整的安装、补丁应用、测试功能。

使用方法:
    python scripts/main.py install          # 完整安装 (推荐)
    python scripts/main.py install-base     # 仅安装基础 wheel
    python scripts/main.py patch            # 应用所有补丁
    python scripts/main.py patch-mamba      # 仅应用 mamba-ssm 补丁
    python scripts/main.py patch-onnx       # 仅应用 ONNX 导出补丁
    python scripts/main.py patch-ultra      # 仅应用 ultralytics 补丁
    python scripts/main.py rebuild          # 重新打包 wheel
    python scripts/main.py test             # 运行测试
    python scripts/main.py verify           # 验证安装

作者: jetson-mamba-ssm 项目
日期: 2026-02-02
"""

import os
import sys
import subprocess
from pathlib import Path

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


def run_script(script_path, description):
    """运行指定脚本"""
    script_dir = Path(__file__).parent
    full_path = script_dir / script_path

    if not full_path.exists():
        print_error(f"找不到脚本: {full_path}")
        return False

    print_info(f"运行 {description}...")
    result = subprocess.run([sys.executable, str(full_path)], cwd=script_dir.parent)
    return result.returncode == 0


def cmd_install():
    """完整安装流程"""
    print_header("完整安装流程")

    steps = [
        ("install/install_wheels.py", "安装预编译 wheel"),
    ]

    for script, desc in steps:
        if not run_script(script, desc):
            print_error(f"{desc} 失败")
            return False

    print_header("安装完成")
    print_success("所有组件已安装完成！")
    print_info("\n说明:")
    print("  • mamba_ssm-2.2.4+jetson.whl 已包含所有必要补丁")
    print("  • causal_conv1d-1.6.0+jetson.whl 已包含兼容层")
    print("  • ultralytics 8.3.55+ 已内置支持 TensorRT 10.x")
    print("\n运行以下命令验证安装:")
    print("  python scripts/main.py verify")
    return True


def cmd_install_base():
    """仅安装基础 wheel"""
    print_header("安装基础 wheel")
    return run_script("install/install_wheels.py", "安装预编译 wheel")


def cmd_patch():
    """应用所有补丁"""
    print_header("应用所有补丁")

    steps = [
        ("patch/apply_patches.py", "mamba-ssm 基础补丁"),
        ("patch/apply_onnx_export_patch.py", "ONNX 导出补丁"),
    ]

    for script, desc in steps:
        if not run_script(script, desc):
            print_error(f"{desc} 失败")
            return False

    print_success("所有补丁应用成功！")
    print_info("\n注意: ultralytics 8.3.55+ 已内置支持，无需额外补丁")
    return True


def cmd_patch_mamba():
    """仅应用 mamba-ssm 基础补丁"""
    return run_script("patch/apply_patches.py", "mamba-ssm 基础补丁")


def cmd_patch_onnx():
    """仅应用 ONNX 导出补丁"""
    return run_script("patch/apply_onnx_export_patch.py", "ONNX 导出补丁")


def cmd_patch_ultra():
    """仅应用 ultralytics 补丁（已废弃）"""
    print_header("ultralytics 补丁状态")
    print_info("ultralytics 8.3.55+ 已内置支持 TensorRT 10.x API")
    print_info("mamba_ssm-2.2.4+jetson.whl 已包含 ONNX 导出支持")
    print_success("无需额外补丁")
    return True


def cmd_rebuild():
    """重新打包 wheel"""
    return run_script("install/rebuild_wheels.py", "重新打包 wheel")


def cmd_test():
    """运行测试"""
    print_header("运行测试")
    return run_script("test/test_onnx_tensorrt_export.py", "ONNX/TensorRT 导出测试")


def cmd_verify():
    """验证安装"""
    print_header("验证安装")
    return run_script("test/verify.py", "mamba-ssm 安装验证")


def cmd_backup():
    """备份 .so 文件"""
    import subprocess
    script_dir = Path(__file__).parent
    backup_script = script_dir / "utils" / "backup_so_files.py"

    if not backup_script.exists():
        print_error(f"找不到备份脚本: {backup_script}")
        return False

    print_info("备份编译后的 .so 文件为压缩包...")
    result = subprocess.run(
        [sys.executable, str(backup_script), "--source", "site-packages"],
        cwd=script_dir.parent.parent
    )
    return result.returncode == 0


def cmd_info():
    """显示项目信息"""
    print_header("Jetson-Mamba-SSM 项目信息")

    info = """
项目路径: ./ (当前目录)

版本信息:
  - mamba-ssm:      2.2.4+onnx
  - causal-conv1d:  1.6.0
  - TensorRT:       10.7.0
  - CUDA:           12.6
  - Python:         3.10

目录结构:
  scripts/
    ├── install/    # 安装脚本
    ├── patch/      # 补丁脚本
    ├── test/       # 测试脚本
    └── utils/      # 工具脚本

  wheels/           # 预编译 wheel 文件
  patches/          # Git 格式补丁文件
  src/              # 修改后的源文件
  docs/             # 完整文档

快速开始:
  1. 完整安装:     python scripts/main.py install
  2. 仅应用补丁:   python scripts/main.py patch
  3. 验证安装:     python scripts/main.py verify
  4. 运行测试:     python scripts/main.py test

文档:
  - README.md                              # 项目概述
  - docs/YOLOV10_TENSORRT_EXPORT_GUIDE.md  # ONNX/TensorRT 导出指南
  - docs/JETSON_MAMBA_SSM_BUILD_GUIDE.md   # 编译指南
"""
    print(info)


def show_usage():
    """显示使用说明"""
    print_header("Jetson-Mamba-SSM 安装脚本")

    usage = """
用法: python scripts/main.py <命令>

命令:
  install          完整安装 (安装 wheel)
  install-base     仅安装基础 wheel (不应用补丁)

  patch            应用所有补丁
  patch-mamba      仅应用 mamba-ssm 基础补丁
  patch-onnx       仅应用 ONNX 导出补丁
  patch-ultra      ultralytics 补丁状态 (已废弃)

  rebuild          重新打包包含补丁的 wheel
  backup           备份编译后的 .so 文件

  test             运行 ONNX/TensorRT 导出测试
  verify           验证 mamba-ssm 安装

  info             显示项目信息
  help             显示此帮助信息

示例:
  # 完整安装 (推荐)
  python scripts/main.py install

  # 分步安装
  python scripts/main.py install-base
  python scripts/main.py patch

  # 验证安装
  python scripts/main.py verify

  # 运行测试
  python scripts/main.py test
"""
    print(usage)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        show_usage()
        return 0

    command = sys.argv[1]

    commands = {
        "install": cmd_install,
        "install-base": cmd_install_base,
        "patch": cmd_patch,
        "patch-mamba": cmd_patch_mamba,
        "patch-onnx": cmd_patch_onnx,
        "patch-ultra": cmd_patch_ultra,
        "rebuild": cmd_rebuild,
        "backup": cmd_backup,
        "test": cmd_test,
        "verify": cmd_verify,
        "info": cmd_info,
        "--help": show_usage,
        "-h": show_usage,
        "help": show_usage,
    }

    if command in commands:
        result = commands[command]()
        return 0 if result else 1
    else:
        print_error(f"未知命令: {command}")
        show_usage()
        return 1


if __name__ == "__main__":
    sys.exit(main())
