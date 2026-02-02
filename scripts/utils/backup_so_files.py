#!/usr/bin/env python3
"""
备份编译后的 .so 文件

将 causal_conv1d_cuda 和 selective_scan_cuda 的 .so 文件备份到单独目录。

作者: jetson-mamba-ssm 项目
日期: 2026-02-02
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
import zipfile

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


def find_so_files():
    """查找所有相关的 .so 文件"""
    print_header("查找编译后的 .so 文件")

    so_files = []
    site_packages = Path("/home/jetson/.local/lib/python3.10/site-packages")

    # 查找 causal_conv1d CUDA 扩展
    for so_file in site_packages.glob("*causal_conv1d*.so"):
        so_files.append({
            'name': so_file.name,
            'path': so_file,
            'type': 'causal_conv1d',
            'arch': 'aarch64' if 'aarch64' in so_file.name else 'x86_64',
            'size': so_file.stat().st_size
        })

    # 查找 selective_scan CUDA 扩展
    for so_file in site_packages.glob("*selective_scan*.so"):
        so_files.append({
            'name': so_file.name,
            'path': so_file,
            'type': 'selective_scan',
            'arch': 'aarch64' if 'aarch64' in so_file.name else 'x86_64',
            'size': so_file.stat().st_size
        })

    # 查找 mamba_ssm 中的 .so 文件
    mamba_ssm_path = site_packages / "mamba_ssm"
    if mamba_ssm_path.exists():
        for so_file in mamba_ssm_path.rglob("*.so"):
            so_files.append({
                'name': f"mamba_ssm/{so_file.relative_to(mamba_ssm_path)}",
                'path': so_file,
                'type': 'mamba_ssm',
                'arch': 'aarch64',
                'size': so_file.stat().st_size
            })

    # 显示找到的文件
    if so_files:
        print_info(f"找到 {len(so_files)} 个 .so 文件:\n")
        for i, so_file in enumerate(so_files, 1):
            arch_tag = f"{GREEN}ARM64{RESET}" if so_file['arch'] == 'aarch64' else f"{YELLOW}x86_64{RESET}"
            size_mb = so_file['size'] / (1024**2)
            print(f"  {i}. {so_file['name']}")
            print(f"     类型: {so_file['type']}, 架构: {arch_tag}, 大小: {size_mb:.1f} MB")
    else:
        print_warning("未找到任何 .so 文件")

    return so_files


def backup_so_files(so_files, backup_dir=None):
    """
    备份 .so 文件为压缩包（不保留原始 .so）

    Args:
        so_files: .so 文件列表
        backup_dir: 备份目录路径
    """
    if not so_files:
        print_error("没有 .so 文件需要备份")
        return False

    # 确定备份目录
    if backup_dir is None:
        project_root = Path(__file__).parent.parent.parent
        backup_dir = project_root / "backup" / "so_files"
    else:
        backup_dir = Path(backup_dir)

    backup_dir = backup_dir.resolve()
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 生成压缩包文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"mamba_ssm_so_files_{timestamp}.tar.gz"
    archive_path = backup_dir / archive_name

    print_info(f"创建压缩包: {archive_path}")

    # 创建临时目录用于打包
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "so_files"
        temp_path.mkdir()

        # 复制 .so 文件到临时目录
        total_size = 0
        file_list = []

        for so_file in so_files:
            # 只备份 ARM64 版本（Jetson 使用）
            if so_file['arch'] != 'aarch64':
                print_info(f"跳过 {so_file['name']} (x86_64 版本)")
                continue

            src = so_file['path']
            dst = temp_path / so_file['name']

            try:
                shutil.copy2(src, dst)
                total_size += so_file['size']
                file_list.append(so_file['name'])
                size_mb = so_file['size'] / (1024**2)
                print_success(f"  {so_file['name']} ({size_mb:.1f} MB)")
            except Exception as e:
                print_error(f"  失败: {so_file['name']}: {e}")

        # 创建 README
        readme_content = f"""# CUDA Extension .so Files Backup

备份时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
JetPack 版本: R36 (release), REVISION: 4.7
Python 版本: 3.10.12
CUDA 版本: 12.6
TensorRT 版本: 10.7.0

## 文件列表

"""
        for name in file_list:
            readme_content += f"- `{name}`\n"

        readme_content += f"""
## 总大小

{total_size / (1024**2):.1f} MB (压缩前)

## 解压方法

```bash
# 解压到临时目录
tar -xzf {archive_name}
cd so_files

# 复制到 site-packages
cp *.so /home/jetson/.local/lib/python3.10/site-packages/
```

## 系统要求

- NVIDIA Jetson (ARM64/aarch64)
- Python 3.10
- CUDA 12.6
- PyTorch 2.x (JetPack 版本)

## 兼容性

| 设备 | 兼容性 |
|------|--------|
| Jetson Orin | ✅ |
| Jetson Xavier | ✅ |
| Jetson Nano | ⚠️ (可能需要重新编译) |

## 注意事项

1. 这些 .so 文件是在特定 JetPack 版本下编译的
2. 不同 JetPack 版本可能需要重新编译
3. ARM64 架构的 .so 文件不能在 x86_64 上使用

---

Generated by jetson-mamba-ssm project
"""

        readme_file = temp_path / "README.md"
        readme_file.write_text(readme_content)

        # 创建压缩包
        import tarfile
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(temp_path, arcname="so_files")

    archive_size = archive_path.stat().st_size / (1024**2)

    print(f"\n{CYAN}{'=' * 60}{RESET}")
    print_success(f"压缩包已创建！")
    print_info(f"文件: {archive_path}")
    print_info(f"大小: {archive_size:.1f} MB")
    print_info(f"压缩率: {archive_size / (total_size / (1024**2)) * 100:.1f}%")
    print_info(f"文件数量: {len(file_list)}")
    print(f"{CYAN}{'=' * 60}{RESET}\n")

    return True


def backup_from_wheels(wheels_dir=None):
    """
    从 wheel 文件中提取并备份 .so 文件

    Args:
        wheels_dir: wheels 目录路径
    """
    print_header("从 wheel 文件中提取 .so 文件")

    if wheels_dir is None:
        project_root = Path(__file__).parent.parent.parent
        wheels_dir = project_root / "wheels"
    else:
        wheels_dir = Path(wheels_dir)

    backup_dir = wheels_dir / "extracted_so" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)

    so_files = []

    # 查找所有 wheel 文件
    for wheel_file in wheels_dir.glob("*.whl"):
        print_info(f"检查 {wheel_file.name}")

        import zipfile
        try:
            with zipfile.ZipFile(wheel_file, 'r') as zf:
                for name in zf.namelist():
                    if name.endswith('.so'):
                        # 提取 .so 文件
                        zf.extract(name, backup_dir)
                        extracted_path = backup_dir / name
                        size_mb = extracted_path.stat().st_size / (1024**2)
                        so_files.append(extracted_path)
                        print_success(f"  {name} ({size_mb:.1f} MB)")
        except Exception as e:
            print_error(f"  失败: {e}")

    if so_files:
        total_size = sum(f.stat().st_size for f in so_files)
        print(f"\n{CYAN}{'=' * 60}{RESET}")
        print_success(f"提取完成！")
        print_info(f"提取目录: {backup_dir}")
        print_info(f"文件数量: {len(so_files)}")
        print_info(f"总大小: {total_size / (1024**2):.1f} MB")
        print(f"{CYAN}{'=' * 60}{RESET}\n")

    return backup_dir if so_files else None


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="备份编译后的 .so 文件")
    parser.add_argument("--source", choices=["site-packages", "wheels"], default="site-packages",
                        help="备份源: site-packages 或 wheels")
    parser.add_argument("--output", "-o", help="输出目录")

    args = parser.parse_args()

    print("=" * 60)
    print("CUDA Extension .so 文件备份工具")
    print("=" * 60)

    if args.source == "site-packages":
        # 从 site-packages 备份
        so_files = find_so_files()

        # 筛选 ARM64 文件
        arm64_files = [f for f in so_files if f['arch'] == 'aarch64']

        if not arm64_files:
            print_error("未找到 ARM64 .so 文件")
            return 1

        backup_so_files(arm64_files, args.output)

    elif args.source == "wheels":
        # 从 wheel 文件提取
        backup_dir = backup_from_wheels()
        if backup_dir:
            print_success(f".so 文件已提取到: {backup_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
