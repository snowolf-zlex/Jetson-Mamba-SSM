#!/usr/bin/env python3
"""
应用 Mamba-SSM Jetson 补丁

自动检测 mamba-ssm 安装位置并应用所需的补丁。
"""
import os
import sys
import shutil
from pathlib import Path


def get_site_packages():
    """获取 site-packages 路径"""
    for path in sys.path:
        if 'site-packages' in path and 'dist-packages' not in path:
            return Path(path)
    raise RuntimeError("无法找到 site-packages 目录")


def apply_patch(src_file, dst_file, backup=True):
    """应用补丁文件"""
    dst_file = Path(dst_file)
    if backup and dst_file.exists():
        backup_file = dst_file.with_suffix('.py.bak')
        shutil.copy2(dst_file, backup_file)
        print(f"  备份: {dst_file} -> {backup_file}")

    shutil.copy2(src_file, dst_file)
    print(f"  应用: {dst_file}")


def main():
    print("=" * 60)
    print("Mamba-SSM Jetson 补丁应用工具")
    print("=" * 60)

    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    src_dir = project_root / 'src'

    # 获取 site-packages
    site_packages = get_site_packages()
    print(f"\n检测到 site-packages: {site_packages}")

    patches = [
        # sitecustomize.py - torch.distributed API 修复
        {
            'src': src_dir / 'sitecustomize' / 'sitecustomize.py',
            'dst': site_packages / 'sitecustomize.py',
            'name': 'torch.distributed API 修复'
        },
        # selective_scan_interface.py - libc10.so 修复
        {
            'src': src_dir / 'mamba_ssm' / 'ops' / 'selective_scan_interface.py',
            'dst': site_packages / 'mamba_ssm' / 'ops' / 'selective_scan_interface.py',
            'name': 'libc10.so 依赖修复 (selective_scan_interface)'
        },
        # ssd_combined.py - libc10.so 修复
        {
            'src': src_dir / 'mamba_ssm' / 'ops' / 'triton' / 'ssd_combined.py',
            'dst': site_packages / 'mamba_ssm' / 'ops' / 'triton' / 'ssd_combined.py',
            'name': 'libc10.so 依赖修复 (ssd_combined)'
        },
        # distributed_utils.py - 分布式 API 兼容
        {
            'src': src_dir / 'mamba_ssm' / 'distributed' / 'distributed_utils.py',
            'dst': site_packages / 'mamba_ssm' / 'distributed' / 'distributed_utils.py',
            'name': '分布式 API 兼容'
        },
    ]

    print("\n将应用以下补丁:")
    for i, patch in enumerate(patches, 1):
        status = "✓" if patch['src'].exists() else "✗"
        print(f"  {i}. [{status}] {patch['name']}")

    response = input("\n是否继续? (y/n): ").strip().lower()
    if response != 'y':
        print("取消操作")
        return

    print("\n应用补丁...")
    for patch in patches:
        if not patch['src'].exists():
            print(f"  跳过: {patch['name']} (源文件不存在)")
            continue
        try:
            apply_patch(patch['src'], patch['dst'], backup=True)
        except Exception as e:
            print(f"  失败: {patch['name']}: {e}")

    print("\n" + "=" * 60)
    print("补丁应用完成!")
    print("=" * 60)
    print("\n运行以下命令验证安装:")
    print("  python scripts/verify.py")


if __name__ == '__main__':
    main()
