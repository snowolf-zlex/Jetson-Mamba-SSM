#!/usr/bin/env python3
"""
Mamba-SSM 安装检查脚本
全面检查 Mamba 在 Jetson 上的编译安装状态
"""
import os
import sys
import subprocess
import traceback
from pathlib import Path

# 颜色输出
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_project_root():
    """获取项目根目录"""
    script_dir = Path(__file__).parent
    return script_dir.parent

def print_header(title):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_section(title):
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}━━━ {title} ━━━{Colors.ENDC}\n")

def print_ok(msg):
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")

def print_fail(msg):
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")

def print_warning(msg):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")

def print_info(msg):
    print(f"  {msg}")

def get_system_info():
    """获取系统信息"""
    print_section("1. 系统环境检查")

    # Python 版本
    python_version = sys.version_info
    print_info(f"Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version >= (3, 9):
        print_ok(f"Python 版本符合要求 (>= 3.9)")
    else:
        print_fail(f"Python 版本过低，需要 3.9+，当前 {python_version.major}.{python_version.minor}")

    # 架构
    import platform
    arch = platform.machine()
    print_info(f"系统架构: {arch}")
    if arch in ['aarch64', 'arm64']:
        print_ok(f"ARM64 架构 (Jetson)")
    elif arch in ['x86_64', 'AMD64']:
        print_warning(f"x86_64 架构 (可直接使用预编译 wheel)")
    else:
        print_warning(f"未知架构: {arch}")

    # PyTorch
    try:
        import torch
        print_info(f"PyTorch 版本: {torch.__version__}")
        print_info(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print_info(f"CUDA 版本: {torch.version.cuda}")
            print_info(f"GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print_info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print_ok(f"CUDA 环境正常")
        else:
            print_warning(f"CUDA 不可用")
    except ImportError:
        print_fail(f"PyTorch 未安装")

def check_cuda_extensions():
    """检查 CUDA 扩展模块"""
    print_section("2. CUDA 扩展模块检查")

    project_root = get_project_root()
    src_dir = project_root / 'src'

    # selective_scan_cuda shim
    print_info("检查 selective_scan_cuda shim...")
    try:
        # 添加项目路径
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        import selective_scan_cuda
        print_ok(f"selective_scan_cuda shim 导入成功")
        print_info(f"  可用函数: {selective_scan_cuda.__all__}")

        # 检查函数是否可调用
        if callable(selective_scan_cuda.fwd):
            print_ok(f"  fwd 函数可调用")
        else:
            print_fail(f"  fwd 函数不可调用")

        if callable(selective_scan_cuda.bwd):
            print_ok(f"  bwd 函数可调用")
        else:
            print_fail(f"  bwd 函数不可调用")

    except ImportError as e:
        print_fail(f"selective_scan_cuda shim 导入失败: {e}")

    # selective_scan_cuda_core
    print_info("\n检查 selective_scan_cuda_core (核心 CUDA 扩展)...")
    try:
        import selective_scan_cuda_core
        print_ok(f"selective_scan_cuda_core 导入成功")

        # 检查 fwd/bwd 函数
        if hasattr(selective_scan_cuda_core, 'fwd'):
            print_ok(f"  fwd 函数存在")
        if hasattr(selective_scan_cuda_core, 'bwd'):
            print_ok(f"  bwd 函数存在")

    except ImportError as e:
        print_fail(f"selective_scan_cuda_core 导入失败: {e}")
        print_info(f"  这意味着 Mamba CUDA 扩展未正确编译安装")

    # mamba_ssm
    print_info("\n检查 mamba_ssm 包...")
    try:
        import mamba_ssm
        print_ok(f"mamba_ssm 包导入成功")
        print_info(f"  版本: {getattr(mamba_ssm, '__version__', 'unknown')}")
        print_info(f"  路径: {mamba_ssm.__file__}")
    except ImportError as e:
        print_fail(f"mamba_ssm 包导入失败: {e}")

def check_mamba_modules():
    """检查 Mamba 核心模块"""
    print_section("3. Mamba 核心模块检查")

    # Mamba 模块
    print_info("检查 Mamba 模块...")
    try:
        from mamba_ssm import Mamba
        print_ok(f"Mamba 类导入成功")

        # 创建实例测试
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        mamba = Mamba(
            d_model=16,
            d_state=8,
            d_conv=4,
            expand=2,
        ).to(device)
        print_ok(f"Mamba 实例创建成功 (d_model=16)")

        # 前向传播测试
        x = torch.randn(1, 10, 16).to(device)
        with torch.no_grad():
            y = mamba(x)
        print_ok(f"Mamba 前向传播成功: {x.shape} -> {y.shape}")

    except ImportError as e:
        print_fail(f"Mamba 导入失败: {e}")
    except Exception as e:
        print_fail(f"Mamba 测试失败: {e}")

    # Mamba2 模块
    print_info("\n检查 Mamba2 模块...")
    try:
        from mamba_ssm import Mamba2
        print_ok(f"Mamba2 类导入成功")
    except ImportError as e:
        print_warning(f"Mamba2 导入失败 (可选): {e}")

def check_yolo_mamba():
    """检查 YOLO Mamba 集成"""
    print_section("4. YOLO Mamba 集成检查")

    project_root = get_project_root()
    yolo_dir = project_root / 'src' / 'yolo'

    print_info("检查 mamba_yolo 模块...")
    try:
        if str(yolo_dir) not in sys.path:
            sys.path.insert(0, str(yolo_dir))

        from mamba_yolo import SS2D, VSSBlock_YOLO, XSSBlock, CrossScan, CrossMerge
        print_ok(f"mamba_yolo 导入成功")
        print_info(f"  可用类: SS2D, VSSBlock_YOLO, XSSBlock, CrossScan, CrossMerge")

        # 测试 SS2D
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print_info("\n  测试 SS2D 模块...")
        ss2d = SS2D(d_model=16, d_state=8, ssm_ratio=2.0, d_conv=3)
        x = torch.randn(1, 16, 8, 8).to(device)

        try:
            with torch.no_grad():
                y = ss2d(x)
            print_ok(f"    SS2D 前向传播成功: {x.shape} -> {y.shape}")
        except NameError as e:
            if 'selective_scan_cuda_core' in str(e):
                print_fail(f"    SS2D 需要 selective_scan_cuda_core (未编译)")
            else:
                raise
        except Exception as e:
            print_fail(f"    SS2D 前向传播失败: {e}")

    except ImportError as e:
        print_fail(f"mamba_yolo 导入失败: {e}")

def check_causal_conv1d():
    """检查 causal_conv1d"""
    print_section("5. causal_conv1d 依赖检查")

    print_info("检查 causal_conv1d...")
    try:
        import causal_conv1d
        print_ok(f"causal_conv1d 导入成功")
        print_info(f"  版本: {getattr(causal_conv1d, '__version__', 'unknown')}")

        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        print_ok(f"  causal_conv1d_fn 可用")
        print_ok(f"  causal_conv1d_update 可用")

    except ImportError as e:
        print_fail(f"causal_conv1d 导入失败: {e}")
        print_info(f"  安装方法: pip install causal-conv1d")
        print_info(f"  或使用预编译 wheel: pip install wheels/causal_conv1d-*.whl")

def check_installed_files():
    """检查已安装的文件"""
    print_section("6. 已安装文件检查")

    try:
        import mamba_ssm
        mamba_path = Path(mamba_ssm.__file__).parent

        print_info(f"mamba_ssm 路径: {mamba_path}")

        # 查找 .so 文件
        so_files = list(mamba_path.glob("**/*.so"))
        if so_files:
            print_ok(f"找到 {len(so_files)} 个编译的 .so 文件:")
            for f in so_files:
                print_info(f"  - {f.name}")
        else:
            print_warning(f"未找到 .so 文件 (可能未编译或安装不完整)")

        # 检查 ops 目录
        ops_path = mamba_path / "ops"
        if ops_path.exists():
            print_info(f"\nops 目录内容:")
            for item in ops_path.iterdir():
                if item.is_file():
                    print_info(f"  - {item.name}")
    except ImportError:
        print_fail(f"mamba_ssm 未安装，无法检查文件")

def run_performance_test():
    """运行简单性能测试"""
    print_section("7. 性能测试 (可选)")

    try:
        import torch
    except ImportError:
        print_warning("PyTorch 未安装，跳过性能测试")
        return

    if not torch.cuda.is_available():
        print_warning("CUDA 不可用，跳过性能测试")
        return

    print_info("运行 Mamba 性能基准测试...")

    try:
        from mamba_ssm import Mamba
        import time

        device = "cuda"
        batch_size = 4
        seq_len = 512
        d_model = 64

        # 创建模型
        model = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        ).to(device)
        model.eval()

        # 预热
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)

        # 测试
        torch.cuda.synchronize()
        start = time.time()
        iterations = 100

        with torch.no_grad():
            for _ in range(iterations):
                _ = model(x)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        throughput = (batch_size * iterations) / elapsed
        latency = (elapsed / iterations) * 1000

        print_ok(f"性能测试完成:")
        print_info(f"  输入形状: ({batch_size}, {seq_len}, {d_model})")
        print_info(f"  迭代次数: {iterations}")
        print_info(f"  总耗时: {elapsed:.3f}s")
        print_info(f"  吞吐量: {throughput:.1f} samples/s")
        print_info(f"  延迟: {latency:.2f} ms")

    except Exception as e:
        print_fail(f"性能测试失败: {e}")

def print_summary(results):
    """打印测试总结"""
    print_section("测试总结")

    total = len(results)
    passed = sum(results.values())

    print(f"\n{'测试项':<40} {'结果':>10}")
    print(f"{'-'*60}")

    for name, result in results.items():
        status = f"{Colors.OKGREEN}✓ 通过{Colors.ENDC}" if result else f"{Colors.FAIL}✗ 失败{Colors.ENDC}"
        print(f"{name:<40} {status:>15}")

    print(f"{'-'*60}")
    print(f"\n总计: {passed}/{total} 项通过")

    if passed == total:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 所有检查通过！Mamba-SSM 安装正确。{Colors.ENDC}")
        return 0
    else:
        print(f"\n{Colors.WARNING}{Colors.BOLD}⚠ 有 {total - passed} 项检查失败{Colors.ENDC}")
        print(f"\n建议:")
        project_root = get_project_root()
        if not results.get("CUDA 扩展"):
            print(f"  - 使用预编译 wheel:")
            print(f"    pip install {project_root}/wheels/mamba_ssm-*.whl")
        if not results.get("causal_conv1d"):
            print(f"  - 安装 causal_conv1d:")
            print(f"    pip install {project_root}/wheels/causal_conv1d-*.whl")
        if not results.get("YOLO Mamba"):
            print(f"  - 运行补丁应用脚本:")
            print(f"    python {project_root}/scripts/apply_patches.py")
        return 1

def main():
    print_header("Mamba-SSM Jetson 安装检查")

    results = {}

    # 运行各项检查
    try:
        get_system_info()
    except Exception as e:
        print_fail(f"系统信息检查出错: {e}")

    try:
        check_cuda_extensions()
        # 简单判断是否有核心扩展
        try:
            import selective_scan_cuda_core
            results["CUDA 扩展"] = True
        except:
            results["CUDA 扩展"] = False
    except Exception as e:
        print_fail(f"CUDA 扩展检查出错: {e}")
        results["CUDA 扩展"] = False

    try:
        check_mamba_modules()
        try:
            from mamba_ssm import Mamba
            results["Mamba 模块"] = True
        except:
            results["Mamba 模块"] = False
    except Exception as e:
        print_fail(f"Mamba 模块检查出错: {e}")
        results["Mamba 模块"] = False

    try:
        check_yolo_mamba()
        try:
            project_root = get_project_root()
            yolo_dir = project_root / 'src' / 'yolo'
            sys.path.insert(0, str(yolo_dir))
            from mamba_yolo import SS2D
            import torch
            # 简单测试
            ss2d = SS2D(d_model=16)
            x = torch.randn(1, 16, 4, 4)
            try:
                y = ss2d(x)
                results["YOLO Mamba"] = True
            except NameError:
                results["YOLO Mamba"] = False
        except:
            results["YOLO Mamba"] = False
    except Exception as e:
        print_fail(f"YOLO Mamba 检查出错: {e}")
        results["YOLO Mamba"] = False

    try:
        check_causal_conv1d()
        try:
            import causal_conv1d
            results["causal_conv1d"] = True
        except:
            results["causal_conv1d"] = False
    except Exception as e:
        print_fail(f"causal_conv1d 检查出错: {e}")
        results["causal_conv1d"] = False

    try:
        check_installed_files()
    except Exception as e:
        print_fail(f"文件检查出错: {e}")

    try:
        run_performance_test()
    except Exception as e:
        print_fail(f"性能测试出错: {e}")

    # 打印总结
    return print_summary(results)

if __name__ == "__main__":
    sys.exit(main())
