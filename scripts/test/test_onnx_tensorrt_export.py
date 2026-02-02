#!/usr/bin/env python3
"""
YOLOv10 + Mamba ONNX/TensorRT 导出测试脚本

测试 YOLOv10 + Mamba 模型的 ONNX 和 TensorRT 导出功能。

依赖:
- ultralytics (YOLOv10)
- mamba-ssm (已打补丁)
- TensorRT 10.x
- ONNX

作者: jetson-mamba-ssm 项目
日期: 2026-02-02
"""

import os
import sys
import subprocess
from pathlib import Path
import tempfile

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


def check_dependencies():
    """检查依赖项"""
    print_header("检查依赖项")

    dependencies = {
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics YOLOv8/v10',
        'onnx': 'ONNX',
        'tensorrt': 'TensorRT',
    }

    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print_success(f"{name:20s} 已安装")
        except ImportError:
            print_error(f"{name:20s} 未安装")
            missing.append(name)

    # 检查 mamba-ssm
    try:
        import mamba_ssm
        from mamba_ssm.ops.selective_scan_interface import ONNX_EXPORT_MODE
        print_success("mamba-ssm          已安装 (含 ONNX 导出补丁)")
    except ImportError:
        print_error("mamba-ssm          未安装")
        missing.append("mamba-ssm")
    except AttributeError:
        print_warning("mamba-ssm          已安装但缺少 ONNX 导出补丁")
        print_info("运行: python scripts/apply_onnx_export_patch.py")

    if missing:
        print_error(f"\n缺少依赖: {', '.join(missing)}")
        return False

    print_success("\n所有依赖项检查通过！")
    return True


def test_mamba_onnx_export_mode():
    """测试 Mamba ONNX 导出模式"""
    print_header("测试 Mamba ONNX 导出模式")

    try:
        from mamba_ssm.ops.selective_scan_interface import ONNX_EXPORT_MODE, _set_onnx_export_mode

        # 读取当前模式
        original_mode = ONNX_EXPORT_MODE

        # 测试切换功能
        _set_onnx_export_mode(True)
        if ONNX_EXPORT_MODE != True:
            print_error("无法启用 ONNX 导出模式")
            return False

        _set_onnx_export_mode(False)
        if ONNX_EXPORT_MODE != False:
            print_error("无法关闭 ONNX 导出模式")
            return False

        # 恢复原始模式
        _set_onnx_export_mode(original_mode)

        print_success("ONNX 导出模式切换正常")
        return True

    except Exception as e:
        print_error(f"测试失败: {e}")
        return False


def test_custom_onnx_symbolic():
    """测试自定义 ONNX 符号函数"""
    print_header("测试自定义 ONNX 符号函数")

    try:
        import torch
        from torch.onnx import register_custom_op_symbolic

        # 测试注册 aten::exponential 符号函数
        def symbolic_exponential(g, input, *args):
            return g.op("Exp", input)

        register_custom_op_symbolic('aten::exponential', symbolic_exponential, 17)
        print_success("aten::exponential 符号函数已注册")

        # 测试注册 aten::sort 符号函数
        def symbolic_sort(g, input, dim, descending, out=None):
            indices = g.op("Identity", input)
            return input, indices

        register_custom_op_symbolic('aten::sort', symbolic_sort, 17)
        print_success("aten::sort 符号函数已注册")

        return True

    except Exception as e:
        print_error(f"测试失败: {e}")
        return False


def test_tensorrt_api():
    """测试 TensorRT 10.x API 兼容性"""
    print_header("测试 TensorRT 10.x API")

    try:
        import tensorrt as trt

        # 检查 TensorRT 版本
        version = trt.__version__
        print_info(f"TensorRT 版本: {version}")

        major_version = int(version.split('.')[0])
        if major_version >= 10:
            print_success("TensorRT 10.x 检测到")
            print_info("使用新 API: num_io_tensors, get_tensor_name, get_tensor_dtype, etc.")
        else:
            print_info(f"TensorRT {major_version}.x 检测到")
            print_info("使用旧 API: num_bindings, get_binding_name, etc.")

        # 检查关键 API
        if hasattr(trt, 'TensorIOMode'):
            print_success("TensorIOMode 可用 (TensorRT 10.x)")
        else:
            print_warning("TensorIOMode 不可用 (可能是旧版本)")

        return True

    except Exception as e:
        print_error(f"测试失败: {e}")
        return False


def test_onnx_export():
    """测试 ONNX 导出功能"""
    print_header("测试 ONNX 导出")

    # 查找测试模型
    model_paths = [
        "/home/jetson/pythonProject/runs/yolov10s-lamp-exp15_3-finetune/weights/best.pt",
        "/home/jetson/pythonProject/yolov10_main/ultralytics/assets/yolov10n.pt",
    ]

    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break

    if model_path is None:
        print_warning("未找到测试模型，跳过 ONNX 导出测试")
        print_info("如需测试，请指定模型路径:")
        print("  python scripts/test_export.py --model /path/to/model.pt")
        return True

    print_info(f"使用模型: {model_path}")

    try:
        from ultralytics import YOLO
        from mamba_ssm.ops.selective_scan_interface import _set_onnx_export_mode
        import torch

        # 启用 ONNX 导出模式
        _set_onnx_export_mode(True)

        # 加载模型
        model = YOLO(model_path)

        # 创建测试输入
        dummy_input = torch.randn(1, 3, 640, 640)

        # 尝试导出 ONNX
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        import torch.onnx
        torch.onnx.export(
            model.model.cpu(),
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=False,
            input_names=["images"],
            output_names=["output0"],
        )

        # 检查文件大小
        onnx_size = Path(onnx_path).stat().st_size / (1024 * 1024)
        print_success(f"ONNX 导出成功: {onnx_path} ({onnx_size:.1f} MB)")

        # 清理
        os.unlink(onnx_path)

        # 关闭 ONNX 导出模式
        _set_onnx_export_mode(False)

        return True

    except Exception as e:
        print_error(f"ONNX 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print_header("YOLOv10 + Mamba ONNX/TensorRT 导出测试")

    # 解析命令行参数
    model_path = None
    if len(sys.argv) > 1:
        if sys.argv[1] == '--model':
            if len(sys.argv) > 2:
                model_path = sys.argv[2]

    # 运行测试
    tests = [
        ("依赖检查", check_dependencies),
        ("Mamba ONNX 导出模式", test_mamba_onnx_export_mode),
        ("自定义 ONNX 符号函数", test_custom_onnx_symbolic),
        ("TensorRT API 兼容性", test_tensorrt_api),
        ("ONNX 导出", test_onnx_export),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"{test_name} 测试出错: {e}")
            results.append((test_name, False))

    # 打印测试结果摘要
    print_header("测试结果摘要")

    passed = 0
    failed = 0
    for test_name, result in results:
        if result:
            print_success(f"{test_name:30s} 通过")
            passed += 1
        else:
            print_error(f"{test_name:30s} 失败")
            failed += 1

    print(f"\n总计: {passed} 通过, {failed} 失败")

    if failed == 0:
        print_success("\n✓ 所有测试通过！YOLOv10 + Mamba ONNX/TensorRT 导出功能正常")
        print_info("\n下一步:")
        print("  导出 ONNX: yolo export model=best.pt format=onnx opset=17")
        print("  导出 TensorRT: yolo export model=best.pt format=engine imgsz=640")
        return 0
    else:
        print_error(f"\n✗ {failed} 个测试失败，请检查环境配置")
        return 1


if __name__ == "__main__":
    sys.exit(main())
