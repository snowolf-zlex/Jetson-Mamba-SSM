#!/usr/bin/env python3
"""
YOLOv10 + Mamba 模型精度导出测试

测试不同精度 (FP32, FP16, INT8) 的导出支持。

依赖:
- ultralytics (YOLOv10)
- mamba-ssm (已打补丁)
- TensorRT

作者: jetson-mamba-ssm 项目
日期: 2026-02-02
"""

import os
import sys
import time
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


def get_file_size_mb(path):
    """获取文件大小 (MB)"""
    return os.path.getsize(path) / (1024 * 1024)


def test_export_precision(model_path, precision, imgsz=640, device=0):
    """
    测试指定精度的导出

    Args:
        model_path: 模型路径 (.pt)
        precision: 精度类型 ('fp32', 'fp16', 'int8')
        imgsz: 输入尺寸
        device: 设备

    Returns:
        dict: 包含成功状态、耗时、文件大小等信息
    """
    print_header(f"测试 {precision.upper()} 导出")

    result = {
        'precision': precision,
        'success': False,
        'time': 0,
        'size_mb': 0,
        'output_path': None,
        'error': None
    }

    try:
        from ultralytics import YOLO
        from mamba_ssm.ops.selective_scan_interface import _set_onnx_export_mode

        # 启用 ONNX 导出模式
        _set_onnx_export_mode(True)

        # 加载模型
        print_info(f"加载模型: {model_path}")
        model = YOLO(model_path)

        # 确定输出文件名
        base_name = Path(model_path).stem
        output_dir = Path(model_path).parent
        engine_path = output_dir / f"{base_name}_{precision}.engine"

        # 根据精度设置导出参数
        export_kwargs = {
            'format': 'engine',
            'imgsz': imgsz,
            'device': device,
            'verbose': False,
        }

        # FP32 是默认精度，half=False
        if precision == 'fp32':
            export_kwargs['half'] = False
        # FP16 启用 half
        elif precision == 'fp16':
            export_kwargs['half'] = True
        # INT8 需要启用 int8
        elif precision == 'int8':
            export_kwargs['int8'] = True
            export_kwargs['half'] = False  # INT8 不需要 half

        print_info(f"导出参数: {export_kwargs}")

        # 开始计时
        start_time = time.time()

        # 导出模型
        model.export(**export_kwargs)

        # 计算耗时
        elapsed = time.time() - start_time
        result['time'] = elapsed

        # 检查输出文件
        if engine_path.exists():
            result['success'] = True
            result['size_mb'] = get_file_size_mb(engine_path)
            result['output_path'] = str(engine_path)
            print_success(f"导出成功: {engine_path.name}")
            print_info(f"文件大小: {result['size_mb']:.2f} MB")
            print_info(f"导出耗时: {elapsed:.1f} 秒")
        else:
            result['error'] = "输出文件未生成"
            print_error(f"导出失败: 输出文件未生成")

        # 关闭 ONNX 导出模式
        _set_onnx_export_mode(False)

    except Exception as e:
        result['error'] = str(e)
        print_error(f"导出失败: {e}")

    return result


def test_onnx_precision(model_path, precision, imgsz=640):
    """
    测试 ONNX 格式的不同精度导出

    Args:
        model_path: 模型路径 (.pt)
        precision: 精度类型 ('fp32', 'fp16')

    Returns:
        dict: 包含成功状态、耗时、文件大小等信息
    """
    print_header(f"测试 ONNX {precision.upper()} 导出")

    result = {
        'precision': f"onnx_{precision}",
        'success': False,
        'time': 0,
        'size_mb': 0,
        'output_path': None,
        'error': None
    }

    try:
        from ultralytics import YOLO
        from mamba_ssm.ops.selective_scan_interface import _set_onnx_export_mode

        # 启用 ONNX 导出模式
        _set_onnx_export_mode(True)

        # 加载模型
        print_info(f"加载模型: {model_path}")
        model = YOLO(model_path)

        # 确定输出文件名
        base_name = Path(model_path).stem
        output_dir = Path(model_path).parent
        onnx_path = output_dir / f"{base_name}_{precision}.onnx"

        # 根据精度设置导出参数
        export_kwargs = {
            'format': 'onnx',
            'imgsz': imgsz,
            'verbose': False,
        }

        # FP16 需要启用 half
        if precision == 'fp16':
            export_kwargs['half'] = True
        else:
            export_kwargs['half'] = False

        print_info(f"导出参数: {export_kwargs}")

        # 开始计时
        start_time = time.time()

        # 导出模型
        model.export(**export_kwargs)

        # 计算耗时
        elapsed = time.time() - start_time
        result['time'] = elapsed

        # 检查输出文件 (YOLO 可能会添加 .onnx 后缀)
        possible_paths = [
            onnx_path,
            output_dir / f"{base_name}.onnx",  # YOLO 可能使用默认名称
        ]

        output_file = None
        for path in possible_paths:
            if path.exists():
                output_file = path
                break

        if output_file:
            result['success'] = True
            result['size_mb'] = get_file_size_mb(output_file)
            result['output_path'] = str(output_file)
            print_success(f"导出成功: {output_file.name}")
            print_info(f"文件大小: {result['size_mb']:.2f} MB")
            print_info(f"导出耗时: {elapsed:.1f} 秒")
        else:
            result['error'] = "输出文件未生成"
            print_error(f"导出失败: 输出文件未生成")

        # 关闭 ONNX 导出模式
        _set_onnx_export_mode(False)

    except Exception as e:
        result['error'] = str(e)
        print_error(f"导出失败: {e}")

    return result


def main():
    """主函数"""
    print_header("YOLOv10 + Mamba 精度导出测试")

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
        print_error("未找到测试模型")
        print_info("请指定模型路径:")
        print("  python scripts/test_export_precision.py --model /path/to/model.pt")
        return 1

    print_info(f"使用模型: {model_path}")

    # 解析命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == '--model' and len(sys.argv) > 2:
            model_path = sys.argv[2]

    # 测试配置
    imgsz = 640
    device = 0

    # 测试 ONNX 精度
    onnx_precisions = ['fp32', 'fp16']
    onnx_results = []

    for precision in onnx_precisions:
        result = test_onnx_precision(model_path, precision, imgsz)
        onnx_results.append(result)
        print()  # 空行

    # 测试 TensorRT 精度
    trt_precisions = ['fp32', 'fp16', 'int8']
    trt_results = []

    for precision in trt_precisions:
        result = test_export_precision(model_path, precision, imgsz, device)
        trt_results.append(result)
        print()  # 空行

    # 打印测试结果摘要
    print_header("测试结果摘要")

    print(f"\n{'='*60}")
    print(f"  ONNX 导出结果")
    print(f"{'='*60}")

    for r in onnx_results:
        status = f"{GREEN}✓{RESET}" if r['success'] else f"{RED}✗{RESET}"
        precision = r['precision'].replace('onnx_', '').upper()
        print(f"{status} {precision:6s} ", end="")
        if r['success']:
            print(f"{r['size_mb']:6.2f} MB, {r['time']:5.1f}s")
        else:
            print(f"失败: {r['error']}")

    print(f"\n{'='*60}")
    print(f"  TensorRT 导出结果")
    print(f"{'='*60}")

    for r in trt_results:
        status = f"{GREEN}✓{RESET}" if r['success'] else f"{RED}✗{RESET}"
        precision = r['precision'].upper()
        print(f"{status} {precision:6s} ", end="")
        if r['success']:
            print(f"{r['size_mb']:6.2f} MB, {r['time']:5.1f}s")
        else:
            print(f"失败: {r['error']}")

    # 统计
    total = len(onnx_results) + len(trt_results)
    passed = sum(1 for r in onnx_results + trt_results if r['success'])

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print_success("\n✓ 所有精度测试通过！")
        return 0
    else:
        print_warning(f"\n! {total - passed} 个精度测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
