#!/usr/bin/env python3
"""
模型摘要查看工具 - 支持 .pt / .onnx / .engine (TensorRT) 格式
用法: python model_info.py <model_path>
"""

import sys
import os
from pathlib import Path


def print_section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def get_pt_info(model_path):
    """查看 PyTorch (.pt) 模型信息"""
    try:
        import torch
    except ImportError:
        print("错误: 需要安装 torch")
        return

    model = torch.load(model_path, map_location='cpu')
    print_section("PyTorch 模型信息")

    # 文件大小
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"文件路径: {model_path}")
    print(f"文件大小: {size_mb:.2f} MB")

    # 模型类型判断
    if isinstance(model, dict):
        if 'model' in model:
            print(f"模型类型: YOLO 训练检查点")
            m = model['model']
            if hasattr(m, 'names'):
                print(f"类别数量: {len(m.names)}")
                print(f"类别名称: {m.names}")
            if hasattr(m, 'args'):
                print(f"模型参数: {m.args}")
        else:
            print(f"模型类型: 字典格式")
            print(f"键列表: {list(model.keys())}")
    elif hasattr(model, 'names'):
        print(f"模型类型: YOLO 模型")
        print(f"类别数量: {len(model.names)}")
        print(f"类别名称: {model.names}")
    else:
        print(f"模型类型: {type(model)}")

    # 尝试获取模型结构
    if isinstance(model, dict) and 'model' in model:
        m = model['model']
        if hasattr(m, 'model'):
            print(f"\n模型结构:")
            total_params = 0
            for i, layer in enumerate(m.model):
                if hasattr(layer, '_modules'):
                    layer_name = list(layer._modules.keys())[0] if layer._modules else f"layer_{i}"
                else:
                    layer_name = str(type(layer).__name__)
                print(f"  [{i}] {layer_name}")
                if hasattr(layer, 'n_p') and hasattr(layer, 'p'):
                    total_params += layer.n_p
            if hasattr(m, 'np'):
                print(f"\n总参数量: {m.np:,} ({m.np/1e6:.2f}M)")


def get_onnx_info(model_path):
    """查看 ONNX 模型信息"""
    try:
        import onnx
        from onnx import helper, numpy_helper
    except ImportError:
        print("错误: 需要安装 onnx")
        return

    model = onnx.load(model_path)
    print_section("ONNX 模型信息")

    # 文件大小
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"文件路径: {model_path}")
    print(f"文件大小: {size_mb:.2f} MB")

    print(f"ONNX 版本: {model.opset_import[0].version if model.opset_import else 'N/A'}")
    print(f"生产者: {model.producer_name} {model.producer_version}")

    # 输入信息
    print(f"\n输入 ({len(model.graph.input)} 个):")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else '?' for d in inp.type.tensor_type.shape.dim]
        print(f"  - {inp.name}: {inp.type.tensor_type.elem_type} {shape}")

    # 输出信息
    print(f"\n输出 ({len(model.graph.output)} 个):")
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else '?' for d in out.type.tensor_type.shape.dim]
        print(f"  - {out.name}: {out.type.tensor_type.elem_type} {shape}")

    # 节点统计
    print(f"\n网络节点: {len(model.graph.node)} 个")

    # 算子统计
    op_count = {}
    for node in model.graph.node:
        op_count[node.op_type] = op_count.get(node.op_type, 0) + 1
    print(f"算子统计:")
    for op, count in sorted(op_count.items(), key=lambda x: -x[1]):
        print(f"  - {op}: {count}")


def get_engine_info(model_path):
    """查看 TensorRT Engine 模型信息 (通过 YOLO 加载)"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 需要安装 ultralytics")
        return

    print_section("TensorRT Engine 信息 (YOLO)")

    # 文件大小
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"文件路径: {model_path}")
    print(f"文件大小: {size_mb:.2f} MB")

    try:
        model = YOLO(model_path)

        # 模型基本信息
        print(f"任务类型: {model.task if hasattr(model, 'task') else 'N/A'}")

        # 类别信息
        if hasattr(model, 'names') and model.names:
            print(f"类别数量: {len(model.names)}")
            print(f"类别名称: {model.names}")

        # 模型参数
        if hasattr(model, 'args') and model.args:
            print(f"\n模型配置:")
            for key, val in model.args.items():
                if key in ['imgsz', 'half', 'dynamic', 'simplify', 'workspace']:
                    print(f"  {key}: {val}")

        # 尝试获取模型的输入输出信息
        if hasattr(model, 'model') and hasattr(model.model, 'inputs'):
            print(f"\n输入信息: {model.model.inputs}")
        if hasattr(model, 'model') and hasattr(model.model, 'outputs'):
            print(f"输出信息: {model.model.outputs}")

        # Triton 模型信息
        if hasattr(model, 'triton_model'):
            print(f"\nTriton 模型: {model.triton_model}")

        print(f"\n提示: 使用 model.predict() 进行推理测试")

    except Exception as e:
        print(f"加载模型失败: {e}")
        print(f"\n可能原因:")
        print(f"  - Engine 文件与当前 TensorRT 版本不兼容")
        print(f"  - 需要在相同环境下重新导出 Engine")
        print(f"  - 建议从 ONNX 重新构建: YOLO('model.onnx').export(format='engine')")


def main():
    if len(sys.argv) < 2:
        print("用法: python model_info.py <model_path>")
        print("\n支持的格式: .pt, .onnx, .engine")
        sys.exit(1)

    model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"错误: 文件不存在: {model_path}")
        sys.exit(1)

    suffix = Path(model_path).suffix.lower()

    if suffix == '.pt':
        get_pt_info(model_path)
    elif suffix == '.onnx':
        get_onnx_info(model_path)
    elif suffix in ['.engine', '.plan', '.trt']:
        get_engine_info(model_path)
    else:
        print(f"不支持的格式: {suffix}")
        print("支持的格式: .pt, .onnx, .engine, .plan, .trt")
        sys.exit(1)


if __name__ == '__main__':
    main()
