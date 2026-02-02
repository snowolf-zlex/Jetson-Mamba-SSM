#!/usr/bin/env python3
"""
验证 Mamba-SSM 在 Jetson 上的安装
"""
import sys
import os


def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def test_causal_conv1d():
    """测试 causal_conv1d"""
    print("\n1. 测试 causal_conv1d_fn...")
    try:
        from causal_conv1d import causal_conv1d_fn
        import torch
        x = torch.randn(2, 32, 64, device='cuda')
        weight = torch.randn(32, 4, device='cuda')
        with torch.no_grad():
            y = causal_conv1d_fn(x, weight, None, None, None, None, 'silu')
        print(f"   ✓ PASS: {x.shape} -> {y.shape}")
        return True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        return False


def test_mamba_module():
    """测试 Mamba 模块"""
    print("\n2. 测试 Mamba 模块...")
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
        import torch
        mamba = Mamba(d_model=64, d_state=8).cuda().half()
        x = torch.randn(2, 32, 64, device='cuda', dtype=torch.float16)
        with torch.no_grad():
            y = mamba(x)
        print(f"   ✓ PASS: {x.shape} -> {y.shape}")
        return True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        return False


def test_selective_scan():
    """测试 selective_scan_cuda"""
    print("\n3. 测试 selective_scan_cuda...")
    try:
        import selective_scan_cuda
        print(f"   ✓ PASS: {selective_scan_cuda.__file__}")
        return True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        return False


def test_yolo_mamba():
    """测试 YOLO Mamba 模块"""
    print("\n4. 测试 SS2D (YOLO)...")
    try:
        sys.path.insert(0, './yolov10_main')
        sys.path.insert(0, './yolov10_main/ultralytics')
        import yolov10_main.ultralytics.nn.AddModules.Structure.mamba_yolo as mamba_yolo
        import torch
        ss2d = mamba_yolo.SS2D(d_model=64, d_state=8).cuda().float()
        x = torch.randn(2, 64, 32, 32, device='cuda', dtype=torch.float32)
        with torch.no_grad():
            y = ss2d(x)
        print(f"   ✓ PASS: {x.shape} -> {y.shape}")
        return True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        return False


def test_vss_block():
    """测试 VSSBlock_YOLO"""
    print("\n5. 测试 VSSBlock_YOLO...")
    try:
        sys.path.insert(0, './yolov10_main')
        import yolov10_main.ultralytics.nn.AddModules.Structure.mamba_yolo as mamba_yolo
        import torch
        vss = mamba_yolo.VSSBlock_YOLO(in_channels=64, hidden_dim=64, ssm_d_state=8).cuda().float()
        x = torch.randn(2, 64, 32, 32, device='cuda', dtype=torch.float32)
        with torch.no_grad():
            y = vss(x)
        print(f"   ✓ PASS: {x.shape} -> {y.shape}")
        return True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        return False


def main():
    print_header("Mamba-SSM Jetson 验证测试")

    # 检查 CUDA
    try:
        import torch
        print(f"\nCUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"CUDA 检查失败: {e}")
        return

    results = []
    results.append(("causal_conv1d_fn", test_causal_conv1d()))
    results.append(("Mamba 模块", test_mamba_module()))
    results.append(("selective_scan_cuda", test_selective_scan()))
    results.append(("SS2D", test_yolo_mamba()))
    results.append(("VSSBlock_YOLO", test_vss_block()))

    print_header("测试结果")

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有测试通过! Mamba-SSM 已正确安装。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查安装。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
