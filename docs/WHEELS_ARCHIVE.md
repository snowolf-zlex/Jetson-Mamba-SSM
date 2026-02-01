# 预编译 Wheel 包存档

**编译日期**: 2026-02-01
**设备**: NVIDIA Jetson Orin (ARM64)
**Python**: 3.10
**CUDA**: 12.x

---

## 版本依赖

| 包 | 版本 | 源码仓库 |
|------|------|----------|
| **mamba-ssm** | 2.2.4 | [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| **causal-conv1d** | 1.6.0 | [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) |

这些 wheel 文件是从上述版本的源码在 Jetson Orin (ARM64) 上编译而来。

---

## Wheel 文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `causal_conv1d-1.6.0-cp310-cp310-linux_aarch64.whl` | 185 MB | Causal Conv1D 扩展 |
| `mamba_ssm-2.2.4-cp310-cp310-linux_aarch64.whl` | 310 MB | Mamba-SSM 主包 |

**位置**: `/home/jetson/jetson-mamba-ssm/wheels/`

---

## 快速安装

### 方法 1: 使用安装脚本 (推荐)

```bash
cd /home/jetson/jetson-mamba-ssm
python scripts/install_wheels.py
```

### 方法 2: 手动安装

```bash
# 1. 设置环境
export CUDA_HOME=/usr/local/cuda-12.6  # 根据实际 CUDA 版本调整

# 2. 安装 wheel 文件
pip install wheels/causal_conv1d-1.6.0-cp310-cp310-linux_aarch64.whl
pip install wheels/mamba_ssm-2.2.4-cp310-cp310-linux_aarch64.whl

# 3. 应用补丁
python scripts/apply_patches.py

# 4. 验证安装
python scripts/verify.py
```

---

## 编译记录

| 模块 | 版本 | 编译时间 | 状态 |
|------|------|----------|------|
| causal_conv1d | 1.6.0 | ~20-40 分钟 | ✓ |
| mamba-ssm | 2.2.4 | ~1-2 分钟 | ✓ |

---

## 兼容性

- **设备**: NVIDIA Jetson (Orin/Xavier/Nano)
- **架构**: ARM64 (aarch64)
- **Python**: 3.10
- **CUDA**: 12.x
- **PyTorch**: 2.x (JetPack 版本)

---

## 修复清单

### 编译时修复
- ✓ CUDA_HOME 环境变量设置
- ✓ 非 editable 模式安装
- ✓ torch.distributed API 兼容性 (sitecustomize.py)

### 运行时修复
- ✓ libc10.so 依赖问题 (使用 causal_conv1d_fn)
- ✓ selective_scan_cuda_core wrapper (mamba_yolo.py)
- ✓ SS2D 空间维度裁剪 (mamba_yolo.py)

---

## 参考文档

- [MAMBA_SSM_JETSON_FIX.md](MAMBA_SSM_JETSON_FIX.md) - 修复说明
- [JETSON_MAMBA_SSM_BUILD_GUIDE.md](JETSON_MAMBA_SSM_BUILD_GUIDE.md) - 完整编译指南

---

**存档版本**: 1.0
**最后更新**: 2026-02-02
