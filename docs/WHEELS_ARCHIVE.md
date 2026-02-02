# 预编译 Wheel 包存档

**编译日期**: 2026-02-01
**设备**: NVIDIA Jetson Orin (ARM64)
**Python**: 3.10
**CUDA**: 12.x

---

## 编译环境与配置

### 硬件环境
- **设备**: NVIDIA Jetson Orin
- **架构**: ARM64 (aarch64)
- **GPU**: Orin (Ampere 架构)
- **内存**: 64GB (建议 32GB+)

### 软件环境
- **操作系统**: Linux 5.15.148-tegra (JetPack 5.x/6.x)
- **Python**: 3.10
- **CUDA**: 12.6
- **cuDNN**: 8.x
- **PyTorch**: 2.x (JetPack 版本)
- **编译器**: GCC 11.4.0 / NVCC 12.6

### 编译模式
| 模块 | 编译模式 | 说明 |
|------|----------|------|
| causal_conv1d | Release + CUDA | 启用 CUDA 优化，Release 模式 |
| mamba-ssm | Release + CUDA | 启用 CUDA 优化，Release 模式 |

**重要**:
- 使用 `pip install .` (非 editable 模式)
- 设置 `CUDA_HOME=/usr/local/cuda-12.6`
- 编译前安装依赖: `pip install wheel packaging ninja triton`

**前置依赖**:
```bash
# mamba-ssm 核心依赖
pip install einops ninja packaging transformers

# triton - GPU 算子库 (必需)
# 版本要求: triton>=2.1.0 (源码标注 2.1.0 或 2.2.0)
pip install triton
```

### 编译时间参考
| 模块 | 版本 | 编译时间 | 说明 |
|------|------|----------|------|
| causal_conv1d | 1.6.0 | ~2 小时 | CUDA C++ 扩展编译，含调试时间 |
| mamba-ssm | 2.2.4 | ~1 小时 | Python 绑定 + 调试问题 |

**实际编译记录** (2026-02-01):
- **21:30** 开始编译 causal_conv1d
- **23:30** causal_conv1d 编译完成 (~2 小时)
- **00:30** mamba-ssm 编译完成 (~1 小时)
- **总耗时**: 约 3 小时 (含调试各种 bug 时间)

> **注意**: 编译时间受设备性能、CUDA 版本、系统负载等因素影响。首次编译需要解决大量依赖和配置问题，耗时较长。

---

## 版本依赖

| 包 | 版本 | 源码仓库 |
|------|------|----------|
| **mamba-ssm** | 2.2.4 | [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| **causal-conv1d** | 1.6.0 | [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) |
| **triton** | >=2.1.0 | [openai/triton](https://github.com/openai/triton) |

这些 wheel 文件是从上述版本的源码在 Jetson Orin (ARM64) 上编译而来。

**triton 版本说明**:
- mamba-ssm 源码标注要求 `triton==2.1.0` 或 `2.2.0`
- 实际上 `triton>=2.1.0` 均可正常工作
- Jetson Orin 实测: `triton 3.5.1` ✅ 兼容

---

## Wheel 文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `causal_conv1d-1.6.0-cp310-cp310-linux_aarch64.whl` | 185 MB | Causal Conv1D 扩展 |
| `mamba_ssm-2.2.4-cp310-cp310-linux_aarch64.whl` | 310 MB | Mamba-SSM 主包 |

**位置**: `wheels/`

---

## 快速安装

### 前置依赖

```bash
# 基础依赖
pip install einops ninja packaging transformers

# triton - mamba-ssm 核心 GPU 算子库
pip install triton
```

### 方法 1: 使用安装脚本 (推荐)

```bash
cd /path/to/Jetson-Mamba-SSM
python scripts/install_wheels.py
```

### 方法 2: 手动安装

```bash
# 1. 设置环境
export CUDA_HOME=/usr/local/cuda-12.6  # 根据实际 CUDA 版本调整

# 2. 安装前置依赖
pip install einops ninja packaging transformers triton

# 3. 安装 wheel 文件
pip install wheels/causal_conv1d-1.6.0-cp310-cp310-linux_aarch64.whl
pip install wheels/mamba_ssm-2.2.4-cp310-cp310-linux_aarch64.whl

# 4. 应用补丁
python scripts/apply_patches.py

# 5. 验证安装
python scripts/verify.py
```

---

## 兼容性

| 组件 | 版本要求 |
|------|----------|
| 设备 | NVIDIA Jetson (Orin/Xavier/Nano) |
| 架构 | ARM64 (aarch64) |
| Python | 3.10 |
| CUDA | 12.x |
| PyTorch | 2.x (JetPack 版本) |

---

## 修复清单

### 编译时修复
- ✓ CUDA_HOME 环境变量设置
- ✓ 非 editable 模式安装 (`pip install .`)
- ✓ torch.distributed API 兼容性 (sitecustomize.py)

### 运行时修复
- ✓ libc10.so 依赖问题 (使用 causal_conv1d_fn)
- ✓ selective_scan_cuda_core wrapper (mamba_yolo.py)
- ✓ SS2D 空间维度裁剪 (mamba_yolo.py)

---

## 从源码编译

如果预编译 wheel 不适用，可按以下步骤从源码编译。

**重要**: 必须先打补丁修复 mamba 源码，否则编译会失败！

```bash
# 1. 设置环境
export CUDA_HOME=/usr/local/cuda-12.6

# 2. 安装前置依赖
pip install wheel packaging ninja einops transformers triton

# 3. 克隆源码
git clone https://github.com/Dao-AILab/causal-conv1d.git
git clone https://github.com/state-spaces/mamba.git

# 4. 应用 Jetson 补丁到 mamba 源码（必须先修复！）
cd mamba
git checkout v2.2.4
patch -p1 < /path/to/Jetson-Mamba-SSM/patches/00_selective_scan_interface.py.patch
patch -p1 < /path/to/Jetson-Mamba-SSM/patches/01_ssd_combined.py.patch

# 5. 编译并安装 causal_conv1d (约 2 小时)
cd ../causal-conv1d
git checkout v1.6.0
pip install .  # 注意: 不要使用 -e 选项

# 6. 编译并安装 mamba-ssm (约 1 小时)
cd ../mamba
pip install .  # 注意: 不要使用 -e 选项

# 7. 应用运行时补丁
cd /path/to/Jetson-Mamba-SSM
python scripts/apply_patches.py
```

**为什么必须先打补丁？**

原始 mamba-ssm 源码使用 `causal_conv1d_cuda.causal_conv1d_fwd`，这在 Jetson 上会因为 `libc10.so` 依赖问题导致编译失败。补丁将代码改为使用 `causal_conv1d_fn`，这是编译成功的关键。

---

## 参考文档

- [MAMBA_SSM_JETSON_FIX.md](MAMBA_SSM_JETSON_FIX.md) - 修复说明
- [JETSON_MAMBA_SSM_BUILD_GUIDE.md](JETSON_MAMBA_SSM_BUILD_GUIDE.md) - 完整编译指南

---

**存档版本**: 1.0
**最后更新**: 2026-02-02
