# Jetson Mamba-SSM

![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin-32B3E6?logo=nvidia)
![Architecture](https://img.shields.io/badge/Architecture-ARM64-E96479?logo=arm)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-green)

在 NVIDIA Jetson (ARM64) 上编译和运行 Mamba-SSM 的补丁和工具。

## 概述

Mamba-SSM 官方仅提供 x86_64 架构的预编译包，本项目提供了在 Jetson (ARM64) 上编译和运行 Mamba-SSM 所需的补丁和修改后的源文件。

## 问题

1. **libc10.so 依赖**: `causal_conv1d_cuda` 模块依赖 Jetson 上不存在的 `libc10.so`
2. **torch.distributed API 缺失**: JetPack PyTorch 缺少某些分布式 API
3. **selective_scan_cuda_core 未找到**: YOLO 集成时的兼容性问题

## 解决方案

- 使用 `causal_conv1d_fn` 替代 `causal_conv1d_cuda.causal_conv1d_fwd`
- 添加 `sitecustomize.py` 修复缺失的分布式 API
- 提供 selective_scan wrapper 用于 YOLO 集成

## 快速开始

### 方法 1: 使用预编译 wheel (推荐 - 最简单)

```bash
cd /home/jetson/jetson-mamba-ssm

# 一键安装 (包含 wheel 安装 + 补丁应用)
python scripts/install_wheels.py
```

或手动安装:

```bash
# 设置环境
export CUDA_HOME=/usr/local/cuda-12.6

# 安装预编译的 wheel 文件
pip install wheels/causal_conv1d-1.6.0-cp310-cp310-linux_aarch64.whl
pip install wheels/mamba_ssm-2.2.4-cp310-cp310-linux_aarch64.whl

# 应用运行时补丁
python scripts/apply_patches.py
```

### 方法 2: 从 PyPI 安装 + 补丁

```bash
# 1. 编译 causal_conv1d
cd causal_conv1d
pip install .

# 2. 编译 mamba-ssm
cd mamba-ssm
pip install .

# 3. 应用补丁
cd jetson-mamba-ssm
python scripts/apply_patches.py
```

### 方法 3: 使用本项目的修改后源文件

```bash
# 直接复制修改后的文件到 site-packages
python scripts/install.py
```

## 验证安装

```bash
python scripts/verify.py
```

预期输出:
```
✓ causal_conv1d_fn
✓ Mamba module
✓ selective_scan_cuda
✓ SS2D
✓ VSSBlock_YOLO
```

## 项目结构

```
jetson-mamba-ssm/
├── README.md              # 本文件
├── LICENSE                # MIT 许可证
├── wheels/                # 预编译 wheel 文件 (Jetson ARM64)
│   ├── causal_conv1d-1.6.0-cp310-cp310-linux_aarch64.whl
│   └── mamba_ssm-2.2.4-cp310-cp310-linux_aarch64.whl
├── patches/               # Git 格式补丁
│   ├── 00_selective_scan_interface.py.patch
│   └── 01_ssd_combined.py.patch
├── src/                   # 修改后的源文件
│   ├── fix_causal_conv1d.py      # causal_conv1d_cuda 兼容层
│   ├── sitecustomize/            # 分布式 API 修复
│   ├── selective_scan_cuda.py    # selective_scan shim
│   ├── mamba_ssm/                # mamba-ssm 修改文件
│   │   ├── ops/
│   │   │   ├── selective_scan_interface.py
│   │   │   └── triton/ssd_combined.py
│   │   └── distributed/distributed_utils.py
│   └── yolo/
│       └── mamba_yolo.py         # YOLO Mamba 集成
├── scripts/               # 安装和验证脚本
│   ├── install_wheels.py         # 安装预编译 wheel
│   ├── apply_patches.py          # 应用运行时补丁
│   ├── verify.py                 # 验证安装
│   ├── check_mamba_install.py    # 全面检查脚本
│   └── run_with_mamba.sh         # 运行环境封装脚本
└── docs/                  # 文档
    ├── WHEELS_ARCHIVE.md         # 预编译 wheel 说明
    ├── JETSON_MAMBA_SSM_BUILD_GUIDE.md  # 编译指南
    └── MAMBA_SSM_JETSON_FIX.md   # 修复说明
```

## 修改说明

### 1. libc10.so 依赖修复

**文件**: `src/mamba_ssm/ops/selective_scan_interface.py`, `src/mamba_ssm/ops/triton/ssd_combined.py`

**修改**: 使用 `causal_conv1d_fn` 替代 `causal_conv1d_cuda.causal_conv1d_fwd`

```python
# Before
conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, weight, bias, None, None, None, True)

# After
if causal_conv1d_fn is not None:
    conv1d_out = causal_conv1d_fn(x, weight, bias, seq_idx=None, initial_states=None, final_states_out=None, activation="silu")
```

### 2. torch.distributed API 修复

**文件**: `src/sitecustomize.py`

添加缺失的分布式 API 存根。

### 3. selective_scan_cuda_core wrapper

**文件**: `src/yolo/mamba_yolo.py`

为 YOLO 集成提供 `selective_scan_cuda_core` wrapper。

## 兼容性

| 组件 | 版本 |
|------|------|
| 设备 | NVIDIA Jetson (Orin/Xavier/Nano) |
| 架构 | ARM64 (aarch64) |
| Python | 3.10 |
| CUDA | 12.x |
| PyTorch | 2.x (JetPack 版本) |
| mamba-ssm | 2.2.4 |
| causal-conv1d | 1.6.0 |

## 已知问题

1. YOLO 模块 (SS2D, VSSBlock_YOLO) 需要 `float32` dtype
2. 反向传播可能不支持 (causal_conv1d_bwd 未实现)

## 参考文档

- [BUILD_GUIDE.md](docs/BUILD_GUIDE.md) - 完整编译指南
- [FIX_EXPLANATION.md](docs/FIX_EXPLANATION.md) - 详细修复说明

## 许可证

MIT License

## 致谢

- [Mamba-SSM](https://github.com/state-spaces/mamba) - Tri Dao, Albert Gu
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) - Tri Dao
