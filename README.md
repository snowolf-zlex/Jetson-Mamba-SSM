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

### 方法 1: 使用预编译 wheel (推荐)

```bash
# 设置环境
export CUDA_HOME=/usr/local/cuda-12.6

# 安装
pip install causal_conv1d
pip install mamba-ssm

# 应用补丁
python scripts/apply_patches.py
```

### 方法 2: 从源码编译

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
├── patches/               # Git 格式补丁
│   ├── 0001-fix-libc10-so-dependency.patch
│   ├── 0002-fix-torch-distributed-api.patch
│   └── 0003-fix-selective-scan-cuda-core.patch
├── src/                   # 修改后的源文件
│   ├── sitecustomize.py   # 分布式 API 修复
│   ├── selective_scan_cuda.py  # selective_scan shim
│   ├── mamba_ssm/         # mamba-ssm 修改文件
│   │   ├── ops/
│   │   │   ├── selective_scan_interface.py
│   │   │   └── triton/ssd_combined.py
│   │   └── distributed/distributed_utils.py
│   └── yolo/
│       └── mamba_yolo.py  # YOLO Mamba 集成
├── scripts/               # 安装和验证脚本
│   ├── apply_patches.py   # 应用补丁
│   ├── install.py         # 安装修改后的文件
│   └── verify.py          # 验证安装
└── docs/                  # 文档
    ├── BUILD_GUIDE.md     # 编译指南
    └── FIX_EXPLANATION.md # 修复说明
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
