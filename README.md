# Jetson Mamba-SSM

![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin-32B3E6?logo=nvidia)
![Architecture](https://img.shields.io/badge/Architecture-ARM64-E96479?logo=arm)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia)
![TensorRT](https://img.shields.io/badge/TensorRT-10.7.0-76B900?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-green)

## 项目定位

**Jetson-Mamba-SSM** 是一套完整的解决方案，支持 **YOLOv10 + Mamba SSM** 模型在 NVIDIA Jetson (ARM64) 上：
- ✅ 运行 Mamba-SSM 模型
- ✅ 导出 ONNX 格式
- ✅ 编译 TensorRT 引擎
- ✅ 多精度推理 (FP32/FP16/INT8)

### 核心特性

| 功能 | 说明 | 状态 |
|------|------|------|
| **Mamba-SSM 运行** | 在 Jetson (ARM64) 上运行 Mamba-SSM 模型 | ✅ 支持 |
| **YOLOv10 + Mamba 集成** | 支持 YOLOv10 中使用 Mamba 模块 | ✅ 支持 |
| **ONNX 导出** | 导出 YOLOv10 + Mamba 模型为 ONNX 格式 | ✅ 支持 |
| **TensorRT 导出** | 编译 TensorRT 引擎 (FP32/FP16/INT8) | ✅ 支持 |
| **多精度推理** | 支持 FP32、FP16、INT8 精度 | ✅ 支持 |
| **TensorRT 10.x** | 兼容 TensorRT 10.x API | ✅ 支持 |

### 技术亮点

- 🎯 **完整 Wheel 方案**: 所有补丁打包为 wheel，一键安装
- 🔧 **libc10.so 兼容**: 解决 Jetson 平台依赖问题
- 📦 **ONNX 导出支持**: Mamba 模块可导出为 ONNX
- 🚀 **TensorRT 优化**: 支持 FP32/FP16/INT8 精度
- 📱 **跨平台兼容**: 支持 Jetson Orin/Xavier/Nano

## 快速开始

### 1. 一键安装 (推荐)

#### 方法 A: 从 GitHub Release 安装

```bash
# 从 GitHub Release 下载最新版本
# 访问: https://github.com/snowolf-zlex/Jetson-Mamba-SSM/releases
# 下载以下文件:
#   - causal_conv1d-1.6.0+jetson-cp310-cp310-linux_aarch64.whl
#   - mamba_ssm-2.2.4+jetson-cp310-cp310-linux_aarch64.whl

# 安装
pip install causal_conv1d-1.6.0+jetson-cp310-cp310-linux_aarch64.whl
pip install mamba_ssm-2.2.4+jetson-cp310-cp310-linux_aarch64.whl
```

#### 方法 B: 从源码安装

```bash
# 克隆项目
git clone https://github.com/snowolf-zlex/Jetson-Mamba-SSM.git
cd Jetson-Mamba-SSM

# 安装完整的 wheel 包
pip install wheels/causal_conv1d-1.6.0+jetson-cp310-cp310-linux_aarch64.whl
pip install wheels/mamba_ssm-2.2.4+jetson-cp310-cp310-linux_aarch64.whl
```

### 2. 导出 TensorRT 引擎

```bash
cd /path/to/your/weights

# 导出 TensorRT (默认 FP16)
yolo export model=best.pt format=engine imgsz=640 device=0

# 导出指定精度
yolo export model=best.pt format=engine imgsz=640 device=0 half=True   # FP16
yolo export model=best.pt format=engine imgsz=640 device=0 half=False  # FP32 

# ❌ 不需要 yaml - 动态量化 (快速测试)                                                                                            
yolo export model=best.pt format=engine imgsz=640 device=0 int8=True                                                              
                                                                                                                                  
# ✅ 需要 yaml - 校准量化 (生产推荐)                                                                                              
yolo export model=best.pt format=engine imgsz=640 device=0 int8=True data=data.yaml   
```

### 3. 推理测试

```bash
# TensorRT 推理
yolo detect predict model=best.engine source=/path/to/image.jpg

# 指定精度推理
yolo detect predict model=best.engine source=/path/to/image.jpg half=True   # FP16
```

## 安装包说明

### Wheel 文件

| 文件 | 版本 | 大小 | 包含内容 |
|------|------|------|----------|
| `causal_conv1d-1.6.0+jetson-*.whl` | 1.6.0+jetson | 185 MB | libc10.so 兼容层 |
| `mamba_ssm-2.2.4+jetson-*.whl` | 2.2.4+jetson | 310 MB | libc10.so 修复 + ONNX 导出 |

### 包含的补丁

#### mamba_ssm-2.2.4+jetson.whl

- ✅ **libc10.so 依赖修复**: 使用 `causal_conv1d_fn` 替代 `causal_conv1d_cuda`
- ✅ **ONNX 导出支持**: 添加 `ONNX_EXPORT_MODE` 和 CPU fallback
- ✅ **torch.exp() 替换**: ONNX 兼容的指数运算

#### causal_conv1d-1.6.0+jetson.whl

- ✅ **causal_conv1d_cuda 兼容层**: 自动注册兼容模块
- ✅ **无外部依赖**: 无需 sitecustomize.py

## 精度支持

### FP32 (单精度)

```bash
# 导出 FP32 TensorRT
yolo export model=best.pt format=engine imgsz=640 half=False

# 推理
yolo detect predict model=best.engine half=False
```

**特点**:
- 精度最高
- 推理速度较慢
- 显存占用较大

### FP16 (半精度)

```bash
# 导出 FP16 TensorRT (默认)
yolo export model=best.pt format=engine imgsz=640 half=True

# 推理
yolo detect predict model=best.engine half=True
```

**特点**:
- 精度接近 FP32
- 推理速度快 ~2x
- 显存占用减少 ~50%

### INT8 (整数量化)

```python
from ultralytics import YOLO

model = YOLO('best.pt')

# 动态量化 (无需 yaml)
model.export(format='engine', imgsz=640, int8=True)

# 校准量化 (需要 yaml，精度更高)
model.export(format='engine', imgsz=640, int8=True, data='data.yaml')
```

**特点**:
- 精度略有下降
- 推理速度最快 ~4x
- 显存占用最少
- **可选**校准数据集提升精度

**是否需要 yaml**:
| 模式 | 需要 yaml | 精度 | 使用场景 |
|------|-----------|------|----------|
| 动态量化 | ❌ | 中等 | 快速测试 |
| 校准量化 | ✅ | 高 | 生产部署 |

## 性能数据 (Jetson Orin)

| 模型格式 | 精度 | 文件大小 | 推理速度 (640x640) | 显存占用 |
|----------|------|----------|---------------------|----------|
| PyTorch (.pt) | FP32 | 28.0 MB | - | - |
| ONNX (.onnx) | FP32 | 35.7 MB | - | - |
| TensorRT (.engine) | FP32 | 32.7 MB | 80ms | 2.1 GB |
| TensorRT (.engine) | FP16 | 32.7 MB | 40ms | 1.1 GB |
| TensorRT (.engine) | INT8 | 32.7 MB | 25ms | 0.6 GB |

## 项目结构

```
jetson-mamba-ssm/
├── README.md                              # 本文件
├── LICENSE                                # MIT 许可证
│
├── wheels/                                # 预编译 wheel (开发用)
│   ├── causal_conv1d-1.6.0+jetson-*.whl   # ✨ 完整版
│   └── mamba_ssm-2.2.4+jetson-*.whl       # ✨ 完整版
│
├── release/                               # GitHub Release 发布包
│   └── YYYY-MM-DD/                        # 按日期组织
│       ├── *.whl                          # Wheel 文件
│       ├── *_so_files_*.tar.gz            # .so 备份
│       ├── RELEASE_NOTES.md               # 发布说明
│       └── install.sh                     # 一键安装脚本
│
├── patches/                               # Git 格式补丁 (源码编译用)
│   ├── 00_selective_scan_interface.py.patch
│   └── 01_ssd_combined.py.patch
│
├── src/                                   # 修改后的源文件 (参考)
│   ├── fix_causal_conv1d.py
│   ├── sitecustomize/
│   ├── mamba_ssm/
│   └── yolo/
│
├── scripts/                               # 工具脚本
│   ├── main.py                            # 统一入口
│   ├── install/                           # 安装脚本
│   ├── patch/                             # 补丁脚本
│   ├── test/                              # 测试脚本
│   └── utils/                             # 工具脚本
│
└── docs/                                  # 完整文档
    ├── YOLOV10_TENSORRT_EXPORT_GUIDE.md   # TensorRT 导出指南
    ├── JETSON_MAMBA_SSM_BUILD_GUIDE.md    # 编译指南
    └── ...
```

## 统一入口命令

```bash
python scripts/main.py <命令>

命令:
  install          一键完整安装
  verify           验证安装
  test             运行测试
  rebuild          重新打包 wheel
  info             显示项目信息
```

## 构建环境

| 组件 | 版本 |
|------|------|
| **硬件** | Jetson Orin (ARM64, Ampere GPU) |
| **操作系统** | Linux 5.15.148-tegra (JetPack R36) |
| **CUDA** | 12.6 |
| **TensorRT** | 10.7.0 |
| **Python** | 3.10.12 |

## 兼容性

| 设备 | 架构 | 状态 |
|------|------|------|
| Jetson Orin | ARM64 | ✅ 完全支持 |
| Jetson Xavier | ARM64 | ✅ 支持 |
| Jetson Nano | ARM64 | ✅ 支持 |

## 文档

| 文档 | 说明 |
|------|------|
| [YOLOV10_TENSORRT_EXPORT_GUIDE.md](docs/YOLOV10_TENSORRT_EXPORT_GUIDE.md) | TensorRT 完整导出指南 |
| [PRECISION_EXPORT_TEST_REPORT.md](docs/PRECISION_EXPORT_TEST_REPORT.md) | FP32/FP16/INT8 精度测试报告 |
| [JETSON_MAMBA_SSM_BUILD_GUIDE.md](docs/JETSON_MAMBA_SSM_BUILD_GUIDE.md) | 从源码编译指南 |
| [MAMBA_SSM_JETSON_FIX.md](docs/MAMBA_SSM_JETSON_FIX.md) | Mamba-SSM Jetson 修复记录 |
| [RELEASE_GUIDE.md](docs/RELEASE_GUIDE.md) | GitHub Release 发布指南 |

## 测试工具

| 脚本 | 功能 |
|------|------|
| `scripts/test/verify.py` | 验证 mamba-ssm 基础安装 |
| `scripts/test/test_onnx_tensorrt_export.py` | 测试 ONNX/TensorRT 导出功能 |
| `scripts/test/test_export_precision.py` | 测试 FP32/FP16/INT8 精度导出 |
| `scripts/utils/model_info.py` | 查看 .pt/.onnx/.engine 模型信息 |

### 精度测试

```bash
# 测试所有精度导出
python scripts/test/test_export_precision.py

# 指定模型测试
python scripts/test/test_export_precision.py --model /path/to/model.pt
```

### 模型信息查看

```bash
# 查看任意格式模型信息
python scripts/utils/model_info.py best.pt
python scripts/utils/model_info.py best.onnx
python scripts/utils/model_info.py best.engine
```

## 许可证

MIT License

## 致谢

- [Mamba-SSM](https://github.com/state-spaces/mamba) - Tri Dao, Albert Gu
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) - Tri Dao
- [Ultralytics YOLOv10](https://github.com/THU-MIG/yolov10) - YOLOv10
