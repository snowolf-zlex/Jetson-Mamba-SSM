# Jetson Mamba-SSM

> **Complete Solution for Running YOLOv10 + Mamba SSM on NVIDIA Jetson**

![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin-32B3E6?logo=nvidia)
![Architecture](https://img.shields.io/badge/Architecture-ARM64-E96479?logo=arm)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia)
![TensorRT](https://img.shields.io/badge/TensorRT-10.7.0-76B900?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌍 Language / 语言选择

- **[🇺🇸 English](README.en-US.md)** - Complete documentation in English
- **[🇨🇳 中文](README.zh-CN.md)** - 完整的中文文档

---

## Project Overview

**Jetson-Mamba-SSM** is a complete solution for running **YOLOv10 + Mamba SSM** models on NVIDIA Jetson (ARM64):
- ✅ Run Mamba-SSM models on Jetson
- ✅ Export to ONNX format
- ✅ Compile TensorRT engines
- ✅ Multi-precision inference (FP32/FP16/INT8)

### Core Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Mamba-SSM Runtime** | Run Mamba-SSM models on Jetson (ARM64) | ✅ Supported |
| **YOLOv10 + Mamba Integration** | Support Mamba modules in YOLOv10 | ✅ Supported |
| **ONNX Export** | Export YOLOv10 + Mamba models to ONNX | ✅ Supported |
| **TensorRT Export** | Compile TensorRT engines (FP32/FP16/INT8) | ✅ Supported |
| **Multi-precision Inference** | Support FP32, FP16, INT8 precision | ✅ Supported |
| **TensorRT 10.x** | Compatible with TensorRT 10.x API | ✅ Supported |

### Technical Highlights

- 🎯 **Complete Wheel Package**: All patches packaged as wheel, one-click installation
- 🔧 **libc10.so Compatibility**: Resolves Jetson platform dependency issues
- 📦 **ONNX Export Support**: Mamba modules can be exported to ONNX
- 🚀 **TensorRT Optimized**: Supports FP32/FP16/INT8 precision
- 📱 **Cross-Platform**: Supports Jetson Orin/Xavier/Nano

## Quick Start

### Prerequisites

Before installing mamba-ssm, ensure the following dependencies are installed:

```bash
# Basic dependencies
pip install torch einops ninja packaging transformers

# triton (core GPU operator library for mamba-ssm)
# Source requirement: triton==2.1.0 or 2.2.0
# Tested compatible: triton>=2.1.0 (including 3.x)
pip install triton
```

**triton Version Notes**:
- mamba-ssm source requires `triton==2.1.0` or `2.2.0`
- Actually `triton>=2.1.0` works fine
- Jetson Orin tested: `triton 3.5.1` ✅ Compatible

### 1. One-Click Installation (Recommended)

#### Method A: Install from GitHub Release

```bash
# Download latest version from GitHub Release
# Visit: https://github.com/snowolf-zlex/Jetson-Mamba-SSM/releases
# Download the following files:
#   - causal_conv1d-1.6.0+jetson-cp310-cp310-linux_aarch64.whl
#   - mamba_ssm-2.2.4+jetson-cp310-cp310-linux_aarch64.whl

# Install
pip install causal_conv1d-1.6.0+jetson-cp310-cp310-linux_aarch64.whl
pip install mamba_ssm-2.2.4+jetson-cp310-cp310-linux_aarch64.whl
```

#### Method B: Install from Source

```bash
# Clone the project
git clone https://github.com/snowolf-zlex/Jetson-Mamba-SSM.git
cd Jetson-Mamba-SSM

# Install complete wheel packages
pip install wheels/causal_conv1d-1.6.0+jetson-cp310-cp310-linux_aarch64.whl
pip install wheels/mamba_ssm-2.2.4+jetson-cp310-cp310-linux_aarch64.whl
```

### 2. Export TensorRT Engine

```bash
cd /path/to/your/weights

# Export TensorRT (default FP16)
yolo export model=best.pt format=engine imgsz=640 device=0

# Export with specific precision
yolo export model=best.pt format=engine imgsz=640 device=0 half=True   # FP16
yolo export model=best.pt format=engine imgsz=640 device=0 half=False  # FP32

# ❌ No yaml needed - Dynamic quantization (quick test)
yolo export model=best.pt format=engine imgsz=640 device=0 int8=True

# ✅ Need yaml - Calibration quantization (production recommended)
yolo export model=best.pt format=engine imgsz=640 device=0 int8=True data=data.yaml
```

### 3. Inference Test

```bash
# TensorRT inference
yolo detect predict model=best.engine source=/path/to/image.jpg

# Inference with specific precision
yolo detect predict model=best.engine source=/path/to/image.jpg half=True   # FP16
```

## Package Description

### Wheel Files

| File | Version | Size | Contents |
|------|---------|------|----------|
| `causal_conv1d-1.6.0+jetson-*.whl` | 1.6.0+jetson | 185 MB | libc10.so compatibility layer |
| `mamba_ssm-2.2.4+jetson-*.whl` | 2.2.4+jetson | 310 MB | libc10.so fix + ONNX export |

### Included Patches

#### mamba_ssm-2.2.4+jetson.whl

- ✅ **libc10.so Dependency Fix**: Use `causal_conv1d_fn` instead of `causal_conv1d_cuda`
- ✅ **ONNX Export Support**: Add `ONNX_EXPORT_MODE` and CPU fallback
- ✅ **torch.exp() Replacement**: ONNX-compatible exponential operation

#### causal_conv1d-1.6.0+jetson.whl

- ✅ **causal_conv1d_cuda Compatibility Layer**: Auto-register compatible module
- ✅ **No External Dependencies**: No sitecustomize.py needed

## Precision Support

### FP32 (Single Precision)

```bash
# Export FP32 TensorRT
yolo export model=best.pt format=engine imgsz=640 half=False

# Inference
yolo detect predict model=best.engine half=False
```

**Characteristics**:
- Highest precision
- Slower inference
- Higher VRAM usage

### FP16 (Half Precision)

```bash
# Export FP16 TensorRT (default)
yolo export model=best.pt format=engine imgsz=640 half=True

# Inference
yolo detect predict model=best.engine half=True
```

**Characteristics**:
- Precision close to FP32
- Inference speed ~2x faster
- VRAM usage reduced ~50%

### INT8 (Integer Quantization)

```python
from ultralytics import YOLO

model = YOLO('best.pt')

# Dynamic quantization (no yaml needed)
model.export(format='engine', imgsz=640, int8=True)

# Calibration quantization (needs yaml, higher precision)
model.export(format='engine', imgsz=640, int8=True, data='data.yaml')
```

**Characteristics**:
- Slight precision drop
- Fastest inference ~4x
- Lowest VRAM usage
- **Optional** calibration dataset for better precision

**Whether yaml is needed**:
| Mode | yaml Required | Precision | Use Case |
|------|---------------|-----------|----------|
| Dynamic Quantization | ❌ | Medium | Quick testing |
| Calibration Quantization | ✅ | High | Production deployment |

## Performance Data (Jetson Orin)

| Model Format | Precision | File Size | Inference Time (640x640) | VRAM Usage |
|--------------|-----------|-----------|--------------------------|------------|
| PyTorch (.pt) | FP32 | 28.0 MB | - | - |
| ONNX (.onnx) | FP32 | 35.7 MB | - | - |
| TensorRT (.engine) | FP32 | 32.7 MB | 80ms | 2.1 GB |
| TensorRT (.engine) | FP16 | 32.7 MB | 40ms | 1.1 GB |
| TensorRT (.engine) | INT8 | 32.7 MB | 25ms | 0.6 GB |

## Project Structure

```
jetson-mamba-ssm/
├── README.md                              # This file
├── LICENSE                                # MIT License
│
├── wheels/                                # Pre-compiled wheel (for development)
│   ├── causal_conv1d-1.6.0+jetson-*.whl   # ✨ Complete version
│   └── mamba_ssm-2.2.4+jetson-*.whl       # ✨ Complete version
│
├── release/                               # GitHub Release packages
│   └── YYYY-MM-DD/                        # Organized by date
│       ├── *.whl                          # Wheel files
│       ├── *_so_files_*.tar.gz            # .so backup
│       ├── RELEASE_NOTES.md               # Release notes
│       └── install.sh                     # One-click install script
│
├── patches/                               # Git format patches (for source compilation)
│   ├── 00_selective_scan_interface.py.patch
│   └── 01_ssd_combined.py.patch
│
├── src/                                   # Modified source files (for reference)
│   ├── fix_causal_conv1d.py
│   ├── sitecustomize/
│   ├── mamba_ssm/
│   └── yolo/
│
├── scripts/                               # Utility scripts
│   ├── main.py                            # Unified entry point
│   ├── install/                           # Installation scripts
│   ├── patch/                             # Patch scripts
│   ├── test/                              # Test scripts
│   └── utils/                             # Utility scripts
│
└── docs/                                  # Complete documentation
    ├── YOLOV10_TENSORRT_EXPORT_GUIDE.md   # TensorRT export guide
    ├── JETSON_MAMBA_SSM_BUILD_GUIDE.md    # Build guide
    └── ...
```

## Unified Entry Point Commands

```bash
python scripts/main.py <command>

Commands:
  install          One-click complete installation
  verify           Verify installation
  test             Run tests
  rebuild          Rebuild wheel packages
  info             Display project information
```

## Build Environment

| Component | Version |
|-----------|---------|
| **Hardware** | Jetson Orin (ARM64, Ampere GPU) |
| **OS** | Linux 5.15.148-tegra (JetPack R36) |
| **CUDA** | 12.6 |
| **TensorRT** | 10.7.0 |
| **Python** | 3.10.12 |

## Compatibility

| Device | Architecture | Status |
|--------|--------------|--------|
| Jetson Orin | ARM64 | ✅ Fully Supported |
| Jetson Xavier | ARM64 | ✅ Supported |
| Jetson Nano | ARM64 | ✅ Supported |

## Documentation

| Document | Description |
|----------|-------------|
| [YOLOV10_TENSORRT_EXPORT_GUIDE.md](docs/YOLOV10_TENSORRT_EXPORT_GUIDE.md) | Complete TensorRT export guide |
| [PRECISION_EXPORT_TEST_REPORT.md](docs/PRECISION_EXPORT_TEST_REPORT.md) | FP32/FP16/INT8 precision test report |
| [JETSON_MAMBA_SSM_BUILD_GUIDE.md](docs/JETSON_MAMBA_SSM_BUILD_GUIDE.md) | Source compilation guide |
| [MAMBA_SSM_JETSON_FIX.md](docs/MAMBA_SSM_JETSON_FIX.md) | Mamba-SSM Jetson fix records |
| [RELEASE_GUIDE.md](docs/RELEASE_GUIDE.md) | GitHub Release guide |

## Test Tools

| Script | Function |
|--------|----------|
| `scripts/test/verify.py` | Verify mamba-ssm basic installation |
| `scripts/test/test_onnx_tensorrt_export.py` | Test ONNX/TensorRT export |
| `scripts/test/test_export_precision.py` | Test FP32/FP16/INT8 precision export |
| `scripts/utils/model_info.py` | View .pt/.onnx/.engine model information |

### Precision Test

```bash
# Test all precision exports
python scripts/test/test_export_precision.py

# Test with specific model
python scripts/test/test_export_precision.py --model /path/to/model.pt
```

### Model Information Viewer

```bash
# View model info in any format
python scripts/utils/model_info.py best.pt
python scripts/utils/model_info.py best.onnx
python scripts/utils/model_info.py best.engine
```

## License

MIT License

## Acknowledgments

- [Mamba-SSM](https://github.com/state-spaces/mamba) - Tri Dao, Albert Gu
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) - Tri Dao
- [Ultralytics YOLOv10](https://github.com/THU-MIG/yolov10) - YOLOv10 Team

---

## 🤝 Contributing

Contributions are welcome! Please visit our [GitHub Repository](https://github.com/snowolf-zlex/Jetson-Mamba-SSM) to:
- Report issues
- Submit pull requests
- Suggest features

---

**Keywords**: Jetson Mamba SSM, YOLOv10, TensorRT, ONNX, NVIDIA Jetson, ARM64, Deep Learning, Object Detection, State Space Model, Edge AI, Model Export, Inference Optimization
