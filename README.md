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

## 📌 Quick Overview / 快速概览

**Jetson-Mamba-SSM** enables YOLOv10 with Mamba SSM modules on NVIDIA Jetson devices (Orin/Xavier/Nano).

| Feature | Status |
|---------|--------|
| Mamba-SSM Runtime on Jetson | ✅ Supported |
| ONNX Export | ✅ Supported |
| TensorRT Engine (FP32/FP16/INT8) | ✅ Supported |
| TensorRT 10.x API | ✅ Supported |

### Performance on Jetson Orin

| Precision | Inference Time | VRAM |
|-----------|----------------|------|
| FP32 | 80ms | 2.1 GB |
| FP16 | 40ms | 1.1 GB |
| INT8 | 25ms | 0.6 GB |

---

## 🚀 Quick Start / 快速开始

### Install / 安装

```bash
# Install dependencies / 安装依赖
pip install torch einops ninja packaging transformers triton

# Install from GitHub Release / 从 GitHub Release 安装
pip install causal_conv1d-1.6.0+jetson-*.whl
pip install mamba_ssm-2.2.4+jetson-*.whl
```

### Export TensorRT / 导出 TensorRT

```bash
# Export FP16 engine / 导出 FP16 引擎
yolo export model=best.pt format=engine imgsz=640

# Run inference / 运行推理
yolo detect predict model=best.engine source=image.jpg
```

---

## 📚 Full Documentation / 完整文档

| Document | Description |
|----------|-------------|
| [English Docs](README.en-US.md) | Complete English documentation |
| [中文文档](README.zh-CN.md) | 完整的中文使用指南 |
| [TensorRT Export Guide](docs/YOLOV10_TENSORRT_EXPORT_GUIDE.md) | Step-by-step export guide |
| [Build Guide](docs/JETSON_MAMBA_SSM_BUILD_GUIDE.md) | Build from source instructions |

---

## 🔧 Key Features / 核心特性

- 🎯 **One-Click Installation** - Wheel packages with all patches included
- 🔧 **libc10.so Fixed** - Resolves Jetson platform dependencies
- 📦 **ONNX Export** - Mamba modules exportable to ONNX format
- 🚀 **TensorRT Optimized** - FP32/FP16/INT8 multi-precision support
- 📱 **Cross-Platform** - Supports Orin, Xavier, Nano

---

## 📦 Project Structure / 项目结构

```
jetson-mamba-ssm/
├── README.md              # Main entry (this file)
├── README.en-US.md        # English documentation
├── README.zh-CN.md        # Chinese documentation
├── wheels/                # Pre-compiled wheel packages
├── release/               # GitHub Release packages
├── patches/               # Git patches for source build
├── scripts/               # Utility scripts
│   ├── main.py            # Unified entry point
│   ├── install/           # Installation scripts
│   ├── test/              # Test scripts
│   └── utils/             # Utility scripts
└── docs/                  # Detailed documentation
```

---

## 📋 System Requirements / 系统要求

| Component | Version |
|-----------|---------|
| Hardware | Jetson Orin / Xavier / Nano |
| OS | Linux (JetPack R36) |
| CUDA | 12.6 |
| TensorRT | 10.7.0 |
| Python | 3.10+ |

---

## 🤝 Contributing / 贡献

Contributions are welcome! Please visit our [GitHub Repository](https://github.com/snowolf-zlex/Jetson-Mamba-SSM) to:
- Report issues
- Submit pull requests
- Suggest features

---

## 📄 License / 许可证

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments / 致谢

- [Mamba-SSM](https://github.com/state-spaces/mamba) - Tri Dao, Albert Gu
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) - Tri Dao
- [Ultralytics YOLOv10](https://github.com/THU-MIG/yolov10) - YOLOv10 Team

---

**Keywords**: Jetson Mamba SSM, YOLOv10, TensorRT, ONNX, NVIDIA Jetson, ARM64, Deep Learning, Object Detection, State Space Model, Edge AI
