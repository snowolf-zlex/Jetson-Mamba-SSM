# YOLOv10 + Mamba TensorRT 导出完整指南

本指南详细说明如何在 NVIDIA Jetson Orin 上将 YOLOv10 + Mamba SSM 模型导出为 TensorRT 引擎格式。

## 目录

1. [环境准备](#环境准备)
2. [安装 mamba-ssm](#安装-mamba-ssm)
3. [应用 ONNX 导出补丁](#应用-onnx-导出补丁)
4. [应用 ultralytics 补丁](#应用-ultralytics-补丁)
5. [ONNX 导出](#onnx-导出)
6. [TensorRT 导出](#tensorrt-导出)
7. [推理测试](#推理测试)
8. [故障排除](#故障排除)

---

## 环境准备

### 系统要求

- **硬件**: NVIDIA Jetson Orin (或兼容的 Jetson 设备)
- **操作系统**: Linux for Tegra (L4T)
- **CUDA**: 12.6
- **Python**: 3.10
- **PyTorch**: 2.5.0a0+ (JetPack 版本)
- **TensorRT**: 10.x

### 安装基础依赖

```bash
# 更新系统包
sudo apt-get update && sudo apt-get install -y \
    python3-dev python3-pip \
    build-essential git cmake \
    ninja-build

# 安装 PyTorch (JetPack 版本)
sudo pip install torch==2.5.0a0+* --extra-index-url https://pypi.nvidia.com

# 安装 ONNX 和 TensorRT
sudo pip install onnx==1.14.1 onnxruntime-gpu
# TensorRT 通常随 JetPack 安装，或单独安装
```

---

## 安装 mamba-ssm

### 方法 1: 使用预编译 wheel (推荐)

```bash
# 克隆项目
git clone https://github.com/snowolf-zlex/Jetson-Mamba-SSM.git
cd jetson-mamba-ssm

# 安装 causal_conv1d
pip install wheels/causal_conv1d-1.6.0+jetson-cp310-cp310-linux_aarch64.whl

# 安装 mamba-ssm
pip install wheels/mamba_ssm-2.2.4+jetson-cp310-cp310-linux_aarch64.whl

# 验证安装
python scripts/test/verify.py
```

### 方法 2: 从源码编译

```bash
# 克隆项目
git clone https://github.com/snowolf-zlex/Jetson-Mamba-SSM.git
cd jetson-mamba-ssm

# 查看编译指南
cat docs/JETSON_MAMBA_SSM_BUILD_GUIDE.md

# 执行编译
python scripts/patch/apply_patches.py  # 编译前补丁
pip install -e .  # 编译安装
```

---

## 应用 ONNX 导出补丁

ONNX 导出补丁为 mamba-ssm 添加了 ONNX 兼容模式。

```bash
# 进入项目目录
cd jetson-mamba-ssm

# 应用 ONNX 导出补丁
python scripts/patch/apply_onnx_export_patch.py
```

预期输出:
```
============================================================
YOLOv10 + Mamba ONNX/TensorRT 导出补丁
============================================================
[INFO] 修补 mamba_ssm/ops/selective_scan_interface.py...
[✓] mamba-ssm ONNX 导出补丁已应用
[INFO] 修补 mamba_ssm/modules/mamba_simple.py...
[✓] mamba_simple.py ONNX 导出补丁已应用
[INFO] 修补 mamba_ssm/modules/mamba2_simple.py...
[✓] mamba2_simple.py ONNX 导出补丁已应用

============================================================
[✓] 所有补丁应用成功！
============================================================
```

验证补丁:
```bash
python -c "
from mamba_ssm.ops.selective_scan_interface import ONNX_EXPORT_MODE
print('✓ ONNX_EXPORT_MODE 可用' if 'ONNX_EXPORT_MODE' in dir() else '✗ 补丁未应用')
"
```

---

## 应用 ultralytics 补丁

ultralytics 补丁添加了自定义 ONNX 符号函数和 TensorRT 10.x API 支持。

### 安装 ultralytics

```bash
# 如果使用 YOLOv10 自定义版本
cd /path/to/yolov10
pip install -e .
```

### 应用 TensorRT 导出补丁

```bash
# 进入项目目录
cd jetson-mamba-ssm

# 注意: ultralytics 8.3.55+ 已内置支持，无需补丁
python scripts/main.py patch-ultra
```

预期输出:
```
============================================================
YOLOv10 + Mamba TensorRT 导出补丁 (ultralytics)
============================================================
[✓] exporter.py ONNX/TensorRT 补丁已应用
[✓] autobackend.py TensorRT 10.x 推理补丁已应用
```

---

## ONNX 导出

### 基本用法

```bash
cd /path/to/your/weights
yolo export model=best.pt format=onnx opset=17 simplify=False
```

### Python API

```python
from ultralytics import YOLO
from mamba_ssm.ops.selective_scan_interface import _set_onnx_export_mode
import torch

# 启用 ONNX 导出模式
_set_onnx_export_mode(True)

# 加载模型
model = YOLO('best.pt')

# 导出 ONNX
model.export(format='onnx', opset=17, simplify=False)

# 关闭 ONNX 导出模式
_set_onnx_export_mode(False)
```

### 预期输出

```
Ultralytics YOLOv8.3.55 🚀 Python-3.10.12 torch-2.5.0a0+872d972e41.nv24.08
YOLOv10s_add_head_EFC_全局感知 summary: 512 layers, 6728319 parameters, 0 gradients, 32.4 GFLOPs

[34m[1mMamba SSM:[0m Enabled ONNX export mode

[34m[1mPyTorch:[0m starting from 'best.pt' with input shape (1, 3, 640, 640) BCHW...

[34m[1mONNX:[0m export success ✅ 3.3s, saved as 'best.onnx' (35.7 MB)
```

---

## TensorRT 导出

### 基本用法

```bash
cd /path/to/your/weights
yolo export model=best.pt format=engine imgsz=640 device=0
```

这将自动完成:
1. ONNX 导出
2. TensorRT 引擎构建

### 精度选项

| 精度 | 参数 | 需要 yaml | 说明 |
|------|------|-----------|------|
| FP32 | `half=False` | ❌ | 最高精度，基准测试 |
| FP16 | `half=True` | ❌ | 推荐使用，精度损失可忽略 |
| INT8 | `int8=True` | ❌ | 动态量化，快速测试 |
| INT8 | `int8=True data=data.yaml` | ✅ | 校准量化，生产推荐 |

```bash
# FP32
yolo export model=best.pt format=engine imgsz=640 device=0 half=False

# FP16 (推荐)
yolo export model=best.pt format=engine imgsz=640 device=0 half=True

# INT8 动态量化 (无需 yaml)
yolo export model=best.pt format=engine imgsz=640 device=0 int8=True

# INT8 校准量化 (需要 yaml)
yolo export model=best.pt format=engine imgsz=640 device=0 int8=True data=data.yaml
```

### INT8 导出详解

#### 何时不需要 yaml

使用 **动态量化** 模式，TensorRT 自动估算量化参数：

```bash
# 无需 yaml，快速测试
yolo export model=best.pt format=engine imgsz=640 device=0 int8=True
```

**适用场景**:
- 快速验证 INT8 功能
- 无校准数据集
- 精度要求不极端严格

#### 何时需要 yaml

使用 **校准量化** 模式，使用真实数据优化量化参数：

```bash
# 需要 yaml，生产推荐
yolo export model=best.pt format=engine imgsz=640 device=0 int8=True data=data.yaml
```

**data.yaml 格式**:
```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 5  # 类别数量
names: ['class1', 'class2', 'class3', 'class4', 'class5']
```

**适用场景**:
- 生产环境部署
- 追求最优 INT8 精度
- 有完整的训练数据集

### Python API

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('best.pt')

# 导出 TensorRT
model.export(format='engine', imgsz=640, device=0)
```

### 预期输出

```
[34m[1mMamba SSM:[0m Enabled ONNX export mode
[34m[1mPyTorch:[0m starting from 'best.pt'...
[34m[1mONNX:[0m export success ✅ 3.3s...
[34m[1mTensorRT:[0m starting export with TensorRT 10.7.0...
...
[34m[1mTensorRT:[0m export success ✅ 423.6s, saved as 'best.engine' (32.7 MB)
```

---

## 推理测试

### TensorRT 推理

```bash
# 基本推理
yolo detect predict model=best.engine source=/path/to/image.jpg task=detect

# 批量推理
yolo detect predict model=best.engine source=/path/to/images/ task=detect

# Webcam 推理
yolo detect predict model=best.engine source=0 task=detect
```

### Python API

```python
from ultralytics import YOLO

# 加载 TensorRT 引擎
model = YOLO('best.engine')

# 推理
results = model('/path/to/image.jpg')
results[0].show()
```

### 性能数据 (Jetson Orin)

| 模型格式 | 预处理 | 推理 | 后处理 | 总计 |
|----------|--------|------|--------|------|
| TensorRT | 29.8ms | 40.5ms | 57.8ms | 128.1ms |

---

## 抑制警告信息

### FutureWarning 警告

PyTorch 和 ultralytics 可能会输出 FutureWarning 警告信息（如 `torch.cuda.amp.autocast` 已弃用），不影响功能但影响可读性。

### 方法 1: 使用环境变量（推荐）

```bash
# 关闭所有 FutureWarning
PYTHONWARNINGS=ignore::FutureWarning yolo export model=best.pt format=engine

# 推理时关闭警告
PYTHONWARNINGS=ignore::FutureWarning yolo detect predict model=best.engine source=image.jpg
```

### 方法 2: 设置永久别名

编辑 `~/.bashrc` 添加：

```bash
# YOLO 无警告别名
alias yolo='PYTHONWARNINGS=ignore::FutureWarning yolo'
```

然后执行：
```bash
source ~/.bashrc
yolo export model=best.pt format=engine  # 自动无警告
```

### 方法 3: Python 代码中抑制

```python
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='engine')
```

---

## 故障排除

### 问题 1: aten::exponential 不支持

**错误信息**:
```
torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::exponential' to ONNX opset version 17 is not supported
```

**原因**: 未应用自定义 ONNX 符号函数补丁

**解决方案**:
```bash
cd jetson-mamba-ssm
# 注意: ultralytics 8.3.55+ 已内置支持，无需补丁
```

### 问题 2: RuntimeError: Expected u.is_cuda() to be true

**错误信息**:
```
RuntimeError: Expected u.is_cuda() to be true, but got false
```

**原因**: Mamba ONNX 导出模式未启用

**解决方案**:
```python
from mamba_ssm.ops.selective_scan_interface import _set_onnx_export_mode
_set_onnx_export_mode(True)
```

### 问题 3: TensorRT 10.x API 错误

**错误信息**:
```
AttributeError: 'IBuilderConfig' object has no attribute 'max_workspace_size'
AttributeError: 'ICudaEngine' object has no attribute 'num_bindings'
```

**原因**: TensorRT 10.x API 变化

**解决方案**:
```bash
cd jetson-mamba-ssm
# 注意: ultralytics 8.3.55+ 已内置支持，无需补丁
```

### 问题 4: 推理时出错

**错误信息**:
```
AttributeError: module 'tensorrt' has no attribute 'TensorMode'
```

**原因**: 使用了错误的枚举名 (TensorMode vs TensorIOMode)

**解决方案**: 确保 autobackend.py 补丁已正确应用

### 问题 5: ImportError: cannot import name 'entrypoint'

**错误信息**:
```
ImportError: cannot import name 'entrypoint' from 'ultralytics.cfg'
```

**原因**: 使用了系统的 ultralytics 而不是 YOLOv10 版本

**解决方案**:
```bash
export PYTHONPATH="/path/to/yolov10:$PYTHONPATH"
```

---

## 完整工作流程示例

```bash
# 1. 克隆项目
git clone https://github.com/snowolf-zlex/Jetson-Mamba-SSM.git
cd Jetson-Mamba-SSM

# 2. 安装 mamba-ssm (如果尚未安装)
python scripts/main.py install

# 3. 验证安装
python scripts/main.py verify

# 4. 运行测试
python scripts/main.py test

# 5. 导出模型
cd /path/to/your/weights
yolo export model=best.pt format=engine imgsz=640 device=0

# 6. 测试推理
yolo detect predict model=best.engine source=/path/to/image.jpg task=detect
```

---

## 文件清单

| 脚本 | 用途 |
|------|------|
| `scripts/apply_onnx_export_patch.py` | 应用 mamba-ssm ONNX 导出补丁 |
| `scripts/apply_ultralytics_patch.py` | 应用 ultralytics TensorRT 补丁 |
| `scripts/test_onnx_tensorrt_export.py` | 测试 ONNX/TensorRT 导出功能 |
| `scripts/apply_patches.py` | 应用 mamba-ssm 基础运行时补丁 |
| `scripts/install_wheels.py` | 安装预编译 wheel |
| `scripts/verify.py` | 验证 mamba-ssm 安装 |

---

## 相关文档

- [JETSON_MAMBA_SSM_BUILD_GUIDE.md](JETSON_MAMBA_SSM_BUILD_GUIDE.md) - Mamba-SSM 编译指南
- [MAMBA_SSM_JETSON_FIX.md](MAMBA_SSM_JETSON_FIX.md) - Mamba-SSM Jetson 修复记录
- [CAUSAL_CONV1D.md](CAUSAL_CONV1D.md) - causal-conv1d 安装指南

---

## 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2026-02-02 | v1.0 | 初始版本，支持 YOLOv10 + Mamba ONNX/TensorRT 完整导出流程 |
