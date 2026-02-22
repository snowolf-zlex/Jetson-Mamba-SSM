# GitHub 仓库设置检查清单

## 如何在 GitHub 网页上验证

### 1. 检查 About 区域（右侧边栏）

访问：`https://github.com/snowolf-zlex/Jetson-Mamba-SSM`

**检查项：**
- [ ] 是否有简短描述（150 字符内）
- [ ] 描述是否包含关键词：Mamba, Jetson, YOLOv10, TensorRT, ARM64

**建议设置：**
```
Complete solution for running YOLOv10 + Mamba SSM on NVIDIA Jetson.
Supports ONNX export, TensorRT (FP32/FP16/INT8), ARM64 optimization.
```

---

### 2. 检查 Topics（标签）

在 About 区域下方，点击 "Add topics"

**检查项：**
- [ ] 是否添加了以下 topics：
  - mamba
  - jetson
  - tensorrt
  - yolov10
  - onnx
  - nvidia
  - arm64
  - deep-learning
  - edge-ai
  - state-space-model
  - cuda
  - object-detection

---

### 3. 检查 README 显示

**检查项：**
- [ ] README.md 是否正确显示
- [ ] 徽章图片是否正常加载
- [ ] 语言选择链接是否可点击
- [ ] 目录结构是否清晰

---

### 4. 检查文件结构

访问以下路径确认文件存在：

**检查项：**
- [ ] `/CITATION.cff` - 学术引用文件
- [ ] `/.github/FUNDING.yml` - 赞助配置
- [ ] `/.github/logo.svg` - Logo 文件
- [ ] `/.github/assets/banner.svg` - 社交媒体预览图

---

### 5. 检查 SEO 效果

**Google 搜索测试**（等待 1-2 周后）：

在 Google 搜索以下关键词，看是否能找到项目：
- `site:github.com Jetson Mamba SSM`
- `site:github.com snowolf-zlex Jetson`
- `Jetson TensorRT YOLOv10 ARM64`

**GitHub 站内搜索测试**（立即生效）：

在 GitHub 搜索：
- `topic:mamba jetson`
- `topic:tensorrt yolov10`

---

## 本地已提交的文件

以下文件已创建并提交到本地仓库：

```
✅ README.md (主入口，含语言链接)
✅ README.en-US.md (完整英文文档)
✅ README.zh-CN.md (完整中文文档)
✅ CITATION.cff (学术引用)
✅ pyproject.toml (更新关键词和 URL)
✅ .github/FUNDING.yml
✅ .github/logo.svg
✅ .github/assets/banner.svg
✅ .github/ISSUE_TEMPLATE/bug_report.yml
✅ .github/ISSUE_TEMPLATE/feature_request.yml
✅ .github/pull_request_template.md
✅ SEO_CHECKLIST.md
```

---

## 需要推送的代码

```bash
# 提交所有更改
git add .
git commit -m "docs: SEO optimization and GitHub configuration

- Add language selection links to all README files
- Add keywords footer for SEO
- Add CITATION.cff for academic citation
- Update pyproject.toml with comprehensive keywords
- Add GitHub templates (issues, PR, funding)
- Add logo and banner assets
- Create SEO checklist"

# 推送到 GitHub
git push origin main
```

---

## 需要在 GitHub 网页手动设置

### 设置 About 和 Topics

1. 访问 `https://github.com/snowolf-zlex/Jetson-Mamba-SSM`
2. 点击右侧齿轮图标（⚙️）编辑 About
3. 输入描述和 topics
4. 点击 "Save changes"

### 设置 Website（可选）

如果有项目网站，可在 About 中添加 Website 链接。

---

## 验证 SEO 效果的第三方工具

| 工具 | 用途 |
|------|------|
| https://seositecheckup.com/ | 免费 SEO 分析 |
| https://search.google.com/search-console | Google 索引监控 |
| https://trends.google.com/ | 关键词趋势分析 |
