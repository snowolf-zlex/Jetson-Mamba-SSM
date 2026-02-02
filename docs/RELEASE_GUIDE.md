# 发布指南

## GitHub Release 工作流程

### 1. 准备发布文件

```bash
# 在项目根目录运行
python scripts/utils/prepare_release.py
```

这会创建 `release/YYYY-MM-DD/` 目录，包含：
- `mamba_ssm-2.2.4+jetson-cp310-cp310-linux_aarch64.whl`
- `causal_conv1d-1.6.0+jetson-cp310-cp310-linux_aarch64.whl`
- `mamba_ssm_so_files_YYYYMMDD_HHMMSS.tar.gz`
- `RELEASE_NOTES.md`
- `install.sh`

### 2. 创建 GitHub Release

#### 方法 A: 使用脚本（推荐）

```bash
# 设置 GitHub Token
export GITHUB_TOKEN=your_token

# 发布到 GitHub
python scripts/utils/release.py --tag v2.2.4+jetson
```

#### 方法 B: 手动发布

1. 访问 https://github.com/snowolf-zlex/Jetson-Mamba-SSM/releases/new
2. 创建新 tag: `v2.2.4+jetson`
3. 上传 `release/YYYY-MM-DD/` 目录下的所有文件
4. 复制 `RELEASE_NOTES.md` 内容到发布说明

### 3. 目录结构说明

```
jetson-mamba-ssm/
├── wheels/          # 开发用，保留在仓库
│   └── *.whl        # 小文件，可提交到 git
│
├── release/         # 发布用，不提交到 git
│   └── YYYY-MM-DD/  # 按日期组织
│       ├── *.whl    # 大文件，通过 GitHub Release 发布
│       ├── *.tar.gz # .so 备份，通过 GitHub Release 发布
│       └── *.md
│
└── backup/          # 已废弃，内容合并到 release/
```

### 4. 大文件管理策略

| 文件类型 | 存储位置 | 说明 |
|---------|---------|------|
| 源代码 | Git 仓库 | < 1 MB |
| 补丁文件 | Git 仓库 | < 100 KB |
| 脚本 | Git 仓库 | < 100 KB |
| 文档 | Git 仓库 | < 500 KB |
| **Wheel 文件** | **GitHub Release** | ~500 MB |
| **.so 备份** | **GitHub Release** | ~500 MB |

### 5. 版本命名规范

- Wheel: `{package}-{version}+jetson-cp310-cp310-linux_aarch64.whl`
- Tag: `v{version}+jetson`
- 示例: `v2.2.4+jetson`

### 6. 发布检查清单

- [ ] 运行 `prepare_release.py` 创建发布包
- [ ] 测试 wheel 文件可正常安装
- [ ] 测试 .so 文件可用
- [ ] 更新 RELEASE_NOTES.md
- [ ] 创建 GitHub Release
- [ ] 上传所有文件
- [ ] 验证下载链接可访问
