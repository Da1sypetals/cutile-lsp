# cuTile LSP 启用与使用指南

> [English Version](../README.md)

## 环境准备

- **Python 环境**：确保使用你要在 VS Code 中选择的 Python 解释器，并在该环境中安装 `cutile-lsp`。
- **依赖安装**：在项目根目录执行：

```sh
pip install -e ".[dev]"
```

- 安装npm依赖

```sh
npm install
npm install -g @vscode/vsce
```

## 安装 VS Code 扩展

```sh
bash build_ext.sh
```

## 选择正确的 Python 解释器

扩展会使用 Python 扩展当前选中的解释器来启动语言服务：

- 在 VS Code 中打开命令面板，选择 **Python: Select Interpreter**。
- 选择已安装 `cutile-lsp` 的解释器。

如果解释器中未安装 `cutile-lsp`，扩展会提示警告并不会启动 LSP。

## 启用 cuTile LSP 的文件标记

LSP 只会处理**显式启用**的文件。请在文件顶部加入启用标记：

```py
# cutile-lsp: on
```

并且用以下标记圈定需要分析的代码段:

```py
# cutile-lsp: start
# ... your kernels and helper functions ...
# cutile-lsp: end
```

## 获得 LSP 功能的关键写法

### 1. 需要类型提示的 kernel

在 kernel 的 docstring 中加入 `<typecheck>` 块, **每行一个**，LSP 会根据这些参数进行类型检查并产生诊断信息：

> 详细的注释规范请参考 [docs/annotation_instructions.md](docs/annotation_instructions.md)。

```py
@ct.kernel
def your_kernel(X, Y, TILE_N: ConstInt):
    """
    <typecheck>
    Tensor((1024, 2048), dtype="float16")
    Tensor((1024, 2048), dtype="float16")
    1024
    </typecheck>
    """
    ...
```

### 2. Inlay Hints（类型提示）

当文件被启用且语法正确时，LSP 会在编辑器中自动显示类型提示。它们会在保存或修改文件后刷新。

## Debug

```sh
python -m cutile_lsp.lsp_pipeline.pipeline examples/kernel.py
```

## 常见问题

- **LSP 没启动**：确认当前 Python 解释器已安装 `cutile-lsp` 并已在 VS Code 中选中。
- **没有任何提示或诊断**：
  - 确认文件顶部有 `# cutile-lsp: on`，并且代码包含 `# cutile-lsp: start` 和 `# cutile-lsp: end` 标记。
  - 同时确保必要的导入语句包含在此区域内。
  - 不要从其他文件导入。本扩展不具备且不会具备处理此情况的能力。
  - 请像使用 C++ 一样编写代码，避免使用花哨的 Python 动态技巧。
- **提示位置不正确**：确保 `# cutile-lsp: start` 位于分析代码块之前，并且不要嵌套多个 `start/end` 对。
- **提示无响应**：避免在 `# cutile-lsp: start` 和 `# cutile-lsp: end` 之间导入繁重的依赖项（比如`torch`），因为扩展会执行中间的代码。
