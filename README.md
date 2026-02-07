# cuTile LSP Setup and Usage Guide

> [中文版本](docs/README_zh.md)

## Environment Setup

- **Python Environment**: Make sure to use the Python interpreter you want to select in VS Code, and install `cutile-lsp` in that environment.
- **Dependency Installation**: Run the following in the project root directory:

```sh
pip install -e ".[dev]"
```

- Install npm dependencies:

```sh
npm install
npm install -g @vscode/vsce
```

## Install VS Code Extension

```sh
bash build_ext.sh
```

## Select the Correct Python Interpreter

The extension will use the currently selected interpreter from the Python extension to start the language server:

- Open the command palette in VS Code and select **Python: Select Interpreter**.
- Choose the interpreter that has `cutile-lsp` installed.

If `cutile-lsp` is not installed in the selected interpreter, the extension will show a warning and will not start the LSP.

## Enable cuTile LSP File Markers

The LSP will only process **explicitly enabled** files. Please add the enable marker at the top of the file:

```py
# cutile-lsp: on
```

And use the following markers to delimit the code section that needs analysis:

```py
# cutile-lsp: start
# ... your kernels and helper functions ...
# cutile-lsp: end
```

## Key Patterns for LSP Features

### 1. Kernels Requiring Type Hints

Add a `<typecheck>` block in the kernel's docstring, **one per line**, and the LSP will perform type checking based on these parameters and generate diagnostics:

> For detailed annotation instructions, see [docs/annotation_instructions.md](docs/annotation_instructions.md).

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

### 2. Inlay Hints (Type Annotations)

When the file is enabled and syntactically correct, the LSP will automatically display type hints in the editor. They will refresh after saving or modifying the file.

## Debug

```sh
python -m cutile_lsp.lsp_pipeline.pipeline examples/kernel.py
```

## FAQ

- **LSP not starting**: Confirm that the current Python interpreter has `cutile-lsp` installed and is selected in VS Code.
- **No hints or diagnostics**: Confirm that the file has `# cutile-lsp: on` at the top, and the code contains `# cutile-lsp: start` and `# cutile-lsp: end` markers.
- **Incorrect hint positions**: Ensure `# cutile-lsp: start` is placed before the code block to be analyzed, and do not nest multiple `start/end` pairs.
