---
name: cutile-annotate
description: Annotate cuTile kernels to allow LSP provide inlay hints and diagnostics

# trigger_keywords:
#     - 

version: 1.0
author: daisy
---

## First Steps: Enable cuTile LSP File Markers

The LSP will only process **explicitly enabled** files. Please add the enable marker at the **top** of the file:

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


## FAQ

- **LSP not starting**: Confirm that the current Python interpreter has `cutile-lsp` installed and is selected in VS Code.
- **No hints or diagnostics**: 
  - Confirm that the file has `# cutile-lsp: on` at the top, and the code contains `# cutile-lsp: start` and `# cutile-lsp: end` markers. 
  - Also ensure necessary imports are included in this region.
  - Don't import from other files. This extension is not and will not be capable of handling this.
  - Please write code like you are using C++, avoid using fancy pythonic dynamic tricks.
- **Incorrect hint positions**: Ensure `# cutile-lsp: start` is placed before the code block to be analyzed, and do not nest multiple `start/end` pairs.
- **Unresponsive hints**: Avoid importing heavy dependencies (like `torch`) between `# cutile-lsp: start` and `# cutile-lsp: end`, as the extension will execute code in between.

# <typecheck> Annotation Guide

This document explains how to properly write `<typecheck>` annotation blocks for cuTile LSP type checking and diagnostics.

---

## What is a `<typecheck>` Block?

`<typecheck>` is a special docstring annotation block that provides parameter type information for cuTile kernel functions. The LSP uses this to:
- Compile kernels and infer internal variable types
- Generate type diagnostics (shape mismatches, type errors, etc.)
- Display Inlay Hints (type annotations) in the editor

---

## Basic Syntax

```py
@ct.kernel
def your_kernel(param1, param2, param3: ConstInt):
    """
    <typecheck>
    Tensor((1024, 2048), dtype="float16")
    Tensor((512,), dtype="float32")
    64
    </typecheck>
    """
    # kernel implementation...
```

---

## Key Rules

### 1. Positional Correspondence

**Each line in `<typecheck>` must correspond one-to-one with function parameters**, both in order and count.

```py
@ct.kernel
def example_kernel(A, B, C, TILE_M: ConstInt, TILE_N: ConstInt):
    """
    <typecheck>
    Tensor((1024, 2048), dtype="float16")  # corresponds to A
    Tensor((2048, 512), dtype="float16")   # corresponds to B
    Tensor((1024, 512), dtype="float32")   # corresponds to C
    32                                      # corresponds to TILE_M
    64                                      # corresponds to TILE_N
    </typecheck>
    """
```

### 2. Parameter Type Formats

#### Tensor Parameters

Use the `Tensor(shape, dtype)` format:

```py
Tensor((batch, m, n), dtype="float16")    # 3D tensor
Tensor((1024, 2048), dtype="bfloat16")    # 2D tensor, fixed shape
Tensor((n,), dtype="float32")             # 1D tensor
```

Supported dtypes:
- `"float16"` / `"fp16"`
- `"bfloat16"` / `"bf16"`
- `"float32"` / `"fp32"`
- `"int32"`
- etc. (depending on cuTile support)

#### Constant Parameters (ConstInt / ConstBool)

Constant parameters can be written directly as values:

```py
@ct.kernel
def kernel_with_consts(data, tile_size: ConstInt, use_fp16: ConstBool):
    """
    <typecheck>
    Tensor((1024, 1024), dtype="float16")
    32           # value for tile_size
    True         # value for use_fp16
    </typecheck>
    """
```

### 3. Must be Valid Python Expressions

Each line must be a valid Python expression. The LSP performs syntax checking:

```py
# ✅ Correct
Tensor((1024, 2048), dtype="float16")
1024
1e-5

# ❌ Incorrect - syntax errors
Tensor(1024, 2048)  # missing shape tuple parentheses
float16             # undefined variable
```

### 4. Empty Lines

Empty lines are ignored, but keeping it clean is recommended:

```py
@ct.kernel
def example(X, Y):
    """
    <typecheck>
    Tensor((1024,), dtype="float32")

    Tensor((1024,), dtype="float32")
    </typecheck>
    """
```

---

## Common Errors

### Error 1: Parameter Count Mismatch

```py
@ct.kernel
def example(a, b, c):
    """
    <typecheck>
    Tensor((10,), dtype="float32")
    Tensor((10,), dtype="float32")
    # Error: missing type for the 3rd parameter c
    </typecheck>
    """
```

**Diagnostic**: `Parameter count mismatch in kernel 'example': function has 3 parameters, but <typecheck> has 2 args.`

### Error 2: Syntax Error

```py
@ct.kernel
def example(X):
    """
    <typecheck>
    Tensor(1024, 2048)  # Error: should be Tensor((1024, 2048), dtype="...")
    </typecheck>
    """
```

**Diagnostic**: `Invalid Python syntax in typecheck parameter: ...`

### Error 3: Missing `<typecheck>` Block

If a kernel lacks the `<typecheck>` annotation, the LSP reports:

**Diagnostic**: `<typecheck> is not annotated. Typecheck will not be performed.`

---

## Best Practices

1. **Maintain parameter order**: Always ensure the order in `<typecheck>` matches the function definition
2. **Use explicit dtypes**: Always specify Tensor data types to avoid ambiguity
3. **Validate shapes**: Ensure Tensor shapes match actual usage within the kernel
4. **Document**: Add parameter descriptions in the docstring to help other developers

```py
@ct.kernel
def well_documented_kernel(X, W, TILE: ConstInt):
    """
    <typecheck>
    Tensor((1024, 2048), dtype="float16")
    Tensor((2048,), dtype="float16")
    64
    </typecheck>

    Args:
        X: Input tensor (M, N).
        W: Weight tensor (N,).
        TILE: Tile size for processing.
    """
```
