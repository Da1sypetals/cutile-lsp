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

---

# cuTile LSP `<typecheck>` 注释指南

> [English Version](#typecheck-annotation-guide)

本文档详细说明如何正确编写 `<typecheck>` 注释块，以便 cuTile LSP 能够进行准确的类型检查和诊断。

---

## 什么是 `<typecheck>` 块？

`<typecheck>` 是一个特殊的 docstring 注释块，用于为 cuTile kernel 函数提供参数类型信息。LSP 会根据这些信息：
- 编译 kernel 并推断内部变量的类型
- 生成类型诊断（如形状不匹配、类型错误等）
- 在编辑器中显示 Inlay Hints（类型提示）

---

## 基本语法

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
    # kernel 实现...
```

---

## 关键规则

### 1. 位置对应规则

**`<typecheck>` 中的每一行必须与函数参数一一对应**，顺序和数量都必须完全一致。

```py
@ct.kernel
def example_kernel(A, B, C, TILE_M: ConstInt, TILE_N: ConstInt):
    """
    <typecheck>
    Tensor((1024, 2048), dtype="float16")  # 对应参数 A
    Tensor((2048, 512), dtype="float16")   # 对应参数 B
    Tensor((1024, 512), dtype="float32")   # 对应参数 C
    32                                      # 对应参数 TILE_M
    64                                      # 对应参数 TILE_N
    </typecheck>
    """
```

### 2. 参数类型格式

#### Tensor 参数

使用 `Tensor(shape, dtype)` 格式：

```py
Tensor((batch, m, n), dtype="float16")    # 3D 张量
Tensor((1024, 2048), dtype="bfloat16")    # 2D 张量，固定形状
Tensor((n,), dtype="float32")             # 1D 张量
```

支持的 dtype：
- `"float16"` / `"fp16"`
- `"bfloat16"` / `"bf16"`
- `"float32"` / `"fp32"`
- `"int32"`
- 等等（取决于 cuTile 支持的数据类型）

#### 常量参数 (ConstInt / ConstBool 等)

常量参数直接写值即可：

```py
@ct.kernel
def kernel_with_consts(data, tile_size: ConstInt, use_fp16: ConstBool):
    """
    <typecheck>
    Tensor((1024, 1024), dtype="float16")
    32           # tile_size 的值
    True         # use_fp16 的值
    </typecheck>
    """
```

### 3. 必须是有效的 Python 表达式

每一行必须是合法的 Python 表达式，LSP 会对其进行语法检查：

```py
# ✅ 正确
Tensor((1024, 2048), dtype="float16")
1024
1e-5

# ❌ 错误 - 语法不完整
Tensor(1024, 2048)  # 缺少 shape 元组括号
float16             # 未定义的变量
```

### 4. 空行处理

空行会被忽略，但建议保持简洁：

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

## 完整示例

### 示例 1: Layer Normalization Kernel

```py
@ct.kernel
def layer_norm_fwd(X, W, B, Y, Mean, Rstd, eps, TILE_N: ConstInt):
    """
    <typecheck>
    Tensor((1024, 2048), dtype="float16")
    Tensor((2048,), dtype="float16")
    Tensor((2048,), dtype="float16")
    Tensor((1024, 2048), dtype="float16")
    Tensor((1024,), dtype="float32")
    Tensor((1024,), dtype="float32")
    1e-5
    1024
    </typecheck>
    """
    # kernel 实现...
```

### 示例 2: Batch Matrix Multiplication

```py
@ct.kernel
def batch_matmul_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    """
    <typecheck>
    Tensor((4, 512, 256), dtype="bfloat16")
    Tensor((4, 256, 1024), dtype="bfloat16")
    Tensor((4, 512, 1024), dtype="bfloat16")
    32
    64
    128
    </typecheck>
    """
    # kernel 实现...
```

### 示例 3: Split-K GeMV with Silu

```py
@ct.kernel
def gemv_silu_mul_split_k_kernel(
    A,
    B1,
    B2,
    C,
    f32acc,
    COUNTS,
    tn: ConstInt,
    tk: ConstInt,
    SPLIT_K: ConstInt,
    approx: ct.Constant[bool],
):
    """
    <typecheck>
    Tensor((1, 1536), dtype="float16")
    Tensor((8960, 1536), dtype="float16")
    Tensor((8960, 1536), dtype="float16")
    Tensor((1, 8960), dtype="float16")
    Tensor((2, 8960), dtype="float32")
    Tensor((280,), dtype="int32")
    32
    64
    8
    True
    </typecheck>
    """
    # kernel 实现...
```

---

## 常见错误

### 错误 1: 参数数量不匹配

```py
@ct.kernel
def example(a, b, c):
    """
    <typecheck>
    Tensor((10,), dtype="float32")
    Tensor((10,), dtype="float32")
    # 错误：缺少第3个参数 c 的类型
    </typecheck>
    """
```

**诊断信息**: `Parameter count mismatch in kernel 'example': function has 3 parameters, but <typecheck> has 2 args.`

### 错误 2: 语法错误

```py
@ct.kernel
def example(X):
    """
    <typecheck>
    Tensor(1024, 2048)  # 错误：应该是 Tensor((1024, 2048), dtype="...")
    </typecheck>
    """
```

**诊断信息**: `Invalid Python syntax in typecheck parameter: ...`

### 错误 3: 缺少 `<typecheck>` 块

如果 kernel 没有 `<typecheck>` 注释，LSP 会报告：

**诊断信息**: `<typecheck> is not annotated. Typecheck will not be performed.`

---

## 最佳实践

1. **保持参数顺序一致**: 始终确保 `<typecheck>` 中的参数顺序与函数定义完全一致
2. **使用明确的 dtype**: 始终指定 Tensor 的数据类型，避免歧义
3. **验证形状**: 确保 Tensor 的形状与 kernel 内部的实际使用一致
4. **文档化**: 在 docstring 中添加参数说明，帮助其他开发者理解

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

