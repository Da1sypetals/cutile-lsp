import inspect
import linecache
import os
import sys
import types

import cuda.tile


def create_module_from_source(module_name: str, source_code: str, filename: str = None):
    """创建模块并支持 inspect.getsource()"""

    if filename is None:
        filename = "<string>"

    # 1. 创建模块
    module = types.ModuleType(module_name)
    module.__file__ = filename

    # 2. 将源码注入 linecache（关键！）
    linecache.cache[filename] = (
        len(source_code),
        None,  # mtime
        source_code.splitlines(keepends=True),
        filename,
    )

    # 3. 执行代码
    exec(compile(source_code, filename, "exec"), module.__dict__)

    # 4. 注册到 sys.modules
    sys.modules[module_name] = module

    return module


def load_kernels_to_pyfunc(code: str, module_name: str = "custom_module"):
    # Load the module dynamically
    module = create_module_from_source(module_name, code)

    python_funcs = []

    for name in module.__dir__():
        item = getattr(module, name)

        if isinstance(item, cuda.tile._execution.kernel):
            pyfunc = item._pyfunc
            python_funcs.append(pyfunc)

    return python_funcs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument("path", type=str, help="Path to the file")
    args = parser.parse_args()

    funcs = load_kernels_to_pyfunc(args.path)
    for func in funcs:
        print(func.__name__)
        print(f"# of line: {len(inspect.getsource(func).splitlines())}")
