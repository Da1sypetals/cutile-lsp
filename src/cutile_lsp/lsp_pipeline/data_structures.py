import inspect
from typing import Callable


class Kernel:
    def __init__(self, pyfunc: Callable, args_str: list[str]):
        self.pyfunc = pyfunc
        self.args_str = args_str
        self.name = pyfunc.__name__
        self.source, self.start_line = inspect.getsourcelines(pyfunc)

    @property
    def launch_func_name(self):
        return f"launc_{self.name}"
