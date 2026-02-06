# CuTile LSP

Language Server Protocol (LSP) support for CuTile kernels.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
from cutile_lsp import layer_norm_fwd

# Your kernel usage here
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
ruff check src/ tests/

# Type check
mypy src/
```

## Project Structure

```
cutile-lsp/
├── src/
│   └── cutile_lsp/
│       ├── __init__.py
│       ├── kernel.py
│       └── lsp_pipeline/
│           ├── __init__.py
│           ├── assemble_code.py
│           ├── data_structures.py
│           ├── extract_code.py
│           ├── get_kernels.py
│           ├── HEAD.py
│           ├── loggings.py
│           ├── parse_args.py
│           └── pipeline.py
├── tests/
│   └── test_kernel.py
├── pyproject.toml
└── README.md
```
