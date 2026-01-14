# mcp-for-triton-kernel

An MCP (Model Context Protocol) server that supports converting PyTorch operations to Triton kernels.

## Installation

```bash
# Clone the project
git clone <repo-url>
cd mcp-for-triton-kernel

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .
```

## Cursor IDE Setup

Create a `.cursor/mcp.json` file and add the following content:

```json
{
  "mcpServers": {
    "triton-kernel": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/mcp-for-triton-kernel",
        "python",
        "-m",
        "mcp_for_triton_kernel.server"
      ]
    }
  }
}
```

Replace `/path/to/mcp-for-triton-kernel` with the actual project path.

## Available MCP Tools

### Information Tools

| Tool | Description |
|------|-------------|
| `get_overview` | Overview of the entire Triton kernel development process |
| `get_triton_syntax` | Reference for Triton syntax, tl functions, and constraints |
| `get_torch_op_info(op_name)` | Query PyTorch operation information |
| `get_kernel_template(pattern)` | Kernel templates (elementwise, reduction, matmul, fused) |

### Execution/Validation Tools

| Tool | Description |
|------|-------------|
| `check_gpu_status` | Check GPU availability and status |
| `run_triton_kernel(code, test_input_code)` | Execute Triton kernel |
| `validate_correctness(kernel_code, reference_code, test_input_code)` | Validate correctness |
| `benchmark_kernel(kernel_code, test_input_code, reference_code)` | Performance measurement |

## Usage Workflow

```
1. get_overview()           → Understand the overall process
2. get_torch_op_info("op")  → Check operation information to convert
3. get_triton_syntax()      → Reference Triton syntax
4. (LLM writes kernel code)
5. run_triton_kernel()      → Test execution
6. validate_correctness()   → Validate correctness
7. benchmark_kernel()       → Measure performance
```

## Supported Operations

- `vector_add`, `vector_mul` - Element-wise operations
- `relu`, `gelu` - Activation functions
- `softmax`, `layer_norm` - Normalization
- `matmul` - Matrix multiplication
- `dropout` - Dropout
- `attention` - Attention
- `cross_entropy` - Loss function
- `sum`, `mean` - Reduction operations

## Project Structure

```
mcp_for_triton_kernel/
├── __init__.py
├── __main__.py           # Support for python -m execution
├── server.py             # MCP server entry point
├── tools/
│   ├── info.py           # Information tools
│   └── execution.py      # Execution/validation tools
├── knowledge/
│   ├── overview.md       # Triton development guide
│   ├── torch_ops.json    # Torch ops information
│   └── triton_syntax.md  # Triton syntax reference
└── utils/
    └── runner.py         # Kernel execution utility
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- Triton >= 2.1.0
- CUDA GPU (for execution)
