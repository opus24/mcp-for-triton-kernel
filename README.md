# mcp-for-triton-kernel

An MCP (Model Context Protocol) server that supports converting PyTorch operations to Triton kernels with state-based workflow management.

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

## Workflow Status System

The system operates through 4 states:

```
start → write → evaluation → end
          ↑         ↓
          └─────────┘ (on failure or optimization)
```

| Status | Description |
|--------|-------------|
| `start` | Information gathering phase |
| `write` | Kernel code writing phase |
| `evaluation` | Validation and benchmarking phase |
| `end` | Final completion with best kernel selection |

**Important Rules:**
- Each tool can only be used in specific states
- Minimum 3 write iterations required before transitioning to `end`
- All tool calls are automatically logged

## Available MCP Tools

### Information Tools (Status: `start`)

| Tool | Description |
|------|-------------|
| `get_overview` | Overview of the entire Triton kernel development process |
| `get_triton_syntax` | Reference for Triton syntax, tl functions, and constraints |
| `get_torch_op_info(op_name)` | Query PyTorch operation information |
| `get_kernel_template(pattern)` | Kernel templates (elementwise, reduction, matmul, fused) |
| `check_gpu_status` | Check GPU availability and status |

### Workflow Tools (Status: `write`)

| Tool | Description |
|------|-------------|
| `get_current_status` | Check current workflow status (available in any state) |
| `check_context_usage` | Check current context usage and get warnings (available in any state) |
| `set_kernel_name(name)` | Set kernel name (used for log/kernel file naming) |
| `write_kernel_code(code)` | Write and save Triton kernel code (with optimization tips) |
| `force_transition_to_write` | Force transition from evaluation to write for optimization |

### Execution/Validation Tools (Status: `write`, `evaluation`)

| Tool | Description |
|------|-------------|
| `run_triton_kernel(code, test_input_code)` | Execute Triton kernel |
| `validate_correctness(kernel_code, reference_code, test_input_code)` | Validate correctness |
| `measure_kernel_time(kernel_code, test_input_code)` | Measure kernel execution time |
| `benchmark_kernel(kernel_code, test_input_code, reference_code)` | Performance measurement with comparison |

### Completion Tools (Status: `end`)

| Tool | Description |
|------|-------------|
| `get_best_kernel` | Get the fastest kernel and complete the session |

## Usage Workflow

```
1. [start] set_kernel_name("sub")   → Set kernel name (creates log file)
2. [start] get_overview()           → Understand the overall process
3. [start] get_torch_op_info("op")  → Check operation information
4. [start] get_triton_syntax()      → Reference Triton syntax
5. [start] check_gpu_status()       → Verify GPU availability
   → Status transitions to 'write'
   → Log file created: mcp_for_triton_kernel/log/triton_sub_log.md

6. [write] write_kernel_code(code)  → Write kernel code (v1)
   → Saved to: mcp_for_triton_kernel/kernel/triton_sub_kernel_v1.py
   → Status transitions to 'evaluation'

7. [evaluation] run_triton_kernel(test_input_code) → Test execution
8. [evaluation] validate_correctness(reference_code, test_input_code) → Validate
9. [evaluation] measure_kernel_time(test_input_code) → Measure performance

10. [evaluation] force_transition_to_write() → Continue optimization
    → Status transitions to 'write'

11. [write] write_kernel_code(code) → Write optimized kernel (v2)
12. ... repeat steps 7-11 at least 3 times ...

13. [evaluation] After validation passes + 3 writes completed
    → Status transitions to 'end'

14. [end] get_best_kernel()         → Get the fastest kernel
    → Final log written to: mcp_for_triton_kernel/log/triton_sub_log.md
```

## File Organization

All generated files are organized as follows:

```
mcp-for-triton-kernel/
└── mcp_for_triton_kernel/
    ├── log/
    │   ├── {session_id}_{timestamp}.jsonl    # JSON log
    │   └── triton_{name}_log.md              # Markdown log (auto-generated)
    ├── kernel/
    │   └── triton_{name}_kernel_v{version}.py  # Kernel code files
    └── tests/
        └── (test files can be added here)
```

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `mcp_for_triton_kernel/log/` | Log files (JSON and Markdown) |
| `mcp_for_triton_kernel/kernel/` | Triton kernel code files |
| `mcp_for_triton_kernel/tests/` | Test code files |

## Logging

All tool calls are automatically logged to the `log/` directory:
- **JSON Log**: `{session_id}_{timestamp}.jsonl` - Structured log for each tool call
- **Markdown Log**: `triton_{kernel_name}_log.md` - Human-readable development log (auto-generated)

Each log entry contains: timestamp, tool name, status before/after, arguments, result

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
mcp-for-triton-kernel/
└── mcp_for_triton_kernel/
    ├── __init__.py
    ├── __main__.py           # Support for python -m execution
    ├── server.py             # MCP server entry point
    ├── state.py              # State management and logging
    ├── log/                  # Log files (auto-created)
    ├── kernel/               # Kernel code files (auto-created)
    ├── tests/                # Test files (auto-created)
    ├── tools/
    │   ├── info.py           # Information tools
    │   ├── execution.py      # Execution/validation tools
    │   └── workflow.py       # Workflow management tools
    ├── knowledge/
    │   ├── overview.md       # Triton development guide
    │   ├── torch_ops.json    # Torch ops information
    │   └── triton_syntax.md  # Triton syntax reference
    └── utils/
        └── runner.py         # Kernel execution utility (file-based)
```

## Key Features

### 1. Automatic File Management
- **Kernel files**: Automatically saved to `kernel/` directory
- **Log files**: JSON and Markdown logs automatically created
- **File naming**: Based on kernel name and version

### 2. Triton Compatibility
- **File-based execution**: Solves Triton's `@jit` decorator limitation
- **Temporary modules**: Code is saved to files and imported dynamically
- **Automatic cleanup**: Temporary files are cleaned up after execution

### 3. Performance Optimization Tips
- Built-in optimization guidance in `write_kernel_code()`
- BLOCK_SIZE tuning recommendations
- Memory access optimization tips
- Autotune usage examples

### 4. Automatic Context Management
- **Context usage tracking**: Automatically tracks estimated token usage
- **Auto-summarization**: When context usage exceeds 70%, automatically creates `docs/summarization.md`
- **New session guidance**: Provides instructions to start a new session
- **Usage monitoring**: `check_context_usage()` tool to monitor current usage

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- Triton >= 2.1.0
- CUDA GPU (for execution)
