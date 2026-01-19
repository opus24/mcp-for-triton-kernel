# CUDA와 Triton 코드 실행 가이드

## 개요

이 프로젝트에서 CUDA와 Triton 코드를 실제로 실행하는 방법을 단계별로 설명합니다.

## 1. 환경 확인 및 준비

### 1.1 의존성 설치

```bash
cd /root/mcp-for-triton-kernel

# uv를 사용하는 경우
uv sync

# pip를 사용하는 경우
pip install -e .

# numpy가 누락된 경우 (PyTorch가 내부적으로 사용)
pip install numpy
# 또는
uv pip install numpy
```

**중요**: PyTorch가 내부적으로 numpy를 사용하므로, numpy가 없으면 경고가 발생합니다. 실행은 되지만 경고를 없애려면 numpy를 설치하세요.

### 1.2 GPU 확인

```bash
# GPU 상태 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## 2. Triton 커널 실행 방법

### 방법 1: 테스트 파일 직접 실행 (가장 간단)

기존 테스트 파일을 수정하여 실행:

```bash
cd /root/mcp-for-triton-kernel/mcp_for_triton_kernel/tests

# 테스트 스크립트 생성
cat > test_add_direct.py << 'EOF'
import sys
sys.path.insert(0, '/root/mcp-for-triton-kernel')

from kernel.triton_add_kernel_v1 import solve
from test_add_kernel import reference, test_case_1
import torch

# 테스트 실행
print("Creating test inputs...")
a, b = test_case_1()
print(f"Input shapes: a={a.shape}, b={b.shape}")

print("Running Triton kernel...")
triton_result = solve(a, b)

print("Running PyTorch reference...")
reference_result = reference(a, b)

# 검증
print(f"\nResults:")
print(f"Triton result shape: {triton_result.shape}")
print(f"Reference result shape: {reference_result.shape}")
print(f"Max difference: {(triton_result - reference_result).abs().max().item():.6f}")
print(f"Mean difference: {(triton_result - reference_result).abs().mean().item():.6f}")
print(f"Are close (rtol=1e-5): {torch.allclose(triton_result, reference_result, rtol=1e-5)}")
print(f"\nFirst 5 values:")
print(f"Triton:  {triton_result[:5]}")
print(f"Reference: {reference_result[:5]}")
EOF

python test_add_direct.py
```

### 방법 2: Python 스크립트로 직접 실행

커널 파일을 직접 import하여 실행:

```bash
python << 'EOF'
import sys
sys.path.insert(0, '/root/mcp-for-triton-kernel')
from mcp_for_triton_kernel.kernel.triton_add_kernel_v1 import solve
import torch

# 테스트 입력 생성
print("Creating test inputs...")
a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')

# 실행
print("Running kernel...")
result = solve(a, b)

print(f'Result shape: {result.shape}')
print(f'First 5 values: {result[:5]}')
print(f'Sum: {result.sum().item():.4f}')
EOF
```

### 방법 3: TritonRunner 사용 (프로그래밍 방식)

`utils/runner.py`의 `TritonRunner` 클래스를 사용:

```bash
python << 'EOF'
import sys
sys.path.insert(0, '/root/mcp-for-triton-kernel')
from mcp_for_triton_kernel.utils.runner import TritonRunner
import torch

runner = TritonRunner()

# 파일에서 커널 실행
print("Loading kernel from file...")
result = runner.execute_from_file(
    '/root/mcp-for-triton-kernel/mcp_for_triton_kernel/kernel/triton_add_kernel_v1.py',
    entry_function='solve',
    args=[torch.randn(1024, device='cuda'), torch.randn(1024, device='cuda')]
)

print(f'Success: {result.success}')
if result.success:
    print(f'Output shape: {result.output.shape}')
    print(f'Execution time: {result.execution_time_ms:.3f} ms')
    print(f'First 5 values: {result.output[:5]}')
else:
    print(f'Error: {result.error}')
    if result.stderr:
        print(f'Stderr: {result.stderr}')
EOF
```

### 방법 4: 정확성 검증 포함 실행

```bash
python << 'EOF'
import sys
sys.path.insert(0, '/root/mcp-for-triton-kernel')
from mcp_for_triton_kernel.utils.runner import TritonRunner
from mcp_for_triton_kernel.tests.test_add_kernel import reference
import torch

runner = TritonRunner()

# 테스트 입력
a = torch.randn(10000, device='cuda')
b = torch.randn(10000, device='cuda')

# Triton 커널 실행
print("Running Triton kernel...")
triton_result = runner.execute_from_file(
    '/root/mcp-for-triton-kernel/mcp_for_triton_kernel/kernel/triton_add_kernel_v1.py',
    entry_function='solve',
    args=[a, b]
)

if not triton_result.success:
    print(f"Error: {triton_result.error}")
    exit(1)

# PyTorch 참조 실행
print("Running PyTorch reference...")
reference_result = reference(a, b)

# 검증
validation = runner.validate_correctness(
    triton_result.output,
    reference_result,
    rtol=1e-5,
    atol=1e-8
)

print(f"\nValidation Results:")
print(f"  Passed: {validation.passed}")
print(f"  Max difference: {validation.max_diff:.6f}")
print(f"  Mean difference: {validation.mean_diff:.6f}")
print(f"  Mismatches: {validation.num_mismatches}/{validation.total_elements}")
print(f"  Execution time: {triton_result.execution_time_ms:.3f} ms")
EOF
```

## 3. 벤치마크 및 성능 측정

### 3.1 성능 측정 스크립트

```bash
python << 'EOF'
import sys
sys.path.insert(0, '/root/mcp-for-triton-kernel')

from mcp_for_triton_kernel.utils.runner import TritonRunner
from mcp_for_triton_kernel.tests.test_add_kernel import reference
import torch

runner = TritonRunner()

# 테스트 입력
print("Preparing test inputs...")
a = torch.randn(10000, device='cuda')
b = torch.randn(10000, device='cuda')

# Triton 커널 벤치마크
print("Benchmarking Triton kernel...")
triton_result = runner.benchmark_from_file(
    '/root/mcp-for-triton-kernel/mcp_for_triton_kernel/kernel/triton_add_kernel_v1.py',
    entry_function='solve',
    args=[a, b],
    warmup=25,
    rep=100,
    reference_fn=lambda a, b: reference(a, b)
)

if triton_result.success:
    print(f"\nTriton Kernel Performance:")
    print(f"  Mean: {triton_result.mean_ms:.3f} ms")
    print(f"  Std:  {triton_result.std_ms:.3f} ms")
    print(f"  Min:  {triton_result.min_ms:.3f} ms")
    print(f"  Max:  {triton_result.max_ms:.3f} ms")

    if triton_result.comparison:
        print(f"\nComparison with PyTorch:")
        print(f"  PyTorch mean: {triton_result.comparison['reference_mean_ms']:.3f} ms")
        print(f"  Speedup: {triton_result.comparison['speedup']:.2f}x")
else:
    print(f"Error: {triton_result.error}")
EOF
```

## 4. CUDA 코드 실행 (참고용)

프로젝트에는 직접적인 CUDA `.cu` 파일은 없지만, `3rd_party/triton_tutorial/` 디렉토리에 Jupyter 노트북이 있습니다.

### 4.1 Jupyter 노트북 실행

```bash
# Jupyter 설치 (필요시)
pip install jupyter

# 노트북 실행
cd /root/mcp-for-triton-kernel/3rd_party/triton_tutorial
jupyter notebook 02_vector_addition.ipynb
```

### 4.2 노트북을 Python 스크립트로 변환

```bash
# 노트북을 스크립트로 변환 (jupyter nbconvert 필요)
cd /root/mcp-for-triton-kernel/3rd_party/triton_tutorial
jupyter nbconvert --to script 02_vector_addition.ipynb

# 변환된 스크립트 실행
python 02_vector_addition.py
```

## 5. 실행 순서 요약

### 빠른 테스트 (1분)

```bash
# 1. 의존성 확인
cd /root/mcp-for-triton-kernel
pip install numpy  # 경고 제거용

# 2. 간단한 실행
python << 'EOF'
import sys
sys.path.insert(0, '/root/mcp-for-triton-kernel')
from mcp_for_triton_kernel.kernel.triton_add_kernel_v1 import solve
import torch
a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')
result = solve(a, b)
print(f"Success! Result shape: {result.shape}")
EOF
```

### 전체 테스트 (5분)

```bash
# 1. 환경 준비
cd /root/mcp-for-triton-kernel
pip install numpy

# 2. GPU 확인
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 3. 커널 실행 및 검증 (방법 4 사용)
# 위의 "방법 4: 정확성 검증 포함 실행" 스크립트 실행

# 4. 성능 측정 (방법 3.1 사용)
# 위의 "3.1 성능 측정 스크립트" 실행
```

## 6. 주의사항

1. **GPU 필수**: 모든 텐서는 `device='cuda'`로 생성해야 합니다. GPU가 없으면 Triton 커널을 실행할 수 없습니다.

2. **numpy 경고**: PyTorch가 내부적으로 numpy를 사용하므로, numpy가 없으면 경고가 발생합니다. 실행은 되지만 경고를 없애려면 `pip install numpy`를 실행하세요.

3. **커널 함수**: 커널 파일은 `solve` 함수를 포함해야 합니다 (또는 `entry_function` 파라미터로 지정).

4. **경로 설정**: `sys.path.insert(0, '/root/mcp-for-triton-kernel')`를 사용하여 모듈을 import할 수 있도록 해야 합니다.

## 7. 문제 해결

### 문제: "No module named 'numpy'"

```bash
pip install numpy
```

### 문제: "CUDA out of memory"

- 더 작은 텐서 크기 사용
- GPU 메모리 정리: `torch.cuda.empty_cache()`

### 문제: "Function 'solve' not found"

- 커널 파일에 `solve` 함수가 있는지 확인
- `entry_function` 파라미터로 다른 함수명 지정

### 문제: "GPU is not available"

- GPU가 실제로 있는지 확인: `nvidia-smi`
- PyTorch가 CUDA를 지원하는 버전인지 확인: `python -c "import torch; print(torch.cuda.is_available())"`
