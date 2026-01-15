"""Test and validate the Triton add kernel."""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from kernel.triton_add_kernel_v1 import solve


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation."""
    return a + b


# Test correctness
print("=" * 60)
print("정확성 검증 (Correctness Validation)")
print("=" * 60)

test_cases = [
    (torch.randn(1024, device='cuda'), torch.randn(1024, device='cuda')),
    (torch.randn(10000, device='cuda'), torch.randn(10000, device='cuda')),
    (torch.randn(3, 4, 5, device='cuda'), torch.randn(3, 4, 5, device='cuda')),
]

all_passed = True
for i, (a, b) in enumerate(test_cases, 1):
    triton_result = solve(a, b)
    torch_result = reference(a, b)
    
    max_diff = (triton_result - torch_result).abs().max().item()
    is_close = torch.allclose(triton_result, torch_result, rtol=1e-5, atol=1e-8)
    
    print(f"\n테스트 케이스 {i}: shape={a.shape}")
    print(f"  최대 차이: {max_diff:.2e}")
    print(f"  검증 통과: {'✅' if is_close else '❌'}")
    
    if not is_close:
        all_passed = False

print("\n" + "=" * 60)
print(f"전체 검증 결과: {'✅ 통과' if all_passed else '❌ 실패'}")
print("=" * 60)

# Benchmark
print("\n" + "=" * 60)
print("성능 측정 (Performance Benchmark)")
print("=" * 60)

import time

a = torch.randn(1000000, device='cuda')
b = torch.randn(1000000, device='cuda')

# Warmup
for _ in range(25):
    _ = solve(a, b)
    _ = reference(a, b)
torch.cuda.synchronize()

# Benchmark Triton
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(100):
    _ = solve(a, b)
end.record()
torch.cuda.synchronize()
triton_time = start.elapsed_time(end) / 100

# Benchmark PyTorch
torch.cuda.synchronize()
start.record()
for _ in range(100):
    _ = reference(a, b)
end.record()
torch.cuda.synchronize()
torch_time = start.elapsed_time(end) / 100

print(f"\n입력 크기: {a.shape}")
print(f"Triton 평균 시간: {triton_time:.4f} ms")
print(f"PyTorch 평균 시간: {torch_time:.4f} ms")
if torch_time > 0:
    speedup = torch_time / triton_time
    if speedup >= 1:
        print(f"속도 향상: {speedup:.2f}x 🚀")
    else:
        print(f"속도: {1/speedup:.2f}x (PyTorch가 더 빠름) ⚠️")


