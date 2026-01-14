"""Execution and validation tools for Triton kernels."""

import json
from typing import Optional, Any

from fastmcp import FastMCP

from ..utils.runner import TritonRunner


# Global runner instance (lazy initialization)
_runner: Optional[TritonRunner] = None


def get_runner() -> TritonRunner:
    """Get or create the global TritonRunner instance."""
    global _runner
    if _runner is None:
        _runner = TritonRunner()
    return _runner


def register_execution_tools(mcp: FastMCP) -> None:
    """Register execution and validation tools to the MCP server."""

    @mcp.tool()
    def check_gpu_status() -> str:
        """
        GPU 상태를 확인합니다.
        
        Triton 커널 실행 전에 GPU 가용성을 확인하세요.
        
        Returns:
            GPU 상태 정보 (가용성, 디바이스명, 메모리 등)
        """
        runner = get_runner()
        
        if not runner.gpu_available:
            return """⚠️ GPU를 사용할 수 없습니다.

Triton 커널 실행에는 CUDA GPU가 필요합니다.
현재 환경에서는 코드 작성만 가능하고, 실행은 GPU 환경에서 해야 합니다.
"""
        
        try:
            import torch
            
            gpu_info = {
                "available": True,
                "device_name": runner.gpu_name,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                "memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
                "max_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
            }
            
            return f"""✅ GPU 사용 가능

디바이스: {gpu_info['device_name']}
디바이스 수: {gpu_info['device_count']}
현재 디바이스: {gpu_info['current_device']}
할당된 메모리: {gpu_info['memory_allocated']}
예약된 메모리: {gpu_info['memory_reserved']}
총 메모리: {gpu_info['max_memory']}
"""
        except Exception as e:
            return f"GPU 상태 확인 중 오류: {e}"

    @mcp.tool()
    def run_triton_kernel(
        code: str,
        test_input_code: str,
        entry_function: str = "solve",
    ) -> str:
        """
        Triton 커널 코드를 실행합니다.
        
        Args:
            code: Triton 커널과 solve 함수가 포함된 전체 Python 코드
            test_input_code: 테스트 입력을 생성하는 Python 코드
                            변수 'args'와 'kwargs'를 정의해야 함
                            예: "args = [torch.randn(1024, device='cuda')]"
            entry_function: 호출할 함수 이름 (기본값: "solve")
        
        Returns:
            실행 결과 (성공 시 출력 정보, 실패 시 에러 메시지)
        
        Example:
            code = '''
            import torch
            import triton
            import triton.language as tl
            
            @triton.jit
            def add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK: tl.constexpr):
                idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
                mask = idx < N
                a = tl.load(a_ptr + idx, mask=mask)
                b = tl.load(b_ptr + idx, mask=mask)
                tl.store(c_ptr + idx, a + b, mask=mask)
            
            def solve(a, b):
                c = torch.empty_like(a)
                N = a.numel()
                grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
                add_kernel[grid](a, b, c, N, BLOCK=256)
                return c
            '''
            
            test_input_code = '''
            import torch
            a = torch.randn(1024, device='cuda')
            b = torch.randn(1024, device='cuda')
            args = [a, b]
            kwargs = {}
            '''
        """
        runner = get_runner()
        
        if not runner.gpu_available:
            return """❌ GPU 없음

GPU가 없어서 커널을 실행할 수 없습니다.
GPU 환경에서 다시 시도하세요.

코드가 문법적으로 올바른지는 확인할 수 있습니다:
""" + _syntax_check(code)
        
        # Parse test inputs
        try:
            input_namespace = {}
            exec(test_input_code, input_namespace)
            args = input_namespace.get("args", [])
            kwargs = input_namespace.get("kwargs", {})
        except Exception as e:
            return f"""❌ 테스트 입력 코드 오류

{type(e).__name__}: {e}

test_input_code는 'args'와 'kwargs' 변수를 정의해야 합니다.
예:
```python
import torch
a = torch.randn(1024, device='cuda')
args = [a]
kwargs = {{}}
```
"""
        
        # Run kernel
        result = runner.execute_code(code, entry_function, args, kwargs)
        
        if result.success:
            output_info = _describe_output(result.output)
            return f"""✅ 실행 성공

실행 시간: {result.execution_time_ms:.3f} ms

출력:
{output_info}

stdout:
{result.stdout if result.stdout else "(없음)"}
"""
        else:
            return f"""❌ 실행 실패

에러 타입: {result.error_type}
에러 메시지: {result.error}

stderr:
{result.stderr if result.stderr else "(없음)"}
"""

    @mcp.tool()
    def validate_correctness(
        kernel_code: str,
        reference_code: str,
        test_input_code: str,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> str:
        """
        Triton 커널 출력을 PyTorch 참조 구현과 비교하여 정확성을 검증합니다.
        
        Args:
            kernel_code: Triton 커널 코드 (solve 함수 포함)
            reference_code: PyTorch 참조 구현 코드 (reference 함수 포함)
            test_input_code: 테스트 입력 생성 코드 (args, kwargs 정의)
            rtol: 상대 허용 오차 (기본값: 1e-5)
            atol: 절대 허용 오차 (기본값: 1e-8)
        
        Returns:
            검증 결과 (통과/실패, 차이 통계)
        
        Example:
            kernel_code = '''
            # ... triton kernel code with solve() function ...
            '''
            
            reference_code = '''
            import torch
            def reference(a, b):
                return a + b
            '''
            
            test_input_code = '''
            import torch
            a = torch.randn(1024, device='cuda')
            b = torch.randn(1024, device='cuda')
            args = [a, b]
            kwargs = {}
            '''
        """
        runner = get_runner()
        
        if not runner.gpu_available:
            return "❌ GPU가 없어서 검증을 수행할 수 없습니다."
        
        # Parse test inputs
        try:
            input_namespace = {}
            exec(test_input_code, input_namespace)
            args = input_namespace.get("args", [])
            kwargs = input_namespace.get("kwargs", {})
        except Exception as e:
            return f"❌ 테스트 입력 코드 오류: {e}"
        
        # Run triton kernel
        triton_result = runner.execute_code(kernel_code, "solve", args, kwargs)
        if not triton_result.success:
            return f"""❌ Triton 커널 실행 실패

에러: {triton_result.error}
{triton_result.stderr}
"""
        
        # Run reference implementation
        ref_result = runner.execute_code(reference_code, "reference", args, kwargs)
        if not ref_result.success:
            return f"""❌ 참조 구현 실행 실패

에러: {ref_result.error}
{ref_result.stderr}
"""
        
        # Validate
        validation = runner.validate_correctness(
            triton_result.output,
            ref_result.output,
            rtol=rtol,
            atol=atol,
        )
        
        if validation.error:
            return f"❌ 검증 중 오류: {validation.error}"
        
        if validation.passed:
            return f"""✅ 검증 통과

최대 차이: {validation.max_diff:.2e}
평균 차이: {validation.mean_diff:.2e}
전체 요소: {validation.total_elements:,}
허용 오차: rtol={rtol}, atol={atol}
"""
        else:
            return f"""❌ 검증 실패

최대 차이: {validation.max_diff:.2e}
평균 차이: {validation.mean_diff:.2e}
불일치 요소: {validation.num_mismatches:,} / {validation.total_elements:,}
허용 오차: rtol={rtol}, atol={atol}

팁: fp16 사용 시 rtol=1e-3, atol=1e-3 정도가 적절합니다.
"""

    @mcp.tool()
    def benchmark_kernel(
        kernel_code: str,
        test_input_code: str,
        reference_code: Optional[str] = None,
        warmup: int = 25,
        rep: int = 100,
    ) -> str:
        """
        Triton 커널의 성능을 측정합니다.
        
        Args:
            kernel_code: Triton 커널 코드 (solve 함수 포함)
            test_input_code: 테스트 입력 생성 코드
            reference_code: (선택) 비교할 PyTorch 참조 구현
            warmup: 워밍업 실행 횟수 (기본값: 25)
            rep: 측정 실행 횟수 (기본값: 100)
        
        Returns:
            성능 측정 결과 (평균, 표준편차, 최소/최대 시간)
        """
        runner = get_runner()
        
        if not runner.gpu_available:
            return "❌ GPU가 없어서 벤치마크를 수행할 수 없습니다."
        
        # Parse test inputs
        try:
            input_namespace = {}
            exec(test_input_code, input_namespace)
            args = input_namespace.get("args", [])
            kwargs = input_namespace.get("kwargs", {})
        except Exception as e:
            return f"❌ 테스트 입력 코드 오류: {e}"
        
        # Get reference function if provided
        reference_fn = None
        if reference_code:
            try:
                ref_namespace = {}
                exec(reference_code, ref_namespace)
                reference_fn = ref_namespace.get("reference")
            except Exception as e:
                return f"❌ 참조 코드 오류: {e}"
        
        # Run benchmark
        result = runner.benchmark(
            kernel_code,
            "solve",
            args,
            kwargs,
            warmup=warmup,
            rep=rep,
            reference_fn=reference_fn,
        )
        
        if not result.success:
            return f"❌ 벤치마크 실패: {result.error}"
        
        output = f"""📊 벤치마크 결과

실행 횟수: {result.num_runs}
평균: {result.mean_ms:.4f} ms
표준편차: {result.std_ms:.4f} ms
최소: {result.min_ms:.4f} ms
최대: {result.max_ms:.4f} ms
"""
        
        if result.comparison:
            speedup = result.comparison.get("speedup", 0)
            ref_mean = result.comparison.get("reference_mean_ms", 0)
            
            if speedup >= 1:
                comparison_text = f"🚀 Triton이 {speedup:.2f}x 빠름"
            else:
                comparison_text = f"⚠️ PyTorch가 {1/speedup:.2f}x 빠름"
            
            output += f"""
PyTorch 참조: {ref_mean:.4f} ms
{comparison_text}
"""
        
        return output


def _syntax_check(code: str) -> str:
    """Check Python syntax without executing."""
    try:
        compile(code, "<string>", "exec")
        return "✅ 문법 검사 통과"
    except SyntaxError as e:
        return f"❌ 문법 오류 (라인 {e.lineno}): {e.msg}"


def _describe_output(output: Any) -> str:
    """Describe the output tensor/value."""
    try:
        import torch
        
        if isinstance(output, torch.Tensor):
            return f"""Tensor:
  shape: {list(output.shape)}
  dtype: {output.dtype}
  device: {output.device}
  min: {output.min().item():.6f}
  max: {output.max().item():.6f}
  mean: {output.mean().item():.6f}"""
        elif isinstance(output, (list, tuple)):
            return f"{type(output).__name__} with {len(output)} elements"
        else:
            return str(output)[:500]
    except Exception as e:
        return f"(출력 설명 불가: {e})"

