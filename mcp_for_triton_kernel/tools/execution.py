"""Execution and validation tools for Triton kernels."""

from typing import Any, Optional

from fastmcp import FastMCP

from ..state import Status, get_state_manager, log_tool_call
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
    @log_tool_call(allowed_statuses=[Status.START])
    def check_gpu_status() -> str:
        """
        GPU 상태를 확인합니다.

        이 도구는 'start' 상태에서만 사용할 수 있습니다.
        Triton 커널 실행 전에 GPU 가용성을 확인하세요.

        Returns:
            GPU 상태 정보 (가용성, 디바이스명, 메모리 등)
        """
        state = get_state_manager()
        state.mark_info_collected("check_gpu_status")

        runner = get_runner()

        status_hint = ""
        if state.can_transition_to_write():
            status_hint = "\n\n✅ 모든 정보 수집 완료! 상태가 'write'로 전환되었습니다."
        else:
            missing = [t for t, done in state.info_collected.items() if not done]
            status_hint = f"\n\n📋 아직 수집이 필요한 정보: {', '.join(missing)}"

        if not runner.gpu_available:
            return f"""⚠️ GPU를 사용할 수 없습니다.

Triton 커널 실행에는 CUDA GPU가 필요합니다.
현재 환경에서는 코드 작성만 가능하고, 실행은 GPU 환경에서 해야 합니다.
{status_hint}"""

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
{status_hint}"""
        except Exception as e:
            return f"GPU 상태 확인 중 오류: {e}{status_hint}"

    @mcp.tool()
    @log_tool_call(allowed_statuses=[Status.WRITE, Status.EVALUATION])
    def run_triton_kernel(
        test_input_code: str,
        entry_function: str = "solve",
    ) -> str:
        """
        현재 버전의 Triton 커널을 실행합니다.

        이 도구는 'write' 또는 'evaluation' 상태에서 사용할 수 있습니다.
        kernel/ 디렉토리에 저장된 최신 커널 파일을 실행합니다.

        Args:
            test_input_code: 테스트 입력을 생성하는 Python 코드
                            변수 'args'와 'kwargs'를 정의해야 함
                            예: "args = [torch.randn(1024, device='cuda')]"
            entry_function: 호출할 함수 이름 (기본값: "solve")

        Returns:
            실행 결과 (성공 시 출력 정보, 실패 시 에러 메시지)
        """
        state = get_state_manager()
        runner = get_runner()

        if not runner.gpu_available:
            return """❌ GPU 없음

GPU가 없어서 커널을 실행할 수 없습니다.
GPU 환경에서 다시 시도하세요.
"""

        # 현재 커널 버전 가져오기
        latest_kernel = state.get_latest_kernel()
        if latest_kernel is None:
            return "❌ 커널이 없습니다. 먼저 write_kernel_code()로 커널을 작성하세요."

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

        # 파일에서 커널 실행
        result = runner.execute_from_file(
            latest_kernel.kernel_file,
            entry_function,
            args,
            kwargs,
        )

        if result.success:
            output_info = _describe_output(result.output)
            return f"""✅ 실행 성공

커널 버전: v{latest_kernel.version}
커널 파일: {latest_kernel.kernel_file}
실행 시간: {result.execution_time_ms:.3f} ms

출력:
{output_info}

stdout:
{result.stdout if result.stdout else "(없음)"}
"""
        else:
            return f"""❌ 실행 실패

커널 버전: v{latest_kernel.version}
커널 파일: {latest_kernel.kernel_file}

에러 타입: {result.error_type}
에러 메시지: {result.error}

stderr:
{result.stderr if result.stderr else "(없음)"}
"""

    @mcp.tool()
    @log_tool_call(allowed_statuses=[Status.WRITE, Status.EVALUATION])
    def validate_correctness(
        reference_code: str,
        test_input_code: str,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> str:
        """
        현재 버전의 Triton 커널 출력을 PyTorch 참조 구현과 비교하여 정확성을 검증합니다.

        이 도구는 'write' 또는 'evaluation' 상태에서 사용할 수 있습니다.
        'evaluation' 상태에서 검증 통과 시 상태 전환이 발생할 수 있습니다.

        Args:
            reference_code: PyTorch 참조 구현 코드 (reference 함수 포함)
            test_input_code: 테스트 입력 생성 코드 (args, kwargs 정의)
            rtol: 상대 허용 오차 (기본값: 1e-5)
            atol: 절대 허용 오차 (기본값: 1e-8)

        Returns:
            검증 결과 (통과/실패, 차이 통계)
        """
        state = get_state_manager()
        runner = get_runner()

        if not runner.gpu_available:
            return "❌ GPU가 없어서 검증을 수행할 수 없습니다."

        # 현재 커널 버전 가져오기
        latest_kernel = state.get_latest_kernel()
        if latest_kernel is None:
            return "❌ 커널이 없습니다. 먼저 write_kernel_code()로 커널을 작성하세요."

        # Parse test inputs
        try:
            input_namespace = {}
            exec(test_input_code, input_namespace)
            args = input_namespace.get("args", [])
            kwargs = input_namespace.get("kwargs", {})
        except Exception as e:
            return f"❌ 테스트 입력 코드 오류: {e}"

        # 파일에서 Triton 커널 실행
        triton_result = runner.execute_from_file(
            latest_kernel.kernel_file,
            "solve",
            args,
            kwargs,
        )
        if not triton_result.success:
            return f"""❌ Triton 커널 실행 실패

커널 파일: {latest_kernel.kernel_file}
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

        # Update kernel validation status
        details = f"최대 차이: {validation.max_diff:.2e}, 평균 차이: {validation.mean_diff:.2e}"
        state.update_kernel_validation(latest_kernel.version, validation.passed, details)

        # Handle state transitions in evaluation state
        transition_info = ""
        if state.get_status() == Status.EVALUATION:
            if validation.passed:
                if state.write_count >= state.min_write_count:
                    # Can transition to end
                    state.transition_to(Status.END, "검증 통과 + 최소 write 조건 충족")
                    transition_info = "\n\n🎉 상태 전환: evaluation → end\n모든 조건을 충족했습니다! get_best_kernel()을 호출하세요."
                else:
                    # 자동으로 write 상태로 전환
                    remaining = state.min_write_count - state.write_count
                    state.transition_to(
                        Status.WRITE, f"검증 통과했지만 최소 {remaining}번 더 write 필요"
                    )
                    transition_info = f"\n\n🔄 상태 전환: evaluation → write\n검증 통과했지만, 최소 {remaining}번 더 write가 필요합니다. 추가 최적화를 진행하세요."
            else:
                # Validation failed - transition back to write
                state.transition_to(Status.WRITE, "검증 실패")
                transition_info = (
                    "\n\n🔄 상태 전환: evaluation → write\n검증 실패로 코드 수정이 필요합니다."
                )

        if validation.passed:
            return f"""✅ 검증 통과

커널 버전: v{latest_kernel.version}
커널 파일: {latest_kernel.kernel_file}

최대 차이: {validation.max_diff:.2e}
평균 차이: {validation.mean_diff:.2e}
전체 요소: {validation.total_elements:,}
허용 오차: rtol={rtol}, atol={atol}
{transition_info}"""
        else:
            return f"""❌ 검증 실패

커널 버전: v{latest_kernel.version}
커널 파일: {latest_kernel.kernel_file}

최대 차이: {validation.max_diff:.2e}
평균 차이: {validation.mean_diff:.2e}
불일치 요소: {validation.num_mismatches:,} / {validation.total_elements:,}
허용 오차: rtol={rtol}, atol={atol}

팁: fp16 사용 시 rtol=1e-3, atol=1e-3 정도가 적절합니다.
{transition_info}"""

    @mcp.tool()
    @log_tool_call(allowed_statuses=[Status.WRITE, Status.EVALUATION])
    def benchmark_kernel(
        test_input_code: str,
        reference_code: Optional[str] = None,
        warmup: int = 25,
        rep: int = 100,
    ) -> str:
        """
        현재 버전의 Triton 커널 성능을 측정합니다.

        이 도구는 'write' 또는 'evaluation' 상태에서 사용할 수 있습니다.

        Args:
            test_input_code: 테스트 입력 생성 코드
            reference_code: (선택) 비교할 PyTorch 참조 구현
            warmup: 워밍업 실행 횟수 (기본값: 25)
            rep: 측정 실행 횟수 (기본값: 100)

        Returns:
            성능 측정 결과 (평균, 표준편차, 최소/최대 시간)
        """
        state = get_state_manager()
        runner = get_runner()

        if not runner.gpu_available:
            return "❌ GPU가 없어서 벤치마크를 수행할 수 없습니다."

        # 현재 커널 버전 가져오기
        latest_kernel = state.get_latest_kernel()
        if latest_kernel is None:
            return "❌ 커널이 없습니다. 먼저 write_kernel_code()로 커널을 작성하세요."

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

        # 파일에서 커널 벤치마크
        result = runner.benchmark_from_file(
            latest_kernel.kernel_file,
            "solve",
            args,
            kwargs,
            warmup=warmup,
            rep=rep,
            reference_fn=reference_fn,
        )

        if not result.success:
            return f"❌ 벤치마크 실패: {result.error}"

        # Update kernel timing
        state.update_kernel_timing(
            latest_kernel.version,
            result.mean_ms,
            result.min_ms,
            result.max_ms,
        )

        # Check if we need to auto-transition to write
        transition_info = ""
        if state.get_status() == Status.EVALUATION:
            if state.write_count < state.min_write_count:
                remaining = state.min_write_count - state.write_count
                state.transition_to(
                    Status.WRITE, f"벤치마크 완료, 최소 {remaining}번 더 write 필요"
                )
                transition_info = f"\n\n🔄 상태 전환: evaluation → write\n최소 {remaining}번 더 write가 필요합니다. 추가 최적화를 진행하세요."

        output = f"""📊 벤치마크 결과

커널 버전: v{latest_kernel.version}
커널 파일: {latest_kernel.kernel_file}

실행 횟수: {result.num_runs}
평균: {result.mean_ms:.4f} ms
표준편차: {result.std_ms:.4f} ms
최소: {result.min_ms:.4f} ms
최대: {result.max_ms:.4f} ms
{transition_info}
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
{transition_info}
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
