"""Workflow tools for Triton kernel development."""

from typing import Optional
from pathlib import Path

from fastmcp import FastMCP

from ..state import Status, get_state_manager, log_tool_call
from ..utils.runner import TritonRunner
from ..utils.context_manager import get_context_manager


# Global runner instance (lazy initialization)
_runner: Optional[TritonRunner] = None


def get_runner() -> TritonRunner:
    """Get or create the global TritonRunner instance."""
    global _runner
    if _runner is None:
        _runner = TritonRunner()
    return _runner


# 성능 최적화 팁
OPTIMIZATION_TIPS = """
## 🚀 성능 최적화 팁

### 1. BLOCK_SIZE 튜닝
- 2의 거듭제곱 사용: 64, 128, 256, 512, 1024
- 작은 데이터: 128-256, 큰 데이터: 512-1024
- @triton.autotune으로 자동 튜닝 가능

### 2. 메모리 접근 최적화
- Coalesced access: 연속 메모리 접근이 빠름
- Stride 최소화: stride(0)이 가장 빠름
- 불필요한 메모리 복사 제거

### 3. 연산 융합 (Fusion)
- 여러 elementwise 연산을 하나의 커널로 합치기
- 중간 결과를 레지스터에 유지
- 메모리 bandwidth 병목 해결

### 4. 레지스터 사용
- accumulator는 float32 사용 (정확도)
- 중간 계산 결과는 레지스터에 유지
- tl.zeros로 초기화된 accumulator 사용

### 5. 마스크 최적화
- 경계 조건 마스크는 필수
- 가능하면 full block 처리 (마스크 불필요)

### 예시: autotune 적용
```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["N"],
)
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```
"""


def register_workflow_tools(mcp: FastMCP) -> None:
    """Register workflow management tools to the MCP server."""

    @mcp.tool()
    def get_current_status() -> str:
        """
        현재 워크플로우 상태를 확인합니다.
        
        Returns:
            현재 상태 정보 (상태, 버전 수, 정보 수집 현황)
        """
        state = get_state_manager()
        context_mgr = get_context_manager()
        
        info_status = "\n".join(
            f"  - {tool}: {'✅' if collected else '❌'}"
            for tool, collected in state.info_collected.items()
        )
        
        kernel_info = ""
        if state.kernel_versions:
            kernel_info = "\n\n커널 버전:"
            for kv in state.kernel_versions:
                validated = "✅" if kv.validation_passed else ("❌" if kv.validation_passed is False else "⏳")
                timing = f"{kv.mean_time_ms:.4f}ms" if kv.mean_time_ms else "미측정"
                kernel_info += f"\n  - v{kv.version}: 검증 {validated}, 시간 {timing}"
        
        log_info = ""
        if state.md_log_file:
            log_info = f"\n\n로그 파일: {state.md_log_file}"
        
        context_info = f"""
Context 사용량: {context_mgr.get_usage_ratio() * 100:.1f}% ({context_mgr.estimated_tokens:,} / {context_mgr.max_context_tokens:,} tokens)
도구 호출 횟수: {context_mgr.tool_call_count}
"""
        
        return f"""📊 현재 워크플로우 상태

상태: {state.get_status_str()}
커널 이름: {state.kernel_name or '미설정'}
작성 횟수: {state.write_count} / {state.min_write_count} (최소 필요)
세션 ID: {state.session_id}

정보 수집 현황:
{info_status}
{kernel_info}
{log_info}
{context_info}

다음 단계:
{_get_next_step_hint(state)}
"""

    @mcp.tool()
    def check_context_usage() -> str:
        """
        현재 context 사용량을 확인합니다.
        
        Returns:
            Context 사용량 정보 및 요약 생성 안내
        """
        context_mgr = get_context_manager()
        usage_ratio = context_mgr.get_usage_ratio()
        
        status = "✅ 정상" if usage_ratio < 0.7 else "⚠️ 주의" if usage_ratio < 0.9 else "🔴 위험"
        
        message = f"""📊 Context 사용량

{status}
- 사용률: {usage_ratio * 100:.1f}%
- 추정 토큰: {context_mgr.estimated_tokens:,} / {context_mgr.max_context_tokens:,}
- 도구 호출 횟수: {context_mgr.tool_call_count}
"""
        
        if usage_ratio >= 0.7:
            message += f"""
⚠️ Context 사용량이 70%를 초과했습니다.

다음 도구 호출 시 자동으로 요약이 생성되고 새 세션을 시작하라는 안내가 표시됩니다.
요약 파일: {context_mgr.summarization_file}
"""
        elif usage_ratio >= 0.5:
            message += """
💡 Context 사용량이 50%를 넘었습니다. 곧 요약이 필요할 수 있습니다.
"""
        
        return message

    @mcp.tool()
    def set_kernel_name(name: str) -> str:
        """
        커널 이름을 설정합니다.
        
        이 이름은 로그 파일명과 커널 파일명에 사용됩니다.
        예: "sub" → triton_sub_log.md, triton_sub_kernel_v1.py
        
        Args:
            name: 커널 이름 (예: "sub", "add", "softmax")
        
        Returns:
            설정 결과
        """
        state = get_state_manager()
        state.set_kernel_name(name)
        
        return f"""✅ 커널 이름 설정 완료

커널 이름: {state.kernel_name}
로그 파일: {state.md_log_file}

이제 정보 수집을 진행하세요:
1. get_overview() - 전체 프로세스 파악
2. get_torch_op_info() - 연산 정보 확인
3. get_triton_syntax() - Triton 문법 참고
4. check_gpu_status() - GPU 확인
"""

    @mcp.tool()
    @log_tool_call(allowed_statuses=[Status.WRITE])
    def write_kernel_code(code: str) -> str:
        """
        Triton 커널 코드를 작성하고 저장합니다.
        
        이 도구는 'write' 상태에서만 사용할 수 있습니다.
        호출 후 자동으로 'evaluation' 상태로 전환됩니다.
        
        커널 코드는 kernel/ 디렉토리에 저장됩니다.
        
        Args:
            code: Triton 커널과 solve 함수가 포함된 전체 Python 코드
        
        Returns:
            저장 결과 및 버전 정보
        """
        state = get_state_manager()
        
        # 커널 이름이 설정되지 않았으면 기본값 사용
        if state.kernel_name is None:
            state.set_kernel_name("unnamed")
        
        # Syntax check first
        try:
            compile(code, "<kernel>", "exec")
        except SyntaxError as e:
            return f"""❌ 문법 오류
            
라인 {e.lineno}: {e.msg}

코드를 수정 후 다시 시도하세요."""

        # 커널 파일 저장
        version = state.write_count + 1
        kernel_filename = f"triton_{state.kernel_name}_kernel_v{version}.py"
        kernel_path = state.kernel_dir / kernel_filename
        
        with open(kernel_path, "w", encoding="utf-8") as f:
            f.write(code)
        
        # Add kernel version
        version = state.add_kernel_version(code, str(kernel_path))
        
        # Transition to evaluation
        state.transition_to(Status.EVALUATION, "코드 작성 완료")
        
        return f"""✅ 커널 코드 저장 완료

버전: v{version}
파일: {kernel_path}
코드 길이: {len(code)} characters
상태: write → evaluation

다음 단계:
1. run_triton_kernel() - 커널 실행 테스트
2. validate_correctness() - 정확성 검증
3. measure_kernel_time() - 성능 측정

현재 작성 횟수: {state.write_count} / {state.min_write_count} (최소 필요)

---
{OPTIMIZATION_TIPS}
"""

    @mcp.tool()
    @log_tool_call(allowed_statuses=[Status.WRITE, Status.EVALUATION])
    def measure_kernel_time(
        test_input_code: str,
        warmup: int = 25,
        rep: int = 100,
    ) -> str:
        """
        현재 버전의 Triton 커널 실행 시간을 측정합니다.
        
        Args:
            test_input_code: 테스트 입력 생성 코드 (args, kwargs 정의)
            warmup: 워밍업 실행 횟수 (기본값: 25)
            rep: 측정 실행 횟수 (기본값: 100)
        
        Returns:
            시간 측정 결과 (평균, 표준편차, 최소/최대)
        """
        state = get_state_manager()
        runner = get_runner()
        
        if not runner.gpu_available:
            return "❌ GPU가 없어서 시간 측정을 수행할 수 없습니다."
        
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
        
        # 파일에서 커널 실행
        result = runner.benchmark_from_file(
            latest_kernel.kernel_file,
            "solve",
            args,
            kwargs,
            warmup=warmup,
            rep=rep,
        )
        
        if not result.success:
            return f"❌ 시간 측정 실패: {result.error}"
        
        # Update kernel timing
        state.update_kernel_timing(
            latest_kernel.version,
            result.mean_ms,
            result.min_ms,
            result.max_ms,
        )
        
        return f"""⏱️ 시간 측정 결과

커널 버전: v{latest_kernel.version}
커널 파일: {latest_kernel.kernel_file}

실행 횟수: {result.num_runs}
평균: {result.mean_ms:.4f} ms
표준편차: {result.std_ms:.4f} ms
최소: {result.min_ms:.4f} ms
최대: {result.max_ms:.4f} ms
"""

    @mcp.tool()
    @log_tool_call(allowed_statuses=[Status.END])
    def get_best_kernel() -> str:
        """
        가장 빠른 커널 정보를 반환하고 세션을 종료합니다.
        
        이 도구는 'end' 상태에서만 사용할 수 있습니다.
        모든 버전 중 검증을 통과하고 가장 빠른 커널을 선택합니다.
        
        Returns:
            최고 성능 커널 정보 및 코드
        """
        state = get_state_manager()
        
        best = state.get_best_kernel()
        
        if best is None:
            return """❌ 유효한 커널을 찾을 수 없습니다.

검증을 통과한 커널이 없거나 시간 측정이 되지 않았습니다.
"""
        
        # 마크다운 로그 완료
        state.finalize_log()
        
        # Generate comparison table
        comparison = "## 모든 버전 비교\n\n"
        comparison += "| 버전 | 검증 | 평균 시간 (ms) | 최소 시간 (ms) | 파일 |\n"
        comparison += "|------|------|---------------|---------------|------|\n"
        
        for kv in state.kernel_versions:
            validated = "✅" if kv.validation_passed else "❌"
            mean_time = f"{kv.mean_time_ms:.4f}" if kv.mean_time_ms else "-"
            min_time = f"{kv.min_time_ms:.4f}" if kv.min_time_ms else "-"
            is_best = " 🏆" if kv.version == best.version else ""
            filename = Path(kv.kernel_file).name if kv.kernel_file else "-"
            comparison += f"| v{kv.version}{is_best} | {validated} | {mean_time} | {min_time} | {filename} |\n"
        
        return f"""🏆 최고 성능 커널

## 선택된 버전: v{best.version}

- **파일**: {best.kernel_file}
- **평균 시간**: {best.mean_time_ms:.4f} ms
- **최소 시간**: {best.min_time_ms:.4f} ms
- **최대 시간**: {best.max_time_ms:.4f} ms

{comparison}

## 최종 커널 코드

```python
{best.code}
```

---
- **세션 ID**: {state.session_id}
- **총 작성 버전**: {state.write_count}
- **로그 파일**: {state.md_log_file}
"""

    @mcp.tool()
    def force_transition_to_write() -> str:
        """
        evaluation 상태에서 write 상태로 강제 전환합니다.
        
        추가 최적화가 필요하거나 다른 접근법을 시도하고 싶을 때 사용합니다.
        
        Returns:
            전환 결과
        """
        state = get_state_manager()
        
        if state.get_status() != Status.EVALUATION:
            return f"❌ 현재 상태({state.get_status_str()})에서는 이 도구를 사용할 수 없습니다.\nevaluation 상태에서만 사용 가능합니다."
        
        state.transition_to(Status.WRITE, "수동 전환: 추가 최적화")
        
        return f"""✅ 상태 전환 완료

evaluation → write

이제 새로운 커널 버전을 작성할 수 있습니다.
현재 작성 횟수: {state.write_count}
남은 최소 작성 횟수: {max(0, state.min_write_count - state.write_count)}

{OPTIMIZATION_TIPS}
"""


def _get_next_step_hint(state) -> str:
    """Get a hint for the next step based on current status."""
    status = state.get_status()
    
    if status == Status.START:
        if state.kernel_name is None:
            return "먼저 set_kernel_name()으로 커널 이름을 설정하세요."
        missing = [tool for tool, done in state.info_collected.items() if not done]
        if missing:
            return f"정보 수집이 필요합니다: {', '.join(missing)}"
        return "모든 정보가 수집되었습니다. write 상태로 전환됩니다."
    
    elif status == Status.WRITE:
        return "write_kernel_code()로 커널 코드를 작성하세요."
    
    elif status == Status.EVALUATION:
        current = state.get_current_version()
        kv = next((k for k in state.kernel_versions if k.version == current), None)
        
        if kv is None:
            return "커널 버전을 찾을 수 없습니다."
        
        steps = []
        if kv.validation_passed is None:
            steps.append("validate_correctness() - 정확성 검증 필요")
        if kv.mean_time_ms is None:
            steps.append("measure_kernel_time() - 시간 측정 필요")
        
        if not steps:
            if state.write_count >= state.min_write_count:
                if kv.validation_passed:
                    return "모든 조건 충족! end 상태로 전환 가능합니다."
                else:
                    return "검증 실패. write 상태로 돌아가서 코드를 수정하세요."
            else:
                remaining = state.min_write_count - state.write_count
                return f"최소 {remaining}번 더 write가 필요합니다. force_transition_to_write()를 호출하세요."
        
        return "\n".join(steps)
    
    elif status == Status.END:
        return "get_best_kernel()로 최고 성능 커널을 확인하세요."
    
    return ""
