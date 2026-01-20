"""Workflow tools for Triton kernel development."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

from ..knowledge import KNOWLEDGE_DIR
from ..state import Status, get_state_manager, log_tool_call
from ..utils.context_manager import get_context_manager
from ..utils.runner import TritonRunner


def _get_optimization_guide(state, version: int) -> str:
    """Get optimization guide for current version."""
    # Load torch_ops.json to get optimization techniques
    torch_ops_path = KNOWLEDGE_DIR / "torch_ops.json"
    if not torch_ops_path.exists():
        return ""

    try:
        with open(torch_ops_path, "r", encoding="utf-8") as f:
            ops_data = json.load(f)

        # Find op info for current kernel
        kernel_name = state.kernel_name
        if kernel_name and kernel_name in ops_data:
            op_info = ops_data[kernel_name]
            if "optimization_techniques" in op_info and op_info["optimization_techniques"]:
                techniques = op_info["optimization_techniques"][:2]  # 최대 2개

                guide = "\n## 📋 버전별 최적화 가이드\n\n"

                if version == 1:
                    guide += "**v1 (현재)**: 기본 구현 - 최적화 없이 기본 기능만 구현하세요.\n"
                    guide += "다음 버전에서 최적화를 적용할 준비를 하세요.\n"
                elif version == 2:
                    if len(techniques) > 0:
                        guide += f"**v2 (현재)**: {techniques[0]['name']} 적용\n"
                        guide += f"- {techniques[0]['description']}\n"
                        guide += "- v1의 기본 구현에 첫 번째 최적화 기법만 추가하세요.\n"
                elif version == 3:
                    if len(techniques) > 1:
                        guide += f"**v3 (현재)**: {techniques[1]['name']} 적용\n"
                        guide += f"- {techniques[1]['description']}\n"
                        guide += "- v1의 기본 구현에 두 번째 최적화 기법만 추가하세요.\n"
                    elif len(techniques) > 0:
                        guide += f"**v3 (현재)**: {techniques[0]['name']}의 변형 적용\n"
                elif version == 4:
                    if len(techniques) >= 2:
                        guide += f"**v4 (현재)**: {techniques[0]['name']} + {techniques[1]['name']} 모두 적용\n"
                        guide += f"- 첫 번째: {techniques[0]['description']}\n"
                        guide += f"- 두 번째: {techniques[1]['description']}\n"
                        guide += "- v2와 v3의 최적화를 모두 결합하세요.\n"
                    elif len(techniques) > 0:
                        guide += f"**v4 (현재)**: {techniques[0]['name']}의 고급 변형 적용\n"

                guide += f"\n**진행 상황**: {version}/4 버전 완료\n"
                if version < 4:
                    guide += f"다음 버전(v{version + 1})에서는 다른 최적화 기법을 적용하세요.\n"

                return guide
    except Exception:
        pass

    return ""


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
    @log_tool_call(allowed_statuses=None)
    def list_ops() -> str:
        """
        등록된 모든 ops 목록을 반환합니다.

        Returns:
            등록된 ops 목록
        """
        from ..state import get_ops_list

        ops_list = get_ops_list()

        if not ops_list:
            return "등록된 ops가 없습니다.\n\nset_kernel_name()으로 ops를 등록하세요."

        ops_info = []
        for op_name in sorted(ops_list):
            state = get_state_manager(op_name)
            status_icon = {
                Status.START: "🟢",
                Status.WRITE: "🟡",
                Status.EVALUATION: "🔵",
                Status.END: "✅",
            }.get(state.get_status(), "⚪")

            version_count = len(state.kernel_versions)
            ops_info.append(
                f"{status_icon} **{op_name}**: {state.get_status_str()} (버전: {version_count})"
            )

        return f"""📋 등록된 Ops 목록 ({len(ops_list)}개)

{chr(10).join(ops_info)}

상태 설명:
- 🟢 start: 정보 수집 단계
- 🟡 write: 코드 작성 단계
- 🔵 evaluation: 검증 및 평가 단계
- ✅ end: 완료 단계
"""

    @mcp.tool()
    @log_tool_call(allowed_statuses=None)
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
                validated = (
                    "✅"
                    if kv.validation_passed
                    else ("❌" if kv.validation_passed is False else "⏳")
                )
                timing = f"{kv.mean_time_ms:.4f}ms" if kv.mean_time_ms else "미측정"
                kernel_info += f"\n  - v{kv.version}: 검증 {validated}, 시간 {timing}"

        log_info = ""
        if state.md_log_file:
            log_info = f"\n\n로그 파일: {state.md_log_file}"

        context_info = f"""
Context 사용량: {context_mgr.get_usage_ratio() * 100:.1f}% "
            f"({context_mgr.estimated_tokens:,} / "
            f"{context_mgr.max_context_tokens:,} tokens)
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
    @log_tool_call(allowed_statuses=None)
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
    @log_tool_call(allowed_statuses=None)
    def set_kernel_name(name: str) -> str:
        """
        커널 이름을 설정합니다.

        이 이름은 로그 파일명과 커널 파일명에 사용됩니다.
        예: "sub" → triton_sub_log.md, triton_sub_kernel_v1.py

        end 상태에서 호출하면 자동으로 새 세션을 시작합니다.

        Args:
            name: 커널 이름 (예: "sub", "add", "softmax")

        Returns:
            설정 결과
        """
        from ..state import set_current_kernel_name

        # 현재 커널 이름 설정 (ops별 StateManager 관리)
        set_current_kernel_name(name)

        # 해당 커널의 StateManager 가져오기
        # get_state_manager 내부에서 kernel_name이 다르면 자동으로 reset됨
        state = get_state_manager(name)

        # set_kernel_name 내부에서도 커널 이름이 바뀌면 자동으로 reset됨
        state.set_kernel_name(name)

        # 최종 확인: 상태가 START가 아니면 강제로 START로 변경
        if state.get_status() != Status.START and state.kernel_name == name.lower().replace(
            " ", "_"
        ):
            state.reset()

        status_msg = ""
        if state.get_status() == Status.START:
            status_msg = (
                "\n\n이제 정보 수집을 진행하세요:\n"
                "1. get_overview() - 전체 프로세스 파악\n"
                "2. get_torch_op_info() - 연산 정보 확인\n"
                "3. get_triton_syntax() - Triton 문법 참고\n"
                "4. check_gpu_status() - GPU 확인"
            )
        elif state.get_status() == Status.WRITE:
            status_msg = "\n\n이제 write_kernel_code()로 커널 코드를 작성하세요."

        return f"""✅ 커널 이름 설정 완료

커널 이름: {state.kernel_name}
로그 파일: {state.md_log_file}
상태: {state.get_status_str()}
{status_msg}
"""

    @mcp.tool()
    @log_tool_call(allowed_statuses=[Status.WRITE])
    def write_test_code(test_code: str) -> str:
        """
        테스트 코드를 작성하고 저장합니다.

        이 도구는 'write' 상태에서만 사용할 수 있습니다.
        write 상태에 처음 도달했을 때는 반드시 이 도구를 먼저 호출해야 합니다.

        테스트 코드는 tests/ 디렉토리에 저장됩니다.

        Args:
            test_code: 테스트 코드 (reference 함수와 테스트 케이스 포함)

        Returns:
            저장 결과
        """
        state = get_state_manager()

        # 커널 이름이 설정되지 않았으면 기본값 사용
        if state.kernel_name is None:
            state.set_kernel_name("unnamed")

        # Syntax check first
        try:
            compile(test_code, "<test>", "exec")
        except SyntaxError as e:
            return f"""❌ 문법 오류

라인 {e.lineno}: {e.msg}

코드를 수정 후 다시 시도하세요."""

        # 테스트 파일 저장
        test_filename = f"test_{state.kernel_name}_kernel.py"
        test_path = state.tests_dir / test_filename

        with open(test_path, "w", encoding="utf-8") as f:
            f.write(test_code)

        # Mark test code as written
        state.test_code_written = True

        # 마크다운 로그에 기록
        state._append_md_log(
            f"""### [{datetime.now().strftime('%H:%M:%S')}] 테스트 코드 작성

- **파일**: `{test_path}`
- **코드 길이**: {len(test_code)} characters

"""
        )

        return f"""✅ 테스트 코드 저장 완료

파일: {test_path}
코드 길이: {len(test_code)} characters

다음 단계:
write_kernel_code()로 커널 코드를 작성하세요.
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

        # write 상태에 처음 도달했을 때는 테스트 코드를 자동으로 생성
        if state.first_write_reached and not state.test_code_written:
            # 자동으로 기본 테스트 코드 생성
            test_code = _generate_default_test_code(state.kernel_name)
            if test_code:
                # 테스트 파일 저장
                test_filename = f"test_{state.kernel_name}_kernel.py"
                test_path = state.tests_dir / test_filename

                with open(test_path, "w", encoding="utf-8") as f:
                    f.write(test_code)

                # Mark test code as written
                state.test_code_written = True

                # 마크다운 로그에 기록
                state._append_md_log(
                    f"""### [{datetime.now().strftime('%H:%M:%S')}] 테스트 코드 자동 생성

- **파일**: `{test_path}`
- **코드 길이**: {len(test_code)} characters

"""
                )
            else:
                return """❌ 테스트 코드를 먼저 작성해야 합니다.

write 상태에 처음 도달했을 때는 반드시 write_test_code()를 먼저 호출해야 합니다.
테스트 코드를 작성한 후 write_kernel_code()를 호출하세요.
"""

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

        # Get optimization guide for current version
        optimization_guide = _get_optimization_guide(state, version)

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
{optimization_guide}
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

        # 현재 커널 버전 가져오기 (best 우선, 없으면 latest)
        from .execution import _get_kernel_to_use

        kernel, kernel_type = _get_kernel_to_use(state)
        if kernel is None:
            return "❌ 커널이 없습니다. 먼저 write_kernel_code()로 커널을 작성하세요."

        latest_kernel = kernel  # 변수명 호환성 유지

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

        # Check if we need to auto-transition to write
        transition_info = ""
        if state.get_status() == Status.EVALUATION:
            if state.write_count < state.min_write_count:
                remaining = state.min_write_count - state.write_count
                state.transition_to(
                    Status.WRITE,
                    f"시간 측정 완료, 최소 {remaining}번 더 write 필요",
                )
                transition_info = f"\n\n🔄 상태 전환: evaluation → write\n최소 {remaining}번 더 write가 필요합니다. 추가 최적화를 진행하세요."

        kernel_type_label = "🏆 best" if kernel_type == "best" else "📝 latest"
        return f"""⏱️ 시간 측정 결과

커널 타입: {kernel_type_label}
커널 버전: v{latest_kernel.version}
커널 파일: {latest_kernel.kernel_file}

실행 횟수: {result.num_runs}
평균: {result.mean_ms:.4f} ms
표준편차: {result.std_ms:.4f} ms
최소: {result.min_ms:.4f} ms
최대: {result.max_ms:.4f} ms
{transition_info}
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

        # 검증 통과 + 시간 측정된 커널만 비교 대상
        valid_kernels = [
            kv
            for kv in state.kernel_versions
            if kv.validation_passed and kv.mean_time_ms is not None
        ]

        # 시간 측정되지 않은 검증 통과 커널 확인
        validated_but_not_timed = [
            kv for kv in state.kernel_versions if kv.validation_passed and kv.mean_time_ms is None
        ]

        # 검증 실패 또는 미검증 커널 확인
        not_validated = [kv for kv in state.kernel_versions if not kv.validation_passed]

        best = state.get_best_kernel()

        if best is None:
            return """❌ 유효한 커널을 찾을 수 없습니다.

검증을 통과한 커널이 없거나 시간 측정이 되지 않았습니다.
"""

        # 마크다운 로그 완료
        state.finalize_log()

        # 경고 메시지 생성
        warnings = ""
        if validated_but_not_timed:
            versions = ", ".join([f"v{kv.version}" for kv in validated_but_not_timed])
            warnings += f"\n⚠️ **시간 미측정 커널**: {versions} (성능 비교에서 제외됨)\n"
        if not_validated:
            versions = ", ".join([f"v{kv.version}" for kv in not_validated])
            warnings += f"\n⚠️ **검증 실패/미검증 커널**: {versions} (성능 비교에서 제외됨)\n"

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
            comparison += (
                f"| v{kv.version}{is_best} | {validated} | "
                f"{mean_time} | {min_time} | {filename} |\n"
            )

        # 성능 비교 요약 추가
        if len(valid_kernels) > 1:
            sorted_kernels = sorted(valid_kernels, key=lambda kv: kv.mean_time_ms)
            fastest = sorted_kernels[0]
            slowest = sorted_kernels[-1]
            speedup = (
                slowest.mean_time_ms / fastest.mean_time_ms if fastest.mean_time_ms > 0 else 1.0
            )
            comparison += f"\n**성능 비교**: v{fastest.version}이 v{slowest.version}보다 {speedup:.2f}x 빠름\n"

        return f"""🏆 최고 성능 커널

## 선택된 버전: v{best.version}

- **파일**: {best.kernel_file}
- **평균 시간**: {best.mean_time_ms:.4f} ms
- **최소 시간**: {best.min_time_ms:.4f} ms
- **최대 시간**: {best.max_time_ms:.4f} ms
- **비교 대상 커널 수**: {len(valid_kernels)}개 (검증 통과 + 시간 측정)
{warnings}
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
    @log_tool_call(allowed_statuses=[Status.EVALUATION])
    def force_transition_to_write() -> str:
        """
        evaluation 상태에서 write 상태로 강제 전환합니다.

        추가 최적화가 필요하거나 다른 접근법을 시도하고 싶을 때 사용합니다.

        Returns:
            전환 결과
        """
        state = get_state_manager()

        if state.get_status() != Status.EVALUATION:
            return (
                f"❌ 현재 상태({state.get_status_str()})에서는 "
                f"이 도구를 사용할 수 없습니다.\n"
                f"evaluation 상태에서만 사용 가능합니다."
            )

        state.transition_to(Status.WRITE, "수동 전환: 추가 최적화")

        return f"""✅ 상태 전환 완료

evaluation → write

이제 새로운 커널 버전을 작성할 수 있습니다.
현재 작성 횟수: {state.write_count}
남은 최소 작성 횟수: {max(0, state.min_write_count - state.write_count)}

{OPTIMIZATION_TIPS}
"""

    @mcp.tool()
    @log_tool_call(allowed_statuses=None)
    def reset_session() -> str:
        """
        현재 세션을 리셋하고 새 세션을 시작합니다.

        새로운 커널을 작성하거나 기존 작업을 처음부터 다시 시작할 때 사용합니다.

        Returns:
            리셋 결과
        """
        state = get_state_manager()
        old_session_id = state.session_id

        state.reset()

        return f"""✅ 세션 리셋 완료

이전 세션 ID: {old_session_id}
새 세션 ID: {state.session_id}
상태: start

다음 단계:
1. set_kernel_name("커널이름") - 커널 이름 설정
2. 정보 수집 도구들 호출 (get_overview, get_torch_op_info 등)
3. write_kernel_code() - 커널 코드 작성
"""

    @mcp.tool()
    @log_tool_call(allowed_statuses=None)
    def reset_all_states() -> str:
        """
        모든 ops의 state를 초기화합니다.

        모든 등록된 ops의 상태를 START로 리셋하고, 세션 ID를 새로 생성합니다.
        커널 버전과 로그는 유지됩니다.

        Returns:
            초기화 결과
        """
        from ..state import reset_all_states as reset_all

        count = reset_all()

        return f"""✅ 모든 State 초기화 완료

초기화된 ops 개수: {count}

모든 ops의 상태가 START로 리셋되었습니다.
세션 ID가 새로 생성되었습니다.

다음 단계:
1. list_ops() - 등록된 ops 목록 확인
2. set_kernel_name("커널이름") - 작업할 커널 선택
3. 정보 수집 도구들 호출
"""

    @mcp.tool()
    @log_tool_call(allowed_statuses=None)
    def clear_all_states() -> str:
        """
        모든 ops의 state를 완전히 삭제합니다.

        모든 등록된 ops를 삭제하고 완전히 초기화합니다.
        주의: 이 작업은 되돌릴 수 없습니다.

        Returns:
            삭제 결과
        """
        from ..state import clear_all_states as clear_all

        count = clear_all()

        return f"""✅ 모든 State 삭제 완료

삭제된 ops 개수: {count}

모든 ops가 완전히 삭제되었습니다.

다음 단계:
1. set_kernel_name("커널이름") - 새로운 ops 등록
2. 정보 수집 도구들 호출
"""


def _generate_default_test_code(kernel_name: str) -> Optional[str]:
    """기본 테스트 코드를 자동으로 생성합니다."""
    if kernel_name is None:
        return None

    # torch_ops.json에서 연산 정보 가져오기
    torch_ops_path = KNOWLEDGE_DIR / "torch_ops.json"

    if not torch_ops_path.exists():
        return None

    try:
        with open(torch_ops_path, "r", encoding="utf-8") as f:
            ops_data = json.load(f)
    except Exception:
        return None

    # 커널 이름으로 연산 찾기 (vector_mul, vector_div 등)
    op_key = None
    normalized_name = kernel_name.lower().strip()

    # 정확한 매칭 시도
    if normalized_name in ops_data:
        op_key = normalized_name
    else:
        # 부분 매칭 시도 (mul -> vector_mul, div -> vector_div)
        for key in ops_data.keys():
            if (
                normalized_name in key.lower()
                or key.lower().replace("vector_", "") == normalized_name
            ):
                op_key = key
                break

    if op_key is None:
        # 기본 elementwise 연산으로 가정
        op_key = "vector_mul"  # 기본값

    op_info = ops_data[op_key]
    torch_equivalent = op_info.get("torch_equivalent", "torch.mul(A, B)")

    # torch 함수 이름 추출 (예: "torch.mul(A, B)" -> "torch.mul")
    torch_func = torch_equivalent.split("(")[0].strip()
    if " 또는 " in torch_func:
        torch_func = torch_func.split(" 또는 ")[0].strip()

    # 기본 테스트 코드 생성
    test_code = f'''"""Test suite for {kernel_name} kernel."""

import torch
import numpy as np

def reference(input1: torch.Tensor, input2: torch.Tensor = None) -> torch.Tensor:
    """
    PyTorch 참조 구현

    Args:
        input1: 첫 번째 입력 텐서
        input2: 두 번째 입력 텐서 (elementwise 연산의 경우)

    Returns:
        참조 결과 텐서
    """
    if input2 is not None:
        # Elementwise 연산
        return {torch_func}(input1, input2)
    else:
        # 단일 입력 연산
        return {torch_func}(input1)


# 테스트 케이스
def test_case_1():
    """기본 테스트: 작은 크기"""
    input1 = torch.randn(1024, device='cuda', dtype=torch.float32)
    input2 = torch.randn(1024, device='cuda', dtype=torch.float32)
    return input1, input2

def test_case_2():
    """중간 크기"""
    input1 = torch.randn(10000, device='cuda', dtype=torch.float32)
    input2 = torch.randn(10000, device='cuda', dtype=torch.float32)
    return input1, input2

def test_case_3():
    """큰 크기"""
    input1 = torch.randn(1000000, device='cuda', dtype=torch.float32)
    input2 = torch.randn(1000000, device='cuda', dtype=torch.float32)
    return input1, input2

# 기본 테스트 입력 (validate_correctness에서 사용)
args = [torch.randn(1024, device='cuda', dtype=torch.float32),
        torch.randn(1024, device='cuda', dtype=torch.float32)]
kwargs = {{}}
'''

    return test_code


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
        if state.first_write_reached and not state.test_code_written:
            return "write_test_code()로 테스트 코드를 먼저 작성하세요."
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
