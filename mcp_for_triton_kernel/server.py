"""MCP Server for Triton Kernel Development.

This server provides tools for:
1. Information about Triton syntax and torch ops
2. Running and validating Triton kernels
3. Benchmarking kernel performance
4. Workflow state management with logging
"""

from fastmcp import FastMCP

from .tools import register_execution_tools, register_info_tools, register_workflow_tools

# Create MCP server instance
mcp = FastMCP(
    name="triton-kernel-mcp",
    instructions="""
    이 MCP 서버는 PyTorch 연산을 Triton 커널로 변환하는 작업을 지원합니다.

    ## 상태 기반 워크플로우

    시스템은 4가지 상태로 구성됩니다:
    1. start: 정보 수집 단계 (get_overview, get_triton_syntax, get_torch_op_info, check_gpu_status)
    2. write: 코드 작성 단계 (write_kernel_code)
    3. evaluation: 검증 및 평가 단계 (validate_correctness, measure_kernel_time, benchmark_kernel)
    4. end: 완료 단계 (get_best_kernel)

    ## 작업 흐름

    1. [start] get_overview → 전체 프로세스 파악
    2. [start] get_torch_op_info → 변환할 연산 정보 확인
    3. [start] get_triton_syntax → Triton 문법 참고
    4. [start] check_gpu_status → GPU 확인
    5. [write] write_kernel_code → 커널 코드 작성
    6. [evaluation] run_triton_kernel → 실행 테스트
    7. [evaluation] validate_correctness → 정확성 검증
    8. [evaluation] measure_kernel_time → 시간 측정
    9. 최소 4번 write 후 모든 테스트 통과 시 → end
    10. [end] get_best_kernel → 최고 성능 커널 반환

    ## 중요 규칙

    - 각 도구는 허용된 상태에서만 실행 가능
    - 모든 도구 호출은 자동으로 로그 기록
    - evaluation에서 end로 가려면 최소 4번의 write 필요
    - 테스트 실패 시 자동으로 write 상태로 복귀
    """,
)

# Register all tools
register_info_tools(mcp)
register_execution_tools(mcp)
register_workflow_tools(mcp)


def main():
    """Run the MCP server."""
    # StateManager는 lazy initialization으로 필요할 때 생성됨
    mcp.run()


if __name__ == "__main__":
    main()
