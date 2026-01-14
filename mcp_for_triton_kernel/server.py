"""MCP Server for Triton Kernel Development.

This server provides tools for:
1. Information about Triton syntax and torch ops
2. Running and validating Triton kernels
3. Benchmarking kernel performance
"""

from fastmcp import FastMCP

from .tools import register_info_tools, register_execution_tools

# Create MCP server instance
mcp = FastMCP(
    name="triton-kernel-mcp",
    instructions="""
    이 MCP 서버는 PyTorch 연산을 Triton 커널로 변환하는 작업을 지원합니다.
    
    작업 흐름:
    1. get_overview로 전체 프로세스 파악
    2. get_torch_op_info로 변환할 연산 정보 확인
    3. get_triton_syntax로 Triton 문법 참고
    4. 커널 코드 작성
    5. run_triton_kernel로 실행 테스트
    6. validate_correctness로 정확성 검증
    7. benchmark_kernel로 성능 측정
    """,
)

# Register all tools
register_info_tools(mcp)
register_execution_tools(mcp)


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()

