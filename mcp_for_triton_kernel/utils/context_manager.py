"""Context usage tracking and summarization management."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from ..state import get_state_manager


class ContextManager:
    """Manages context usage and automatic summarization."""
    
    def __init__(self):
        self.package_dir = Path(__file__).parent.parent
        self.summarization_file = self.package_dir.parent / "docs" / "summarization.md"
        self.summarization_file.parent.mkdir(exist_ok=True)
        
        # Context usage tracking
        self.estimated_tokens = 0
        self.context_threshold = 0.70  # 70% threshold
        self.max_context_tokens = 128000  # Typical context window (adjust as needed)
        
        # Tool call tracking
        self.tool_call_count = 0
        self.last_summarization_time: Optional[datetime] = None
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for English, more for Korean)."""
        # Simple estimation: Korean text uses more tokens
        korean_chars = sum(1 for c in text if ord(c) >= 0xAC00 and ord(c) <= 0xD7A3)
        other_chars = len(text) - korean_chars
        # Korean: ~2 chars per token, English: ~4 chars per token
        return int(korean_chars / 2 + other_chars / 4)
    
    def add_usage(self, text: str, tool_name: str = ""):
        """Add estimated token usage."""
        tokens = self.estimate_tokens(text)
        self.estimated_tokens += tokens
        self.tool_call_count += 1
        
        # Check if we need to summarize
        usage_ratio = self.estimated_tokens / self.max_context_tokens
        if usage_ratio >= self.context_threshold:
            return self.create_summarization()
        
        return None
    
    def get_usage_ratio(self) -> float:
        """Get current context usage ratio."""
        return self.estimated_tokens / self.max_context_tokens
    
    def create_summarization(self) -> str:
        """Create summarization document and return message for new session."""
        state = get_state_manager()
        
        # Generate summarization content
        content = f"""# MCP for Triton Kernel - 세션 요약

> **이 문서는 context 사용량이 70%를 초과하여 자동 생성되었습니다.**
> **새로운 세션을 시작하여 계속 작업하세요.**

---

## 생성 시간
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 현재 상태

### 워크플로우 상태
- **상태**: {state.get_status_str()}
- **세션 ID**: {state.session_id}
- **커널 이름**: {state.kernel_name or "미설정"}
- **작성 횟수**: {state.write_count} / {state.min_write_count} (최소 필요)

### 정보 수집 현황
"""
        
        for tool, collected in state.info_collected.items():
            status = "✅ 완료" if collected else "❌ 미완료"
            content += f"- **{tool}**: {status}\n"
        
        content += "\n### 커널 버전\n\n"
        if state.kernel_versions:
            content += "| 버전 | 검증 | 평균 시간 (ms) | 파일 |\n"
            content += "|------|------|---------------|------|\n"
            for kv in state.kernel_versions:
                validated = "✅" if kv.validation_passed else ("❌" if kv.validation_passed is False else "⏳")
                mean_time = f"{kv.mean_time_ms:.4f}" if kv.mean_time_ms else "-"
                filename = Path(kv.kernel_file).name if kv.kernel_file else "-"
                content += f"| v{kv.version} | {validated} | {mean_time} | {filename} |\n"
        else:
            content += "아직 작성된 커널이 없습니다.\n"
        
        content += f"""
### Context 사용량
- **추정 토큰**: {self.estimated_tokens:,} / {self.max_context_tokens:,}
- **사용률**: {self.get_usage_ratio() * 100:.1f}%
- **도구 호출 횟수**: {self.tool_call_count}

---

## 다음 단계

### 현재 상태에서 계속하기

"""
        
        if state.get_status_str() == "start":
            content += """1. `set_kernel_name("커널이름")` - 커널 이름 설정
2. 정보 수집 도구들 호출 (get_overview, get_torch_op_info 등)
3. `write_kernel_code()` - 커널 코드 작성
"""
        elif state.get_status_str() == "write":
            content += """1. `write_kernel_code(code)` - 커널 코드 작성
2. 자동으로 evaluation 상태로 전환
"""
        elif state.get_status_str() == "evaluation":
            latest = state.get_latest_kernel()
            if latest:
                content += f"""1. `run_triton_kernel(test_input_code)` - 커널 실행 테스트
2. `validate_correctness(reference_code, test_input_code)` - 정확성 검증
3. `measure_kernel_time(test_input_code)` - 성능 측정

현재 버전: v{latest.version}
"""
            if state.write_count < state.min_write_count:
                content += f"\n⚠️ 최소 {state.min_write_count - state.write_count}번 더 write가 필요합니다.\n"
        elif state.get_status_str() == "end":
            content += """1. `get_best_kernel()` - 최고 성능 커널 확인
"""
        
        content += f"""
---

## 로그 파일

- **JSON 로그**: {state.json_log_file}
- **마크다운 로그**: {state.md_log_file or "미생성"}

---

## 새 세션 시작 방법

새로운 Cursor 창을 열거나, 현재 세션을 재시작하여 계속 작업하세요.
모든 상태 정보는 위에 요약되어 있습니다.

**중요**: 이전 세션의 StateManager는 초기화되지만, 생성된 커널 파일과 로그는 그대로 유지됩니다.
"""
        
        # Write to summarization file
        with open(self.summarization_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        self.last_summarization_time = datetime.now()
        
        return f"""⚠️ **Context 사용량이 70%를 초과했습니다**

📊 **현재 사용률**: {self.get_usage_ratio() * 100:.1f}%
📝 **요약 파일 저장**: {self.summarization_file}

## 새 세션 시작 안내

요약이 `docs/summarization.md`에 저장되었습니다.
**새로운 Cursor 창을 열거나 세션을 재시작**하여 계속 작업하세요.

### 현재 상태 요약
- 상태: {state.get_status_str()}
- 커널 이름: {state.kernel_name or "미설정"}
- 작성 횟수: {state.write_count} / {state.min_write_count}

새 세션에서 `docs/summarization.md`를 참고하여 이전 작업을 이어갈 수 있습니다.
"""


# Global instance
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get the global ContextManager instance."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager

