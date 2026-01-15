"""State management and logging for MCP Triton Kernel workflow."""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Any, Dict, List
from functools import wraps


class Status(Enum):
    """Workflow status states."""
    START = "start"
    WRITE = "write"
    EVALUATION = "evaluation"
    END = "end"


@dataclass
class KernelVersion:
    """Information about a kernel version."""
    version: int
    code: str
    created_at: str
    kernel_file: Optional[str] = None  # 커널 파일 경로
    test_file: Optional[str] = None    # 테스트 파일 경로
    validation_passed: Optional[bool] = None
    mean_time_ms: Optional[float] = None
    min_time_ms: Optional[float] = None
    max_time_ms: Optional[float] = None


@dataclass
class LogEntry:
    """Single log entry."""
    timestamp: str
    tool: str
    status_before: str
    status_after: str
    args: Dict[str, Any]
    result: Dict[str, Any]
    kernel_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class StateManager:
    """Manages workflow state, kernel versions, and logging."""
    
    # Singleton instance
    _instance: Optional["StateManager"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.session_id = str(uuid.uuid4())[:8]
        self.status = Status.START
        self.kernel_versions: List[KernelVersion] = []
        self.write_count = 0
        self.min_write_count = 3
        self.kernel_name: Optional[str] = None  # 커널 이름 (예: "sub", "add")
        
        # Track which info tools have been called
        self.info_collected = {
            "get_overview": False,
            "get_torch_op_info": False,
            "get_triton_syntax": False,
            "check_gpu_status": False,
        }
        
        # mcp_for_triton_kernel 디렉토리 (현재 패키지 디렉토리)
        self.package_dir = Path(__file__).parent
        
        # 디렉토리 설정: mcp_for_triton_kernel/log/, mcp_for_triton_kernel/kernel/, mcp_for_triton_kernel/tests/
        self.log_dir = self.package_dir / "log"
        self.kernel_dir = self.package_dir / "kernel"
        self.tests_dir = self.package_dir / "tests"
        
        # 디렉토리 생성
        self.log_dir.mkdir(exist_ok=True)
        self.kernel_dir.mkdir(exist_ok=True)
        self.tests_dir.mkdir(exist_ok=True)
        
        # JSON 로그 파일
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_log_file = self.log_dir / f"{self.session_id}_{timestamp}.jsonl"
        
        # 마크다운 로그 파일 (커널 이름 설정 후 생성)
        self.md_log_file: Optional[Path] = None
        
        self._log_entries: List[LogEntry] = []
    
    def set_kernel_name(self, name: str):
        """커널 이름을 설정하고 마크다운 로그 파일을 생성합니다."""
        self.kernel_name = name.lower().replace(" ", "_")
        self.md_log_file = self.log_dir / f"triton_{self.kernel_name}_log.md"
        self._init_md_log()
    
    def _init_md_log(self):
        """마크다운 로그 파일 초기화."""
        if self.md_log_file is None:
            return
        
        content = f"""# Triton {self.kernel_name} Kernel Development Log

## 세션 정보
- **세션 ID**: {self.session_id}
- **시작 시간**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **상태**: {self.status.value}

---

## 작업 로그

"""
        with open(self.md_log_file, "w", encoding="utf-8") as f:
            f.write(content)
    
    def _append_md_log(self, content: str):
        """마크다운 로그에 내용 추가."""
        if self.md_log_file is None:
            return
        
        with open(self.md_log_file, "a", encoding="utf-8") as f:
            f.write(content)
    
    def reset(self):
        """Reset the state manager for a new session."""
        self.session_id = str(uuid.uuid4())[:8]
        self.status = Status.START
        self.kernel_versions = []
        self.write_count = 0
        self.kernel_name = None
        self.info_collected = {
            "get_overview": False,
            "get_torch_op_info": False,
            "get_triton_syntax": False,
            "check_gpu_status": False,
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_log_file = self.log_dir / f"{self.session_id}_{timestamp}.jsonl"
        self.md_log_file = None
        self._log_entries = []
    
    def get_status(self) -> Status:
        """Get current status."""
        return self.status
    
    def get_status_str(self) -> str:
        """Get current status as string."""
        return self.status.value
    
    def can_transition_to_write(self) -> bool:
        """Check if all required info has been collected."""
        return all(self.info_collected.values())
    
    def mark_info_collected(self, tool_name: str):
        """Mark an info tool as having been called."""
        if tool_name in self.info_collected:
            self.info_collected[tool_name] = True
        
        # Auto-transition from start to write if all info collected
        if self.status == Status.START and self.can_transition_to_write():
            self.status = Status.WRITE
            self._append_md_log(f"### [{datetime.now().strftime('%H:%M:%S')}] 상태 전환: start → write\n\n모든 정보 수집 완료\n\n")
    
    def transition_to(self, new_status: Status, reason: str = "") -> bool:
        """
        Attempt to transition to a new status.
        
        Returns:
            True if transition successful, False otherwise
        """
        old_status = self.status
        
        # Validate transition
        valid_transitions = {
            Status.START: [Status.WRITE],
            Status.WRITE: [Status.EVALUATION],
            Status.EVALUATION: [Status.WRITE, Status.END],
            Status.END: [],  # Terminal state
        }
        
        if new_status not in valid_transitions.get(old_status, []):
            return False
        
        # Special check: evaluation -> end requires min writes
        if old_status == Status.EVALUATION and new_status == Status.END:
            if self.write_count < self.min_write_count:
                return False
        
        self.status = new_status
        self._append_md_log(f"### [{datetime.now().strftime('%H:%M:%S')}] 상태 전환: {old_status.value} → {new_status.value}\n\n{reason}\n\n")
        return True
    
    def add_kernel_version(self, code: str, kernel_file: str, test_file: Optional[str] = None) -> int:
        """
        Add a new kernel version.
        
        Returns:
            Version number
        """
        self.write_count += 1
        version = self.write_count
        
        kernel_version = KernelVersion(
            version=version,
            code=code,
            created_at=datetime.now().isoformat(),
            kernel_file=kernel_file,
            test_file=test_file,
        )
        self.kernel_versions.append(kernel_version)
        
        # 마크다운 로그에 기록
        self._append_md_log(f"""### [{datetime.now().strftime('%H:%M:%S')}] 커널 v{version} 작성

- **파일**: `{kernel_file}`
- **코드 길이**: {len(code)} characters

""")
        
        return version
    
    def update_kernel_validation(self, version: int, passed: bool, details: str = ""):
        """Update validation result for a kernel version."""
        for kv in self.kernel_versions:
            if kv.version == version:
                kv.validation_passed = passed
                break
        
        status = "✅ 통과" if passed else "❌ 실패"
        self._append_md_log(f"""### [{datetime.now().strftime('%H:%M:%S')}] 커널 v{version} 검증: {status}

{details}

""")
    
    def update_kernel_timing(self, version: int, mean_ms: float, min_ms: float, max_ms: float):
        """Update timing info for a kernel version."""
        for kv in self.kernel_versions:
            if kv.version == version:
                kv.mean_time_ms = mean_ms
                kv.min_time_ms = min_ms
                kv.max_time_ms = max_ms
                break
        
        self._append_md_log(f"""### [{datetime.now().strftime('%H:%M:%S')}] 커널 v{version} 성능 측정

| 지표 | 값 |
|------|-----|
| 평균 | {mean_ms:.4f} ms |
| 최소 | {min_ms:.4f} ms |
| 최대 | {max_ms:.4f} ms |

""")
    
    def get_best_kernel(self) -> Optional[KernelVersion]:
        """Get the kernel with best (lowest) mean time."""
        valid_kernels = [
            kv for kv in self.kernel_versions 
            if kv.validation_passed and kv.mean_time_ms is not None
        ]
        
        if not valid_kernels:
            # If no timed kernels, return the last validated one
            validated = [kv for kv in self.kernel_versions if kv.validation_passed]
            return validated[-1] if validated else None
        
        return min(valid_kernels, key=lambda kv: kv.mean_time_ms)
    
    def get_current_version(self) -> int:
        """Get the current (latest) version number."""
        return self.write_count
    
    def get_latest_kernel(self) -> Optional[KernelVersion]:
        """Get the latest kernel version."""
        if self.kernel_versions:
            return self.kernel_versions[-1]
        return None
    
    def log(
        self,
        tool: str,
        status_before: str,
        status_after: str,
        args: Dict[str, Any],
        result: Dict[str, Any],
        kernel_info: Optional[Dict[str, Any]] = None,
    ):
        """Log a tool invocation."""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            tool=tool,
            status_before=status_before,
            status_after=status_after,
            args=args,
            result=result,
            kernel_info=kernel_info,
        )
        
        self._log_entries.append(entry)
        
        # Write to JSON log file
        with open(self.json_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    
    def finalize_log(self):
        """세션 종료 시 마크다운 로그를 완료합니다."""
        if self.md_log_file is None:
            return
        
        best = self.get_best_kernel()
        
        summary = f"""---

## 최종 결과

- **총 작성 버전**: {self.write_count}
- **최고 성능 버전**: v{best.version if best else 'N/A'}
- **최고 성능 시간**: {best.mean_time_ms:.4f} ms (평균)
- **종료 시간**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### 버전 비교

| 버전 | 검증 | 평균 시간 (ms) | 최소 시간 (ms) |
|------|------|---------------|---------------|
"""
        for kv in self.kernel_versions:
            validated = "✅" if kv.validation_passed else "❌"
            mean_time = f"{kv.mean_time_ms:.4f}" if kv.mean_time_ms else "-"
            min_time = f"{kv.min_time_ms:.4f}" if kv.min_time_ms else "-"
            is_best = " 🏆" if best and kv.version == best.version else ""
            summary += f"| v{kv.version}{is_best} | {validated} | {mean_time} | {min_time} |\n"
        
        if best:
            summary += f"""
### 최종 커널 코드 (`{best.kernel_file}`)

```python
{best.code}
```
"""
        
        self._append_md_log(summary)
    
    def get_logs(self) -> List[LogEntry]:
        """Get all log entries."""
        return self._log_entries


# Global instance
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get the global StateManager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


def log_tool_call(allowed_statuses: Optional[List[Status]] = None):
    """
    Decorator to add logging and status checking to tool functions.
    
    Args:
        allowed_statuses: List of statuses where this tool can be called.
                         None means any status.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from .utils.context_manager import get_context_manager
            
            state = get_state_manager()
            context_mgr = get_context_manager()
            status_before = state.get_status_str()
            
            # Check if tool is allowed in current status
            if allowed_statuses is not None:
                if state.get_status() not in allowed_statuses:
                    allowed_str = ", ".join(s.value for s in allowed_statuses)
                    return f"❌ 이 도구는 현재 상태({status_before})에서 사용할 수 없습니다.\n허용된 상태: {allowed_str}"
            
            # Execute the tool
            try:
                result = func(*args, **kwargs)
                success = True
                error_msg = None
            except Exception as e:
                result = f"❌ 오류 발생: {str(e)}"
                success = False
                error_msg = str(e)
            
            status_after = state.get_status_str()
            
            # Prepare args for logging (exclude large code blocks)
            logged_args = {}
            for k, v in kwargs.items():
                if isinstance(v, str) and len(v) > 500:
                    logged_args[k] = f"<{len(v)} chars>"
                else:
                    logged_args[k] = v
            
            # Track context usage
            result_text = str(result) if result else ""
            args_text = str(logged_args)
            usage_text = f"{func.__name__} {args_text} {result_text}"
            summary_msg = context_mgr.add_usage(usage_text, func.__name__)
            
            # Log the call
            state.log(
                tool=func.__name__,
                status_before=status_before,
                status_after=status_after,
                args=logged_args,
                result={
                    "success": success,
                    "error": error_msg,
                },
            )
            
            # If context usage exceeded threshold, prepend summary message
            if summary_msg:
                return f"{summary_msg}\n\n---\n\n{result}"
            
            return result
        
        return wrapper
    return decorator
