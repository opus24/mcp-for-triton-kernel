"""Pytest configuration for Triton kernel tests."""

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--kernel-version",
        action="store",
        default="best",
        help="Kernel version to test: v1, v2, v3, v4, or best (default: best)",
        choices=["v1", "v2", "v3", "v4", "best"],
    )
    parser.addoption(
        "--test-log-file",
        action="store",
        default=None,
        help="Path to log file for test results (default: tests/pytest_results.log)",
    )


def _load_kernel_module(kernel_name: str, version: str):
    """Load a kernel module dynamically."""
    kernel_dir = Path(__file__).parent.parent / "kernel"

    if version == "best":
        # Find the highest version available
        versions = []
        for v in ["v4", "v3", "v2", "v1"]:
            kernel_file = kernel_dir / f"triton_{kernel_name}_kernel_{v}.py"
            if kernel_file.exists():
                versions.append((v, kernel_file))
        if not versions:
            raise FileNotFoundError(f"No kernel files found for {kernel_name}")
        version, kernel_file = versions[0]  # Use highest version
    else:
        kernel_file = kernel_dir / f"triton_{kernel_name}_kernel_{version}.py"
        if not kernel_file.exists():
            raise FileNotFoundError(f"Kernel file not found: {kernel_file}")

    # Load module dynamically
    spec = importlib.util.spec_from_file_location(
        f"triton_{kernel_name}_kernel_{version}", kernel_file
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load kernel module: {kernel_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)

    return module, version


@pytest.fixture(scope="session")
def kernel_version(request):
    """Get the kernel version from pytest option."""
    return request.config.getoption("--kernel-version")


@pytest.fixture(scope="session")
def softmax_kernel(kernel_version):
    """Load softmax kernel module."""
    module, actual_version = _load_kernel_module("softmax", kernel_version)
    return module, actual_version


@pytest.fixture(scope="session")
def add_kernel(kernel_version):
    """Load add kernel module."""
    module, actual_version = _load_kernel_module("add", kernel_version)
    return module, actual_version


@pytest.fixture(scope="session")
def mul_kernel(kernel_version):
    """Load mul kernel module."""
    module, actual_version = _load_kernel_module("mul", kernel_version)
    return module, actual_version


@pytest.fixture(scope="session")
def sub_kernel(kernel_version):
    """Load sub kernel module."""
    module, actual_version = _load_kernel_module("sub", kernel_version)
    return module, actual_version


# Pytest hooks for logging
class TestLogger:
    """Logger for pytest test results."""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.test_results = []
        self.start_time = None
        self.end_time = None

    def log(self, message: str, print_to_console: bool = True):
        """Log a message to file and optionally to console."""
        if print_to_console:
            print(message)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def log_test_result(self, nodeid: str, outcome: str, duration: float = None, error: str = None):
        """Log individual test result."""
        result = {
            "nodeid": nodeid,
            "outcome": outcome,
            "duration": duration,
            "error": error,
        }
        self.test_results.append(result)

        status_emoji = {
            "passed": "✅",
            "failed": "❌",
            "skipped": "⏭️",
            "error": "⚠️",
        }.get(outcome, "❓")

        duration_str = f" ({duration:.3f}s)" if duration else ""
        message = f"{status_emoji} {nodeid}{duration_str}"
        if error:
            message += f"\n   Error: {error}"

        self.log(message)

    def log_summary(
        self, passed: int, failed: int, skipped: int, error: int, total_duration: float
    ):
        """Log test summary."""
        self.log("\n" + "=" * 80)
        self.log("📊 테스트 결과 요약")
        self.log("=" * 80)
        self.log(f"시작 시간: {self.start_time}")
        self.log(f"종료 시간: {self.end_time}")
        self.log(f"총 실행 시간: {total_duration:.3f}초")
        self.log("")
        self.log(f"✅ 통과: {passed}")
        self.log(f"❌ 실패: {failed}")
        self.log(f"⏭️  건너뜀: {skipped}")
        self.log(f"⚠️  오류: {error}")
        self.log(f"📝 총 테스트: {passed + failed + skipped + error}")
        self.log("")

        if failed > 0 or error > 0:
            self.log("=" * 80)
            self.log("❌ 실패한 테스트:")
            self.log("=" * 80)
            for result in self.test_results:
                if result["outcome"] in ["failed", "error"]:
                    self.log(f"  - {result['nodeid']}")
                    if result.get("error"):
                        self.log(f"    {result['error']}")
            self.log("")

        self.log(f"📄 전체 로그: {self.log_file}")
        self.log("=" * 80)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Configure pytest and initialize logger."""
    # Determine log file path
    log_file_option = config.getoption("--test-log-file")
    if log_file_option:
        log_file = Path(log_file_option)
    else:
        log_file = Path(__file__).parent / "pytest_results.log"

    # Create log file directory if needed
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = TestLogger(log_file)
    logger.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Clear previous log
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("pytest 실행 로그\n")
        f.write(f"시작 시간: {logger.start_time}\n")
        f.write("=" * 80 + "\n\n")

    # Store logger in config
    config._test_logger = logger

    # Log kernel version
    kernel_version = config.getoption("--kernel-version")
    logger.log(f"🔧 커널 버전: {kernel_version}\n")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results."""
    outcome = yield
    rep = outcome.get_result()

    # Get logger from config
    logger = item.config._test_logger

    if rep.when == "call":  # Only log on actual test call, not setup/teardown
        duration = rep.duration if hasattr(rep, "duration") else None
        error = None

        if rep.failed:
            if hasattr(rep, "longrepr") and rep.longrepr:
                error = str(rep.longrepr).split("\n")[0]  # First line of error

        logger.log_test_result(
            nodeid=item.nodeid,
            outcome=rep.outcome,
            duration=duration,
            error=error,
        )


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Log test session summary."""
    logger = session.config._test_logger
    logger.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate summary
    passed = sum(1 for r in logger.test_results if r["outcome"] == "passed")
    failed = sum(1 for r in logger.test_results if r["outcome"] == "failed")
    skipped = sum(1 for r in logger.test_results if r["outcome"] == "skipped")
    error = sum(1 for r in logger.test_results if r["outcome"] == "error")

    # Calculate total duration
    total_duration = sum(r["duration"] or 0 for r in logger.test_results)

    # Log summary
    logger.log_summary(passed, failed, skipped, error, total_duration)

    # Also print summary to console
    print("\n" + "=" * 80)
    print(f"📄 테스트 결과가 로그 파일에 저장되었습니다: {logger.log_file}")
    print("=" * 80)
