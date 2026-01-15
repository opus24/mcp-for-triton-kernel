"""Utility for running Triton kernels dynamically."""

import sys
import os
import traceback
import importlib.util
import tempfile
import shutil
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from pathlib import Path
import io
import contextlib


@dataclass
class KernelResult:
    """Result of kernel execution."""
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: Optional[float] = None


@dataclass
class ValidationResult:
    """Result of correctness validation."""
    passed: bool
    max_diff: float = 0.0
    mean_diff: float = 0.0
    num_mismatches: int = 0
    total_elements: int = 0
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Result of kernel benchmarking."""
    success: bool
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    num_runs: int = 0
    error: Optional[str] = None
    comparison: dict = field(default_factory=dict)


class TritonRunner:
    """Runner for dynamically executing Triton kernel code.
    
    Triton의 @jit 데코레이터는 파일에 정의된 함수만 지원합니다.
    이 클래스는 코드를 임시 파일에 저장하고 import하여 실행합니다.
    """
    
    def __init__(self):
        self._check_gpu_available()
        self._temp_modules = []
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.device = torch.device("cuda")
                self.gpu_name = torch.cuda.get_device_name(0)
            else:
                self.device = torch.device("cpu")
                self.gpu_name = None
            return self.gpu_available
        except ImportError:
            self.gpu_available = False
            self.device = None
            self.gpu_name = None
            return False
    
    def _load_module_from_file(self, file_path: str, module_name: str = None):
        """파일에서 모듈을 동적으로 로드합니다."""
        if module_name is None:
            module_name = Path(file_path).stem
        
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        return module
    
    def _create_temp_module(self, code: str) -> tuple:
        """코드를 임시 파일에 저장하고 모듈로 로드합니다.
        
        Returns:
            (module, temp_file_path)
        """
        import uuid
        
        # 고유한 모듈 이름 생성
        module_name = f"triton_kernel_{uuid.uuid4().hex[:8]}"
        
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp(prefix="triton_")
        temp_file = os.path.join(temp_dir, f"{module_name}.py")
        
        # 코드 저장
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(code)
        
        # sys.path에 임시 디렉토리 추가
        if temp_dir not in sys.path:
            sys.path.insert(0, temp_dir)
        
        # 모듈 로드
        module = self._load_module_from_file(temp_file, module_name)
        
        self._temp_modules.append((temp_dir, module_name))
        
        return module, temp_file
    
    def _cleanup_temp_modules(self):
        """임시 모듈과 파일을 정리합니다."""
        for temp_dir, module_name in self._temp_modules:
            # 모듈 언로드
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # sys.path에서 제거
            if temp_dir in sys.path:
                sys.path.remove(temp_dir)
            
            # 파일 삭제
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        
        self._temp_modules = []
    
    def execute_from_file(
        self,
        file_path: str,
        entry_function: str = "solve",
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
    ) -> KernelResult:
        """파일에서 Triton 커널을 로드하고 실행합니다."""
        if not self.gpu_available:
            return KernelResult(
                success=False,
                error="GPU is not available. Triton requires CUDA.",
                error_type="GPUNotAvailable",
            )
        
        args = args or []
        kwargs = kwargs or {}
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):
                
                module = self._load_module_from_file(file_path)
                
                if not hasattr(module, entry_function):
                    return KernelResult(
                        success=False,
                        error=f"Entry function '{entry_function}' not found in {file_path}",
                        error_type="FunctionNotFound",
                        stdout=stdout_capture.getvalue(),
                        stderr=stderr_capture.getvalue(),
                    )
                
                func = getattr(module, entry_function)
                
                import time
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                
                return KernelResult(
                    success=True,
                    output=result,
                    execution_time_ms=(end - start) * 1000,
                    stdout=stdout_capture.getvalue(),
                    stderr=stderr_capture.getvalue(),
                )
                
        except Exception as e:
            tb = traceback.format_exc()
            return KernelResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue() + "\n" + tb,
            )
    
    def execute_code(
        self,
        code: str,
        entry_function: str = "solve",
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
    ) -> KernelResult:
        """
        Triton 커널 코드를 실행합니다.
        
        코드를 임시 파일에 저장하고 import하여 실행합니다.
        이를 통해 Triton의 @jit 제약을 우회합니다.
        """
        if not self.gpu_available:
            return KernelResult(
                success=False,
                error="GPU is not available. Triton requires CUDA.",
                error_type="GPUNotAvailable",
            )
        
        args = args or []
        kwargs = kwargs or {}
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):
                
                # 코드를 임시 파일에 저장하고 모듈로 로드
                module, temp_file = self._create_temp_module(code)
                
                if not hasattr(module, entry_function):
                    return KernelResult(
                        success=False,
                        error=f"Entry function '{entry_function}' not found in code",
                        error_type="FunctionNotFound",
                        stdout=stdout_capture.getvalue(),
                        stderr=stderr_capture.getvalue(),
                    )
                
                func = getattr(module, entry_function)
                
                import time
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                
                return KernelResult(
                    success=True,
                    output=result,
                    execution_time_ms=(end - start) * 1000,
                    stdout=stdout_capture.getvalue(),
                    stderr=stderr_capture.getvalue(),
                )
                
        except SyntaxError as e:
            return KernelResult(
                success=False,
                error=f"Syntax error at line {e.lineno}: {e.msg}",
                error_type="SyntaxError",
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
            )
        except Exception as e:
            tb = traceback.format_exc()
            return KernelResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue() + "\n" + tb,
            )
    
    def validate_correctness(
        self,
        triton_output,
        reference_output,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> ValidationResult:
        """
        Validate Triton kernel output against reference.
        """
        try:
            import torch
            
            if not isinstance(triton_output, torch.Tensor):
                return ValidationResult(
                    passed=False,
                    error="triton_output is not a tensor",
                )
            
            if not isinstance(reference_output, torch.Tensor):
                return ValidationResult(
                    passed=False,
                    error="reference_output is not a tensor",
                )
            
            # Move to same device for comparison
            if triton_output.device != reference_output.device:
                reference_output = reference_output.to(triton_output.device)
            
            # Compute differences
            diff = torch.abs(triton_output - reference_output)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            # Check with tolerance
            passed = torch.allclose(triton_output, reference_output, rtol=rtol, atol=atol)
            
            # Count mismatches
            relative_diff = diff / (torch.abs(reference_output) + atol)
            mismatches = (relative_diff > rtol).sum().item()
            
            return ValidationResult(
                passed=passed,
                max_diff=max_diff,
                mean_diff=mean_diff,
                num_mismatches=int(mismatches),
                total_elements=triton_output.numel(),
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                error=str(e),
            )
    
    def benchmark(
        self,
        code: str,
        entry_function: str = "solve",
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
        warmup: int = 25,
        rep: int = 100,
        reference_fn: Callable = None,
    ) -> BenchmarkResult:
        """
        Benchmark Triton kernel performance.
        """
        if not self.gpu_available:
            return BenchmarkResult(
                success=False,
                error="GPU is not available",
            )
        
        args = args or []
        kwargs = kwargs or {}
        
        try:
            import torch
            
            # 코드를 임시 파일에 저장하고 모듈로 로드
            module, temp_file = self._create_temp_module(code)
            
            if not hasattr(module, entry_function):
                return BenchmarkResult(
                    success=False,
                    error=f"Entry function '{entry_function}' not found",
                )
            
            func = getattr(module, entry_function)
            
            ms_times = []
            
            # Warmup
            for _ in range(warmup):
                func(*args, **kwargs)
                torch.cuda.synchronize()
            
            # Benchmark
            for _ in range(rep):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                func(*args, **kwargs)
                end.record()
                
                torch.cuda.synchronize()
                ms_times.append(start.elapsed_time(end))
            
            import statistics
            mean_ms = statistics.mean(ms_times)
            std_ms = statistics.stdev(ms_times) if len(ms_times) > 1 else 0.0
            
            result = BenchmarkResult(
                success=True,
                mean_ms=mean_ms,
                std_ms=std_ms,
                min_ms=min(ms_times),
                max_ms=max(ms_times),
                num_runs=rep,
            )
            
            # Compare with reference if provided
            if reference_fn is not None:
                ref_times = []
                for _ in range(warmup):
                    reference_fn(*args, **kwargs)
                    torch.cuda.synchronize()
                
                for _ in range(rep):
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    
                    start.record()
                    reference_fn(*args, **kwargs)
                    end.record()
                    
                    torch.cuda.synchronize()
                    ref_times.append(start.elapsed_time(end))
                
                ref_mean = statistics.mean(ref_times)
                result.comparison = {
                    "reference_mean_ms": ref_mean,
                    "speedup": ref_mean / mean_ms if mean_ms > 0 else 0.0,
                }
            
            return result
            
        except Exception as e:
            tb = traceback.format_exc()
            return BenchmarkResult(
                success=False,
                error=f"{str(e)}\n{tb}",
            )
    
    def benchmark_from_file(
        self,
        file_path: str,
        entry_function: str = "solve",
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
        warmup: int = 25,
        rep: int = 100,
        reference_fn: Callable = None,
    ) -> BenchmarkResult:
        """파일에서 커널을 로드하여 벤치마크합니다."""
        if not self.gpu_available:
            return BenchmarkResult(
                success=False,
                error="GPU is not available",
            )
        
        args = args or []
        kwargs = kwargs or {}
        
        try:
            import torch
            
            module = self._load_module_from_file(file_path)
            
            if not hasattr(module, entry_function):
                return BenchmarkResult(
                    success=False,
                    error=f"Entry function '{entry_function}' not found in {file_path}",
                )
            
            func = getattr(module, entry_function)
            
            ms_times = []
            
            # Warmup
            for _ in range(warmup):
                func(*args, **kwargs)
                torch.cuda.synchronize()
            
            # Benchmark
            for _ in range(rep):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                func(*args, **kwargs)
                end.record()
                
                torch.cuda.synchronize()
                ms_times.append(start.elapsed_time(end))
            
            import statistics
            mean_ms = statistics.mean(ms_times)
            std_ms = statistics.stdev(ms_times) if len(ms_times) > 1 else 0.0
            
            result = BenchmarkResult(
                success=True,
                mean_ms=mean_ms,
                std_ms=std_ms,
                min_ms=min(ms_times),
                max_ms=max(ms_times),
                num_runs=rep,
            )
            
            # Compare with reference if provided
            if reference_fn is not None:
                ref_times = []
                for _ in range(warmup):
                    reference_fn(*args, **kwargs)
                    torch.cuda.synchronize()
                
                for _ in range(rep):
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    
                    start.record()
                    reference_fn(*args, **kwargs)
                    end.record()
                    
                    torch.cuda.synchronize()
                    ref_times.append(start.elapsed_time(end))
                
                ref_mean = statistics.mean(ref_times)
                result.comparison = {
                    "reference_mean_ms": ref_mean,
                    "speedup": ref_mean / mean_ms if mean_ms > 0 else 0.0,
                }
            
            return result
            
        except Exception as e:
            tb = traceback.format_exc()
            return BenchmarkResult(
                success=False,
                error=f"{str(e)}\n{tb}",
            )
