"""Utility for running Triton kernels dynamically."""

import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional
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
    """Runner for dynamically executing Triton kernel code."""
    
    def __init__(self):
        self._check_gpu_available()
    
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
    
    def execute_code(
        self,
        code: str,
        entry_function: str = "solve",
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
    ) -> KernelResult:
        """
        Execute Triton kernel code dynamically.
        
        Args:
            code: Complete Python code containing the kernel and solve function
            entry_function: Name of the function to call (default: "solve")
            args: Positional arguments for the entry function
            kwargs: Keyword arguments for the entry function
        
        Returns:
            KernelResult with output or error information
        """
        if not self.gpu_available:
            return KernelResult(
                success=False,
                error="GPU is not available. Triton requires CUDA.",
                error_type="GPUNotAvailable",
            )
        
        args = args or []
        kwargs = kwargs or {}
        
        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Create isolated namespace
        namespace = {}
        
        try:
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):
                # Execute the code to define functions
                exec(code, namespace)
                
                # Get the entry function
                if entry_function not in namespace:
                    return KernelResult(
                        success=False,
                        error=f"Entry function '{entry_function}' not found in code",
                        error_type="FunctionNotFound",
                        stdout=stdout_capture.getvalue(),
                        stderr=stderr_capture.getvalue(),
                    )
                
                func = namespace[entry_function]
                
                # Execute the function
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
        
        Args:
            triton_output: Output tensor from Triton kernel
            reference_output: Reference output (usually from PyTorch)
            rtol: Relative tolerance
            atol: Absolute tolerance
        
        Returns:
            ValidationResult with comparison statistics
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
        reference_fn=None,
    ) -> BenchmarkResult:
        """
        Benchmark Triton kernel performance.
        
        Args:
            code: Complete Python code containing the kernel
            entry_function: Name of the function to call
            args: Positional arguments
            kwargs: Keyword arguments
            warmup: Number of warmup runs
            rep: Number of timed runs
            reference_fn: Optional reference function for comparison
        
        Returns:
            BenchmarkResult with timing statistics
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
            import triton
            
            # Execute code to get the function
            namespace = {}
            exec(code, namespace)
            
            if entry_function not in namespace:
                return BenchmarkResult(
                    success=False,
                    error=f"Entry function '{entry_function}' not found",
                )
            
            func = namespace[entry_function]
            
            # Use triton's benchmarking utility
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
            return BenchmarkResult(
                success=False,
                error=str(e),
            )

