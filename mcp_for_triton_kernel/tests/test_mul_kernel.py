"""Tests for mul kernel."""

import torch


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of mul."""
    return torch.mul(a, b)


# Test case data generators
def _test_case_1_data():
    """Same shape tensors."""
    return torch.randn(1024, device="cuda"), torch.randn(1024, device="cuda")


def _test_case_2_data():
    """Larger tensors."""
    return torch.randn(10000, device="cuda"), torch.randn(10000, device="cuda")


def _test_case_3_data():
    """2D tensors."""
    return torch.randn(256, 256, device="cuda"), torch.randn(256, 256, device="cuda")


def _test_case_4_data():
    """Small tensors."""
    return torch.randn(10, device="cuda"), torch.randn(10, device="cuda")


# Actual pytest tests
def test_case_1(mul_kernel):
    """Test same shape tensors."""
    kernel_module, version = mul_kernel
    a, b = _test_case_1_data()

    expected = reference(a, b)
    actual = kernel_module.solve(a, b)

    assert torch.allclose(
        actual, expected, rtol=1e-5, atol=1e-8
    ), f"Mul kernel v{version} failed for test_case_1"


def test_case_2(mul_kernel):
    """Test larger tensors."""
    kernel_module, version = mul_kernel
    a, b = _test_case_2_data()

    expected = reference(a, b)
    actual = kernel_module.solve(a, b)

    assert torch.allclose(
        actual, expected, rtol=1e-5, atol=1e-8
    ), f"Mul kernel v{version} failed for test_case_2"


def test_case_3(mul_kernel):
    """Test 2D tensors."""
    kernel_module, version = mul_kernel
    a, b = _test_case_3_data()

    expected = reference(a, b)
    actual = kernel_module.solve(a, b)

    assert torch.allclose(
        actual, expected, rtol=1e-5, atol=1e-8
    ), f"Mul kernel v{version} failed for test_case_3"


def test_case_4(mul_kernel):
    """Test small tensors."""
    kernel_module, version = mul_kernel
    a, b = _test_case_4_data()

    expected = reference(a, b)
    actual = kernel_module.solve(a, b)

    assert torch.allclose(
        actual, expected, rtol=1e-5, atol=1e-8
    ), f"Mul kernel v{version} failed for test_case_4"
