"""Tests for softmax kernel."""

import torch


def reference(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """PyTorch reference implementation of softmax."""
    return torch.nn.functional.softmax(x, dim=dim)


# Test case data generators
def _test_case_1_data():
    """2D tensor, last dimension."""
    return torch.randn(32, 128, device="cuda"), -1


def _test_case_2_data():
    """2D tensor, first dimension."""
    return torch.randn(32, 128, device="cuda"), 0


def _test_case_3_data():
    """3D tensor."""
    return torch.randn(4, 32, 128, device="cuda"), -1


def _test_case_4_data():
    """Small tensor."""
    return torch.randn(10, 5, device="cuda"), -1


# Actual pytest tests
def test_case_1(softmax_kernel):
    """Test 2D tensor, last dimension."""
    kernel_module, version = softmax_kernel
    x, dim = _test_case_1_data()

    expected = reference(x, dim)
    actual = kernel_module.solve(x, dim)

    assert torch.allclose(
        actual, expected, rtol=1e-5, atol=1e-8
    ), f"Softmax kernel v{version} failed for test_case_1"


def test_case_2(softmax_kernel):
    """Test 2D tensor, first dimension."""
    kernel_module, version = softmax_kernel
    x, dim = _test_case_2_data()

    expected = reference(x, dim)
    actual = kernel_module.solve(x, dim)

    assert torch.allclose(
        actual, expected, rtol=1e-5, atol=1e-8
    ), f"Softmax kernel v{version} failed for test_case_2"


def test_case_3(softmax_kernel):
    """Test 3D tensor."""
    kernel_module, version = softmax_kernel
    x, dim = _test_case_3_data()

    expected = reference(x, dim)
    actual = kernel_module.solve(x, dim)

    assert torch.allclose(
        actual, expected, rtol=1e-5, atol=1e-8
    ), f"Softmax kernel v{version} failed for test_case_3"


def test_case_4(softmax_kernel):
    """Test small tensor."""
    kernel_module, version = softmax_kernel
    x, dim = _test_case_4_data()

    expected = reference(x, dim)
    actual = kernel_module.solve(x, dim)

    assert torch.allclose(
        actual, expected, rtol=1e-5, atol=1e-8
    ), f"Softmax kernel v{version} failed for test_case_4"
