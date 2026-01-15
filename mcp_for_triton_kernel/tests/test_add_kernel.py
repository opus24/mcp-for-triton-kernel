import torch


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of add."""
    return torch.add(a, b)


# Test cases
def test_case_1():
    """Same shape tensors."""
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    return a, b


def test_case_2():
    """Larger tensors."""
    a = torch.randn(10000, device="cuda")
    b = torch.randn(10000, device="cuda")
    return a, b


def test_case_3():
    """2D tensors."""
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    return a, b


def test_case_4():
    """Small tensors."""
    a = torch.randn(10, device="cuda")
    b = torch.randn(10, device="cuda")
    return a, b
