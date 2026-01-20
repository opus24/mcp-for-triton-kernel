import torch
import torch.nn.functional as F


def reference(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for softmax (last dimension)"""
    return F.softmax(x, dim=-1)


# Test cases
def get_test_cases():
    return [
        {"args": [torch.randn(128, 256, device="cuda")], "kwargs": {}},
        {"args": [torch.randn(1024, 1024, device="cuda")], "kwargs": {}},
        {"args": [torch.randn(4096, 4096, device="cuda")], "kwargs": {}},
        {"args": [torch.randn(32, 128, 512, device="cuda")], "kwargs": {}},
    ]
