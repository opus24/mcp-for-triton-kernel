import torch


def reference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for vector add"""
    return A + B


# Test cases
def get_test_cases():
    return [
        # Small size
        {
            "args": [torch.randn(1024, device="cuda"), torch.randn(1024, device="cuda")],
            "kwargs": {},
        },
        # Medium size
        {
            "args": [
                torch.randn(1024 * 1024, device="cuda"),
                torch.randn(1024 * 1024, device="cuda"),
            ],
            "kwargs": {},
        },
        # Large size
        {
            "args": [
                torch.randn(1024 * 1024 * 16, device="cuda"),
                torch.randn(1024 * 1024 * 16, device="cuda"),
            ],
            "kwargs": {},
        },
    ]
