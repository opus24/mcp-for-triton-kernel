import torch
import torch.nn.functional as F


def reference(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float = None
) -> torch.Tensor:
    """PyTorch reference implementation for scaled dot-product attention"""
    if scale is None:
        scale = 1.0 / (Q.shape[-1] ** 0.5)
    return F.scaled_dot_product_attention(Q, K, V, scale=scale)


# Test cases
def get_test_cases():
    return [
        # Small: B=2, H=4, S=64, D=32
        {
            "args": [
                torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.float16),
                torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.float16),
                torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.float16),
            ],
            "kwargs": {},
        },
        # Medium: B=2, H=8, S=128, D=64
        {
            "args": [
                torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16),
                torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16),
                torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16),
            ],
            "kwargs": {},
        },
        # Large: B=1, H=12, S=512, D=64
        {
            "args": [
                torch.randn(1, 12, 512, 64, device="cuda", dtype=torch.float16),
                torch.randn(1, 12, 512, 64, device="cuda", dtype=torch.float16),
                torch.randn(1, 12, 512, 64, device="cuda", dtype=torch.float16),
            ],
            "kwargs": {},
        },
    ]
