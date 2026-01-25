from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1: torch.Tensor = x1 * cos - x2 * sin
    y2: torch.Tensor = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size: int = head_size
        assert rotary_dim == head_size
        inv_freq: torch.Tensor = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t: torch.Tensor = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs: torch.Tensor = torch.einsum("i,j -> ij", t, inv_freq)
        cos: torch.Tensor = freqs.cos()
        sin: torch.Tensor = freqs.sin()
        cache: torch.Tensor = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin: torch.Tensor = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query: torch.Tensor = apply_rotary_emb(query, cos, sin)
        key: torch.Tensor = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
) -> RotaryEmbedding:
    assert rope_scaling is None
    rotary_emb: RotaryEmbedding = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
