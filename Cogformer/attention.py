import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, BoolTensor
from typing import Tuple, Optional, List, Dict, Any
import math

class MyDynamicCache(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0 
        
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        return len(self.key_cache)
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[1]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=1)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=1)
            
    
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[1]

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(q: Tensor, k: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, q_)
    q_out = torch.view_as_real(q_ * freqs_cis).flatten(3)
    k_out = torch.view_as_real(k_ * freqs_cis).flatten(3)
    return q_out.type_as(q), k_out.type_as(k)

def scaled_dot_product_attention(query, key, value, attn_mask) -> torch.Tensor:

    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros_like(attn_mask, dtype=query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return attn_weight @ value, attn_weight

def generate_attn_masks(bos_masks: BoolTensor) -> BoolTensor:
    assert bos_masks.ndim == 2
    attn_masks = torch.cumsum(bos_masks.long(), dim=-1)
    attn_masks = (attn_masks[..., None] == attn_masks[..., None, :])
    attn_masks = torch.tril(attn_masks)
    return attn_masks[:, None]

class VanillaAttention(nn.Module):

    def __init__(self, layer_idx: int, dim: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.qw = nn.Parameter(torch.empty((dim, n_heads * head_dim)))
        self.kw = nn.Parameter(torch.empty((dim, n_heads * head_dim)))
        self.vw = nn.Parameter(torch.empty((dim, n_heads * head_dim)))
        self.ow = nn.Parameter(torch.empty((n_heads * head_dim, dim)))

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Optional[Tensor] = None,
        past_key_value: Optional[MyDynamicCache] = None,
    ) -> Tensor:

        bsz, slen, _ = x.shape
        q: Tensor = x @ self.qw
        k: Tensor = x @ self.kw
        v: Tensor = x @ self.vw
        
        q = q.view(bsz, -1, self.n_heads, self.head_dim)
        k = k.view(bsz, -1, self.n_heads, self.head_dim)
        v = v.view(bsz, -1, self.n_heads, self.head_dim)
        
        if self.training or past_key_value is None:
            q, k = apply_rotary_emb(q, k, freqs_cis[:slen])
        else:
            cached_seq_len = past_key_value.get_seq_length(layer_idx=self.layer_idx)
            q, k = apply_rotary_emb(q, k, freqs_cis[cached_seq_len : cached_seq_len + slen])

        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)
            
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # attn = F.scaled_dot_product_attention(q, k, v, mask)
        # attn_w = None
        attn, attn_w = scaled_dot_product_attention(q, k, v, mask)

        attn = attn.transpose(1, 2).reshape(bsz, -1, self.n_heads * self.head_dim)
        return attn @ self.ow, attn_w, past_key_value
 
class CogAttention(nn.Module):

    def __init__(self, layer_idx: int, dim: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.qw = nn.Parameter(torch.empty((dim, n_heads * head_dim)))
        self.kw = nn.Parameter(torch.empty((dim, n_heads * head_dim)))
        self.vw = nn.Parameter(torch.empty((dim, n_heads * head_dim)))
        self.ow = nn.Parameter(torch.empty((n_heads * head_dim, dim)))

        self.scale_factor = 1 / math.sqrt(head_dim)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor = None,
        past_key_value: Optional[MyDynamicCache] = None,
    ) -> Tensor:

        bsz, slen, _ = x.shape
        q: Tensor = x @ self.qw
        k: Tensor = x @ self.kw
        v: Tensor = x @ self.vw
    
        q = q.view(bsz, -1, self.n_heads, self.head_dim)
        k = k.view(bsz, -1, self.n_heads, self.head_dim)
        v = v.view(bsz, -1, self.n_heads, self.head_dim)
        
        if self.training or past_key_value is None:
            q, k = apply_rotary_emb(q, k, freqs_cis[:slen])
        else:
            cached_seq_len = past_key_value.get_seq_length(layer_idx=self.layer_idx)
            q, k = apply_rotary_emb(q, k, freqs_cis[cached_seq_len : cached_seq_len + slen])

        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)
            
        q = q.transpose(1, 2) * self.scale_factor
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        p = q @ k.transpose(-2, -1)
        abs_p = torch.abs(p)
        abs_p.masked_fill_(mask.logical_not(), -float("inf"))
        attn_w = torch.sign(p) * F.softmax(abs_p, dim=-1)
        attn = attn_w @ v
        
        attn = attn.transpose(1, 2).reshape(bsz, -1, self.n_heads * self.head_dim)
        return attn @ self.ow, attn_w, past_key_value