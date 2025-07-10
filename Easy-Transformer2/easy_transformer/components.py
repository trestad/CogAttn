from mimetypes import init
from typing import Callable, Union, List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from Cogformer.attention import apply_rotary_emb, precompute_freqs_cis

from functools import *

from easy_transformer.hook_points import HookPoint
from easy_transformer.utils import gelu_new, solu, gelu_fast
from easy_transformer.EasyTransformerConfig import EasyTransformerConfig

from fancy_einsum import einsum

from easy_transformer.caching import (
    EasyTransformerKeyValueCache,
    EasyTransformerKeyValueCacheEntry,
)

# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty(self.cfg.d_vocab, self.cfg.d_model))

    def forward(self, tokens):
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        return self.W_E[tokens, :]  # Shape [batch pos d_model]


class Unembed(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):

        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        # Note that there's a separate variable for d_vocab_out and d_vocab (the input vocab size). For language tasks these are always the same, but for algorithmic tasks we may want them to be different.
        self.W_U = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_vocab_out))
        self.b_U = nn.Parameter(torch.zeros(self.cfg.d_vocab_out))

    def forward(self, residual):
        return (
            einsum(
                "batch pos d_model, d_model vocab -> batch pos vocab",
                residual,
                self.W_U,
            )
            + self.b_U
        )  # [batch, pos, d_vocab]


class RMSNorm(nn.Module):
    def __init__(
        self, cfg: Union[Dict, EasyTransformerConfig], length: Optional[int] = None
    ):

        """
        RMSNorm - LayerNorm without the centering and bias (RMS = Root Mean Square)

        length (Optional[int]): If the dimension of the RMSNorm. If not provided, assumed to be d_model
        """
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps
        if length is None:
            self.length = self.cfg.d_model
        else:
            self.length = length

        self.w = nn.Parameter(torch.ones(self.length))

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(self, x):
        scale = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )  # [batch, pos, 1]
        x = self.hook_normalized(x / scale)  # [batch, pos, length]
        return x * self.w
    
# Attention
class Attention(nn.Module):
    def __init__(
        self, cfg: Union[Dict, EasyTransformerConfig], attn_type="global", layer_id=None
    ):
        """Attention Block - params have shape [head_index, d_model, d_head] (or [head_index, d_head, d_model] for W_O) and multiply on the right. attn_scores refers to query key dot product immediately before attention softmax

        Convention: All attention pattern-style matrices have shape [batch, head_index, query_pos, key_pos]

        Args:
            cfg (Union[Dict, EasyTransformerConfig]): Config
            attn_type (str, optional): "global" or "local", used by GPT-Neo. Local attention means the model can only attend back cfg.window_size tokens (here, 256). Not used by any other model at the moment. Defaults to "global".
            layer_id (int, optional): The index of the current layer. Used by the Mistal models (labelled here as stanford-gpt2) to scale down attention scores pre softmax for numerical stability reasons by 1/(layer_id+1). Defaults to None.
        """
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_Q = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head)
        )
        self.W_K = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head)
        )
        self.W_V = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head)
        )
        self.W_O = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model)
        )
        self.b_Q = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head))
        self.b_K = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head))
        self.b_V = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head))
        self.b_O = nn.Parameter(torch.zeros(self.cfg.d_model))

        self.attn_type = attn_type
        # Create a max_ctx x max_ctx mask, with True iff that query position
        # can attend to that key position (query is first axis, key is second axis)
        causal_mask = torch.tril(torch.ones((self.cfg.n_ctx, self.cfg.n_ctx)).bool())
        
        self.register_buffer("mask", causal_mask)

        self.register_buffer("IGNORE", torch.tensor(-1e5))

        self.register_buffer('freqs_cis', precompute_freqs_cis(self.cfg.d_head, self.cfg.n_ctx * 2))

        self.layer_id = layer_id

        # attn_scale is a constant that we divide the attention scores by pre-softmax. I'm not entirely sure why it matters, but it's probably a mix of softmax not being scale invariant and numerical stability?
        
        self.attn_scale = np.sqrt(self.cfg.d_head)
       
        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]

        # Added by Arthur; all are ResidPre, but we want finer access
        assert self.cfg.positional_embedding_type in [
            "standard",
            "rotary",
        ], f"q_input hooks only support standard and rotary positional embeddings, not {self.cfg.positional_embedding_type}"
        self.hook_q_input = HookPoint()  # [batch, pos, d_model]
        self.hook_k_input = HookPoint()  # [batch, pos, d_model]
        self.hook_v_input = HookPoint()  # [batch, pos, d_model]

        self.hook_z = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_attn = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_result = HookPoint()  # [batch, head_index, head_index, d_model]


    def forward(
        self,
        resid_pre: torch.Tensor,  # goddamn normalized thing
        shortformer_pos_embed: Optional[torch.Tensor] = None,
        past_kv_cache_entry: Optional[EasyTransformerKeyValueCacheEntry] = None,
    ):
        """
        shortformer_pos_embed is only used if self.cfg.positional_embedding_type == "shortformer", else defaults to None and is irrelevant. See EasyTransformerConfig for more details
        past_kv_cache_entry is an optional entry of past keys and values for this layer, only relevant if generating text. Defaults to None

        """

        # Normal attention
        q = self.hook_q(
            einsum(
                "batch pos d_model, head_index d_model d_head \
                -> batch pos head_index d_head",
                resid_pre,
                self.W_Q,
            )
            + self.b_Q
        )  # [batch, pos, head_index, d_head]
        k = self.hook_k(
            einsum(
                "batch pos d_model, head_index d_model d_head \
                -> batch pos head_index d_head",
                resid_pre,
                self.W_K,
            )
            + self.b_K
        )  # [batch, pos, head_index, d_head]
           
        v = self.hook_v(
            einsum(
                "batch pos d_model, head_index d_model d_head \
                -> batch pos head_index d_head",
                resid_pre,
                self.W_V,
            )
            + self.b_V
        )  # [batch, pos, head_index, d_head]

        # if past_kv_cache_entry is not None:
        assert past_kv_cache_entry is None, "past_kv_cache_entry is not None"
        # Appends the new keys and values to the cached values, and automatically updates the cache
        # kv_cache_pos_offset = past_kv_cache_entry.past_keys.size(1)
        # k, v = past_kv_cache_entry.append(k, v)
        # else:
        # Not using a cache
        kv_cache_pos_offset = 0

        q, k = apply_rotary_emb(q, k, self.freqs_cis[:q.shape[1]])

        attn_scores = (
            einsum(
                "batch query_pos head_index d_head, \
                batch key_pos head_index d_head \
                -> batch head_index query_pos key_pos",
                q,
                k,
            )
            / self.attn_scale
        )  # [batch, head_index, query_pos, key_pos]
        
        if self.attn_type == 'vanilla':
            attn_scores = self.apply_causal_mask(
                attn_scores, kv_cache_pos_offset
            )  # [batch, head_index, query_pos, key_pos]
        
            attn_scores = self.hook_attn_scores(attn_scores)
        
            attn_matrix = self.hook_attn(
                F.softmax(attn_scores, dim=-1)
            )  # [batch, head_index, query_pos, key_pos]
        elif self.attn_type == 'np':
            attn_scores = self.apply_causal_mask_np(
                attn_scores, kv_cache_pos_offset
            )  # [batch, head_index, query_pos, key_pos]
        
            attn_scores = self.hook_attn_scores(attn_scores)
            max_v_all = torch.abs(attn_scores).max(-1, keepdim=True).values
            qk_sign = torch.sign(attn_scores)
            attn_w = qk_sign * torch.exp(qk_sign * attn_scores - max_v_all) # mask掉的地方值是0，qk_sign也是0，不用再mask了
            sum_abs_attn_w = torch.abs(attn_w).sum(-1, keepdim=True) + 1e-6
            attn_matrix = self.hook_attn(attn_w / sum_abs_attn_w)
        
        z = self.hook_z(
            einsum(
                "batch key_pos head_index d_head, \
                batch head_index query_pos key_pos -> \
                batch query_pos head_index d_head",
                v,
                attn_matrix,
            )
        )  # [batch, pos, head_index, d_head]
        
        if not self.cfg.use_attn_result:
            out = (
                (
                    einsum(
                        "batch pos head_index d_head, \
                        head_index d_head d_model -> \
                        batch pos d_model",
                        z,
                        self.W_O,
                    )
                )
                + self.b_O
            )  # [batch, pos, d_model]
        else:
            # Explicitly calculate the attention result so it can be accessed by a hook
            # This is off by default because it can easily eat through your GPU memory.
            result = self.hook_result(
                einsum(
                    "batch pos head_index d_head, \
                        head_index d_head d_model -> \
                        batch pos head_index d_model",
                    z,
                    self.W_O,
                )
            )  # [batch, pos, head_index, d_model]
            out = (
                einops.reduce(
                    result, "batch position index model->batch position model", "sum"
                )
                + self.b_O
            )  # [batch, pos, d_model]
        return out

    def apply_causal_mask(self, attn_scores, past_kv_pos_offset):
        # The query context length is the number of positions we take queries from - if not using a past_kv_cache this is just the context length (for the current prompt), but if we're caching it's just a single token.
        query_ctx_length = attn_scores.size(-2)
        # The key context length is the number of positions in the past - this includes all positions in the cache
        # If not caching, query_ctx_length == key_ctx_length
        key_ctx_length = attn_scores.size(-1)

        assert (
            query_ctx_length + past_kv_pos_offset == key_ctx_length
        ), f"query_ctx_length {query_ctx_length} + past_kv_pos_offset {past_kv_pos_offset} != key_ctx_length {key_ctx_length} - you likely have a bug."
        return torch.where(
            self.mask[
                past_kv_pos_offset : past_kv_pos_offset + query_ctx_length,
                :key_ctx_length,
            ],
            attn_scores,
            self.IGNORE,
        )
        
    def apply_causal_mask_np(self, attn_scores, past_kv_pos_offset):
        # The query context length is the number of positions we take queries from - if not using a past_kv_cache this is just the context length (for the current prompt), but if we're caching it's just a single token.
        query_ctx_length = attn_scores.size(-2)
        # The key context length is the number of positions in the past - this includes all positions in the cache
        # If not caching, query_ctx_length == key_ctx_length
        key_ctx_length = attn_scores.size(-1)

        assert (
            query_ctx_length + past_kv_pos_offset == key_ctx_length
        ), f"query_ctx_length {query_ctx_length} + past_kv_pos_offset {past_kv_pos_offset} != key_ctx_length {key_ctx_length} - you likely have a bug."
        return torch.where(
            self.mask[
                past_kv_pos_offset : past_kv_pos_offset + query_ctx_length,
                :key_ctx_length,
            ],
            attn_scores,
            0,
        )

# MLP Layers
class MLP(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.gw = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_mlp))
        self.pw = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty(self.cfg.d_mlp, self.cfg.d_model))

    def forward(self, x):
        g = x @ self.gw
        x = g * F.silu(x @ self.pw)
        x = x @ self.W_out
        return x


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig], block_index):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        
        self.ln1 = RMSNorm(cfg)  # moved here by Arthur
        
        self.ln2 = RMSNorm(cfg)
        
        assert self.cfg.attn_types is not None
        attn_type = self.cfg.attn_types[block_index]

        self.attn = Attention(cfg, attn_type, block_index)
        self.mlp = MLP(cfg)

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(
        self,
        resid_pre: torch.Tensor,
        shortformer_pos_embed: Optional[torch.Tensor] = None,
        past_kv_cache_entry: Optional[EasyTransformerKeyValueCacheEntry] = None,
    ):
        """A single Transformer block.

        Args:
            resid_pre (torch.Tensor): The residual stream - shape [batch, pos, d_model]
            cache (EasyTransformerKeyValueCache): A cache of previous keys and values, used only when generating text. Defaults to None.
            shortformer_pos_embed (torch.Tensor, optional): Only used for positional_embeddings_type == "shortformer". The positional embeddings. See EasyTransformerConfig for details. Defaults to None.

        Returns:
            _type_: _description_
        """
        resid_pre = self.hook_resid_pre(resid_pre)  # [batch, pos, d_model]
        # normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.hook_attn_out(
            self.attn(
                self.ln1(resid_pre),  # edited by Arthur from normalized ... so we can go headwise
                shortformer_pos_embed=shortformer_pos_embed,
                past_kv_cache_entry=past_kv_cache_entry,
            )
        )  # [batch, pos, d_model]
        
        resid_mid = self.hook_resid_mid(
            resid_pre + attn_out
        )  # [batch, pos, d_model]

        mlp_out = self.hook_mlp_out(
            self.mlp(self.ln2(resid_mid))
        )  # [batch, pos, d_model]
        
        resid_post = self.hook_resid_post(
            resid_mid + mlp_out
        )  # [batch, pos, d_model]
        
        return resid_post
