import rich.progress
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, Tuple, List, Union

from Cogformer.attention import precompute_freqs_cis, VanillaAttention, CogAttention, MyDynamicCache

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)  # force fp32
        return output * self.weight

class Expert(nn.Module):

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.gw = nn.Parameter(torch.empty((dim, hidden_dim))) # W_in
        self.pw = nn.Parameter(torch.empty((dim, hidden_dim))) # 
        self.ow = nn.Parameter(torch.empty((hidden_dim, dim)))

    def forward(self, x: Tensor) -> Tensor:
        g: Tensor = x @ self.gw
        x = g * F.silu(x @ self.pw)
        x = x @ self.ow
        return x

FFNRecipe = Union[None, int, Tuple[int, ...], List[int]]
def _normalize_ffn_recipe(ffn_recipe: FFNRecipe):
    if ffn_recipe is None:
        return ()
    elif isinstance(ffn_recipe, int):
        return tuple([1] * ffn_recipe)
    else:
        return tuple(ffn_recipe)

class FFN(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        shared_expert_recipe: FFNRecipe = None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.ffn_recipe = _normalize_ffn_recipe(shared_expert_recipe)

        fuel_quantity = sum(self.ffn_recipe)
        self.ffn_dim = hidden_dim // fuel_quantity
        
        self.shared_experts = nn.ModuleList()
        for fuel in self.ffn_recipe:
            self.shared_experts.append(Expert(dim, fuel * self.ffn_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.shared_experts[0](x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        head_dim: int,
        hidden_dim: int,
        norm_eps: float,
        n_layers: int,
        attn_type: str,
        shared_expert_recipe,
        **kwargs,
    ):
        super().__init__()
        self.layer_id = layer_id
        
        if 'vanilla' == attn_type:
            self.attention = VanillaAttention(layer_id, dim, n_heads, head_dim)
        elif 'np' == attn_type:
            if layer_id == 0 or layer_id == n_layers - 1:
                self.attention = VanillaAttention(layer_id, dim, n_heads, head_dim)
            else:
                self.attention = CogAttention(layer_id, dim, n_heads, head_dim)

        self.ffn = FFN(
            dim,
            hidden_dim,
            shared_expert_recipe,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Optional[Tensor] = None,
        past_key_values: Optional[MyDynamicCache] = None,
    ):
        attn_out, attn_w, present_key_value  = self.attention(
            self.attention_norm(x),
            freqs_cis,
            mask,
            past_key_values, 
        )
        h = x + attn_out
        out = self.ffn(self.ffn_norm(h))
        out = h + out

        return out, attn_w, present_key_value


class Transformer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        max_seq_length: int,
        dim: int,
        n_heads: int,
        head_dim: int,
        hidden_dim: int,
        norm_eps: float,
        attn_type: str,
        shared_expert_recipe,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = torch.nn.ModuleList()
        
        if attn_type is None:
            attn_type = bar_type
        
        for layer_id in range(n_layers):
            self.layers.append(TransformerBlock(
                layer_id=layer_id,
                dim=dim,
                n_heads=n_heads,
                head_dim=head_dim,
                hidden_dim=hidden_dim,
                norm_eps=norm_eps,
                n_layers=n_layers,
                attn_type=attn_type,
                shared_expert_recipe=shared_expert_recipe,
            ))
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            head_dim, max_seq_length * 2
        ))
        self.register_buffer("shortcut_bar", torch.zeros(4095)) # not used

    def update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        past_key_values: MyDynamicCache,
    ):
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        device = input_tensor.device

        bsz, sequence_length = input_tensor.shape[:2]
        target_length = past_seen_tokens + sequence_length

        causal_mask = torch.full(
            (bsz, 1, sequence_length, target_length), fill_value=True, dtype=torch.bool, device=device
        )
        
        mask_len = attention_mask.shape[-1]
        causal_mask[:, :, :, :mask_len] = attention_mask[:, :, -sequence_length:, ]

        return causal_mask
    
    def forward(
        self,
        tokens: Tensor,
        mask: Optional[Tensor] = None,
        past_key_values: Optional[MyDynamicCache] = None,
    ):
        h = self.embedding(tokens)
        all_attn_w = []
        
        if not self.training:
            mask = self.update_causal_mask(mask, h, past_key_values)   
            
        for layer in self.layers:
            h, attn_w, next_cache = layer(h, self.freqs_cis, mask, past_key_values)

            all_attn_w.append(attn_w)
            
        h = self.norm(h)
        
        return self.output(h), all_attn_w, next_cache

    @torch.inference_mode(mode=True)
    def generate(
        self,
        tokens: Tensor,
        mask: Tensor,
        generate_length: int,
    ):
        bsz, slen = tokens.shape
        
        past_key_values = MyDynamicCache()
        
        glen = min(generate_length, self.max_seq_length - slen + 1)
        gen_log_probs = torch.empty((bsz, glen, self.vocab_size), dtype=tokens.dtype)
        gen_tokens = torch.empty((bsz, glen), dtype=torch.long)

        input_ids = tokens
        
        for pos in rich.progress.track(range(glen), "Generating...", total=glen):
            logits, _, past_key_values = self.forward(input_ids, past_key_values=past_key_values, mask=mask)
            assert not torch.isnan(logits).any()
            logits = logits[:, -1:, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            gen_log_probs[:, pos: pos + 1, :] = log_probs
            input_ids = torch.argmax(log_probs, dim=-1)
            gen_tokens[:, pos: pos + 1] = input_ids
            
        return gen_log_probs, gen_tokens


def create_model(checkpoint_fp: str):
    with open(checkpoint_fp, 'rb') as f:
        state_dict = torch.load(f)
    model_cfg = state_dict["model_cfg"]
    model = Transformer(**model_cfg)
    model.load_state_dict(state_dict["model"])
    model.eval()
    return model