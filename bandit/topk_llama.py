import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb, repeat_kv
from typing import Optional, Tuple, Union
from transformers.cache_utils import Cache


class TopKAttention(nn.Module):
    def __init__(self, config, k=256, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.k = k  # Top K similarities to consider
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape and transpose for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings if needed
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Compute attention scores
        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores += attention_mask

        # Select top K attention scores and corresponding values
        topk_scores, topk_indices = torch.topk(attn_scores, self.k, dim=-1)
        # Create a mask for top K indices
        mask = torch.zeros_like(attn_scores).scatter_(-1, topk_indices, 1.0)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Compute softmax over the top K scores
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute the attention output
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)

        # Final linear projection
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def replace_with_topk_attention(model, config, k=256):
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            device = next(module.parameters()).device
            new_module = TopKAttention(config, k=k, layer_idx=module.layer_idx).to(device)
            new_module.load_state_dict(module.state_dict(), strict=False)
            setattr(model, name, new_module)
    return model
