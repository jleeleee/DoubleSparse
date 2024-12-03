import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaConfig, LlamaAttention,LlamaSdpaAttention, apply_rotary_pos_emb, repeat_kv
from typing import Optional, Tuple, Union
from transformers.cache_utils import Cache
import logging
from adaptive_softmax.sftm_pt import adaptive_softmax_batched

##
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LlamaAdaptiveTopKAttention(LlamaAttention):
    """Multi-headed attention with Top-K mechanism."""
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None, top_k: int = 256):
        super().__init__(config, layer_idx)
        self.top_k = top_k


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        if q_len > 1:
            return self.flash_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
           
            
        if position_embeddings is None:
            print(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        kv_seq_len = key_states.shape[-2]

        attn_weights = adaptive_softmax_batched(query_states/math.sqrt(self.head_dim), key_states, self.top_k) 

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
        elif q_len == kv_seq_len:
            boolean_mask = torch.tril(torch.ones(q_len, kv_seq_len, dtype=torch.bool, device=attn_weights.device))
            attention_mask = torch.zeros(q_len, kv_seq_len, dtype=torch.float16, device=attn_weights.device)
            attention_mask = attention_mask.masked_fill(boolean_mask == False, float('-inf')).view(1, 1, q_len, kv_seq_len)
            attn_weights = attn_weights + attention_mask

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


    @staticmethod
    def convert_llama_attention_to_adaptive_top_k(model: nn.Module, config: LlamaConfig, top_k: int = 256) -> nn.Module:
        # return model
        """Convert all LlamaAttention layers in the model to LlamaTopKAttention."""
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                model._modules[name] = LlamaAdaptiveTopKAttention.convert_llama_attention_to_adaptive_top_k(module, config, top_k=top_k)

            if isinstance(module, LlamaAttention):
                device = next(module.parameters()).device
                new_module = LlamaAdaptiveTopKAttention(config, module.layer_idx, top_k=top_k).half().to(device)
                new_module.load_state_dict(module.state_dict(), strict=True)
                model._modules[name] = new_module
                model._modules[name].flash_forward = module.forward


        return model
