import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaConfig, LlamaAttention,LlamaSdpaAttention, apply_rotary_pos_emb, repeat_kv
from typing import Optional, Tuple, Union
from transformers.cache_utils import Cache
import logging

##
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple

@staticmethod
def convert_llama_attention_to_top_k(model: nn.Module, config: LlamaConfig, top_k: int = 256) -> nn.Module:
    # return model
    """Convert all LlamaAttention layers in the model to LlamaTopKAttention."""
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = LlamaTopKAttention.convert_llama_attention_to_top_k(module, config, top_k=top_k)

        if isinstance(module, LlamaAttention):
            device = next(module.parameters()).device
            new_module = LlamaTopKAttention(config, module.layer_idx, top_k=top_k).half().to(device)
            new_module.load_state_dict(module.state_dict(), strict=True)
            model._modules[name] = new_module
            model._modules[name].flash_forward = module.forward


    return model
