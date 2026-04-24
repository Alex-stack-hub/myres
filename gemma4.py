# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The vLLM team.
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gemma 4 model implementation for vLLM."""
from collections.abc import Iterable
from itertools import islice
import regex as re
import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors
from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

# ====================== 全模块Dump配置 ======================
import os
DUMP_PATH = "./gemma4_full_dump"  # 可改成绝对路径
os.makedirs(DUMP_PATH, exist_ok=True)
logger.info(f"[Gemma4 Full Dump] 已开启，保存路径: {DUMP_PATH}")

def save_tensor(tensor: torch.Tensor, name: str, layer_idx: int = -1, module: str = "global"):
    """全模块Dump工具：单卡保存，自动转CPU，文件名清晰标记"""
    if get_tensor_model_parallel_rank() != 0:
        return
    if tensor is None:
        return
    
    # 构建文件名：layer{idx}_{module}_{name}.pt
    if layer_idx >= 0:
        save_name = f"layer{layer_idx}_{module}_{name}.pt"
    else:
        save_name = f"global_{module}_{name}.pt"
    
    save_path = os.path.join(DUMP_PATH, save_name)
    try:
        torch.save(tensor.detach().cpu(), save_path)
    except Exception as e:
        logger.error(f"[Dump Error] {save_path}: {e}")

# ====================== 以下是原有代码 + 全模块Dump插入 ======================
def _get_text_config(config):
    if hasattr(config, "text_config"):
        return config.text_config
    return config


class Gemma4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_idx: int = -1,  # 新增：接收层索引
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma4 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_act` and `hidden_activation` to "
                "`gelu_pytorch_tanh`."
            )
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MLP全节点Dump
        save_tensor(x, "input", self.layer_idx, "mlp")
        gate_up, _ = self.gate_up_proj(x)
        save_tensor(gate_up, "gate_up_proj_output", self.layer_idx, "mlp")
        x = self.act_fn(gate_up)
        save_tensor(x, "act_fn_output", self.layer_idx, "mlp")
        x, _ = self.down_proj(x)
        save_tensor(x, "down_proj_output", self.layer_idx, "mlp")
        return x


class Gemma4Router(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_idx: int = -1,  # 新增：接收层索引
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps, has_weight=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.register_buffer(
            "root_size",
            torch.tensor(self.hidden_size**-0.5),
            persistent=False,
        )
        self.proj = GateLinear(
            self.hidden_size,
            config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Router全节点Dump
        save_tensor(x, "input", self.layer_idx, "router")
        x = self.norm(x)
        save_tensor(x, "after_norm", self.layer_idx, "router")
        x = x * self.root_size.to(x.dtype)
        save_tensor(x, "after_root_size", self.layer_idx, "router")
        x = x * self.scale.to(x.dtype)
        save_tensor(x, "after_scale", self.layer_idx, "router")
        router_logits, _ = self.proj(x)
        save_tensor(router_logits, "router_logits", self.layer_idx, "router")
        return router_logits


class Gemma4MoE(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_idx: int = -1,  # 新增：接收层索引
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))
        per_expert_scale = self.per_expert_scale

        def routing_function(
            hidden_states: torch.Tensor,
            gating_output: torch.Tensor,
            topk: int,
            renormalize: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            _, topk_ids = torch.topk(gating_output, k=topk, dim=-1)
            router_probabilities = torch.nn.functional.softmax(gating_output, dim=-1)
            indicator = torch.nn.functional.one_hot(
                topk_ids, num_classes=gating_output.size(-1)
            ).sum(dim=-2)
            gate_weights = indicator * router_probabilities
            renorm_factor = torch.sum(gate_weights, dim=-1, keepdim=True)
            renorm_factor = torch.where(renorm_factor > 0.0, renorm_factor, 1.0)
            dispatch_weights = gate_weights / renorm_factor
            topk_weights = dispatch_weights.gather(1, topk_ids)
            expert_scales = per_expert_scale[topk_ids].to(topk_weights.dtype)
            topk_weights = topk_weights * expert_scales
            return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.top_k_experts,
            hidden_size=config.hidden_size,
            intermediate_size=getattr(
                config,
                "moe_intermediate_size",
                getattr(config, "expert_intermediate_size", None),
            ),
            reduce_results=True,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            custom_routing_function=routing_function,
            activation="gelu",
        )

    def forward(self, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        # MoE全节点Dump
        save_tensor(x, "input", self.layer_idx, "moe")
        save_tensor(router_logits, "router_logits_input", self.layer_idx, "moe")
        out = self.experts(x, router_logits)
        save_tensor(out, "experts_output", self.layer_idx, "moe")
        return out


class Gemma4Attention(nn.Module):
    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        use_k_eq_v: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        attn_logits_soft_cap: float | None = None,
        prefix: str = "",
        layer_idx: int = -1,  # 新增：接收层索引
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = hidden_size
        self.use_k_eq_v = use_k_eq_v
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = 1.0

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)

        layer_idx_extract = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx_extract]
        self.is_sliding = layer_type == "sliding_attention"
        sliding_window = config.sliding_window if self.is_sliding else None

        if layer_type in config.rope_parameters:
            rope_parameters = dict(config.rope_parameters[layer_type])
        else:
            rope_parameters = dict(config.rope_parameters.copy())
            if self.is_sliding:
                rope_parameters["rope_theta"] = getattr(
                    config, "rope_local_base_freq", 10000.0
                )

        kv_sharing_target_layer_name = None
        self.is_kv_shared_layer = False
        num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)
        if num_kv_shared_layers > 0:
            first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared_layers
            if layer_idx_extract >= first_kv_shared_layer_idx:
                self.is_kv_shared_layer = True
                prev_layers = config.layer_types[:first_kv_shared_layer_idx]
                current_layer_type = config.layer_types[layer_idx_extract]
                kv_shared_layer_index = (
                    len(prev_layers) - 1 - prev_layers[::-1].index(current_layer_type)
                )
                if kv_shared_layer_index >= 0:
                    if ".layers." in prefix:
                        param_name_before_layers = prefix.split(".layers.")[0]
                    else:
                        raise ValueError(
                            "Unexpected prefix format for Gemma4Attention: "
                            f"'{prefix}'. Expected to contain '.layers.'."
                        )
                    kv_sharing_target_layer_name = (
                        f"{param_name_before_layers}.layers."
                        f"{kv_shared_layer_index}.self_attn.attn"
                    )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=attn_logits_soft_cap,
            per_layer_sliding_window=sliding_window,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Attention全节点Dump
        save_tensor(hidden_states, "input", self.layer_idx, "attn")
        save_tensor(positions, "positions", self.layer_idx, "attn")
        
        qkv, _ = self.qkv_proj(hidden_states)
        save_tensor(qkv, "qkv_proj_output", self.layer_idx, "attn")
        
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        save_tensor(q, "q_split", self.layer_idx, "attn")
        save_tensor(k, "k_split", self.layer_idx, "attn")
        save_tensor(v, "v_split", self.layer_idx, "attn")

        # Q norm
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)
        save_tensor(q, "q_after_norm", self.layer_idx, "attn")

        if not self.is_kv_shared_layer:
            # K norm + RoPE
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k = self.k_norm(k)
            k = k.flatten(-2, -1)
            save_tensor(k, "k_after_norm", self.layer_idx, "attn")
            
            q, k = self.rotary_emb(positions, q, k)
            save_tensor(q, "q_after_rope", self.layer_idx, "attn")
            save_tensor(k, "k_after_rope", self.layer_idx, "attn")

            # V norm
            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = self.v_norm(v)
            v = v.flatten(-2, -1)
            save_tensor(v, "v_after_norm", self.layer_idx, "attn")
        else:
            # Shared: only RoPE Q
            q = self.rotary_emb(positions, q, k)[0]
            save_tensor(q, "q_after_rope_shared", self.layer_idx, "attn")

        attn_output = self.attn(q, k, v)
        save_tensor(attn_output, "attn_core_output", self.layer_idx, "attn")
        
        output, _ = self.o_proj(attn_output)
        save_tensor(output, "o_proj_output", self.layer_idx, "attn")
        return output


class Gemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", 0
        )
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx

        layer_type = config.layer_types[layer_idx]
        self.is_full_attention = layer_type == "full_attention"
        if self.is_full_attention:
            head_dim = getattr(config, "global_head_dim", config.head_dim)
        else:
            head_dim = config.head_dim

        use_k_eq_v = self.is_full_attention and getattr(
            config, "attention_k_eq_v", False
        )

        if use_k_eq_v:
            num_kv_heads = getattr(
                config, "num_global_key_value_heads", config.num_key_value_heads
            )
        else:
            num_kv_heads = config.num_key_value_heads

        self.self_attn = Gemma4Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            use_k_eq_v=use_k_eq_v,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=getattr(config, "attn_logit_softcapping", None),
            prefix=f"{prefix}.self_attn",
            layer_idx=self.layer_idx,  # 传递层索引给Attention
        )

        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = (
            getattr(config, "use_double_wide_mlp", False) and is_kv_shared_layer
        )
        layer_intermediate_size = config.intermediate_size * (
            2 if use_double_wide_mlp else 1
        )

        self.mlp = Gemma4MLP(
            hidden_size=self.hidden_size,
            intermediate_size=layer_intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            layer_idx=self.layer_idx,  # 传递层索引给MLP
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.enable_moe_block = getattr(config, "enable_moe_block", False) or getattr(
            config, "use_second_mlp_block", False
        )
        if self.enable_moe_block:
            self.router = Gemma4Router(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.router",
                layer_idx=self.layer_idx,  # 传递层索引给Router
            )
            self.moe = Gemma4MoE(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.moe",
                layer_idx=self.layer_idx,  # 传递层索引给MoE
            )
            self.post_feedforward_layernorm_1 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.router = None
            self.moe = None
            self.post_feedforward_layernorm_1 = None
            self.post_feedforward_layernorm_2 = None
            self.pre_feedforward_layernorm_2 = None

        if (
            self.hidden_size_per_layer_input is not None
            and self.hidden_size_per_layer_input > 0
        ):
            self.per_layer_input_gate = ReplicatedLinear(
                self.hidden_size,
                self.hidden_size_per_layer_input,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.per_layer_input_gate",
                return_bias=False,
            )
            self.per_layer_projection = ReplicatedLinear(
                self.hidden_size_per_layer_input,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.per_layer_projection",
                return_bias=False,
            )
            self.post_per_layer_input_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        per_layer_input: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # DecoderLayer全节点Dump
        save_tensor(hidden_states, "input", self.layer_idx, "decoder")
        save_tensor(positions, "positions", self.layer_idx, "decoder")
        if residual is not None:
            save_tensor(residual, "residual_input", self.layer_idx, "decoder")
        if per_layer_input is not None:
            save_tensor(per_layer_input, "per_layer_input", self.layer_idx, "decoder")

        # Attention分支
        residual = hidden_states
        hidden_states = self.input_layernorm(residual)
        save_tensor(hidden_states, "after_input_layernorm", self.layer_idx, "decoder")
        
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )
        save_tensor(hidden_states, "after_self_attn", self.layer_idx, "decoder")
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        save_tensor(hidden_states, "after_post_attn_layernorm", self.layer_idx, "decoder")
        
        hidden_states = hidden_states + residual
        save_tensor(hidden_states, "after_attn_residual_add", self.layer_idx, "decoder")
        
        residual = hidden_states

        # MLP/MoE分支
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        save_tensor(hidden_states, "after_pre_ff_layernorm", self.layer_idx, "decoder")
        
        hidden_states = self.mlp(hidden_states)
        save_tensor(hidden_states, "after_mlp", self.layer_idx, "decoder")

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
            save_tensor(hidden_states_1, "after_post_ff_layernorm_1", self.layer_idx, "decoder")
            
            router_logits = self.router(residual)
            hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
            save_tensor(hidden_states_2, "after_pre_ff_layernorm_2", self.layer_idx, "decoder")
            
            hidden_states_2 = self.moe(hidden_states_2, router_logits)
            save_tensor(hidden_states_2, "after_moe", self.layer_idx, "decoder")
            
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
            save_tensor(hidden_states_2, "after_post_ff_layernorm_2", self.layer_idx, "decoder")
            
            hidden_states = hidden_states_1 + hidden_states_2
            save_tensor(hidden_states, "after_mlp_moe_combine", self.layer_idx, "decoder")

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        save_tensor(hidden_states, "after_post_ff_layernorm", self.layer_idx, "decoder")
        
        hidden_states = hidden_states + residual
        save_tensor(hidden_states, "after_ff_residual_add", self.layer_idx, "decoder")

        # PLE分支
        if per_layer_input is not None and self.per_layer_input_gate is not None:
            gate = self.per_layer_input_gate(hidden_states)
            save_tensor(gate, "ple_gate", self.layer_idx, "decoder")
            
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            save_tensor(gate, "ple_gate_gelu", self.layer_idx, "decoder")
            
            gated_per_layer = gate * per_layer_input
            save_tensor(gated_per_layer, "ple_gated_per_layer", self.layer_idx, "decoder")
            
            per_layer_contribution = self.per_layer_projection(gated_per_layer)
            save_tensor(per_layer_contribution, "ple_per_layer_proj", self.layer_idx, "decoder")
            
            per_layer_contribution = self.post_per_layer_input_norm(
                per_layer_contribution
            )
            save_tensor(per_layer_contribution, "ple_post_norm", self.layer_idx, "decoder")
            
            hidden_states = hidden_states + per_layer_contribution
            save_tensor(hidden_states, "after_ple_add", self.layer_idx, "decoder")

        # Layer scalar
        hidden_states = hidden_states * self.layer_scalar
        save_tensor(hidden_states, "output", self.layer_idx, "decoder")
        
        # 第一层额外重点标记
        if self.layer_idx == 0:
            save_tensor(hidden_states, "LAYER0_FOCUS_OUTPUT", self.layer_idx, "decoder")

        return hidden_states, None


@support_torch_compile
class Gemma4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _get_text_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", 0
        )
        self.vocab_size_per_layer_input = getattr(
            config, "vocab_size_per_layer_input", config.vocab_size
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        if (
            self.hidden_size_per_layer_input is not None
            and self.hidden_size_per_layer_input > 0
        ):
            total_ple_dim = self.hidden_size_per_layer_input * config.num_hidden_layers
            self.embed_tokens_per_layer = VocabParallelEmbedding(
                self.vocab_size_per_layer_input,
                total_ple_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens_per_layer",
            )
            self.register_buffer(
                "embed_scale_per_layer",
                torch.tensor(self.hidden_size_per_layer_input**0.5),
                persistent=False,
            )
            self.per_layer_model_projection = ColumnParallelLinear(
                config.hidden_size,
                total_ple_dim,
                bias=False,
                gather_output=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.per_layer_model_projection",
            )
            self.per_layer_projection_norm = RMSNorm(
                self.hidden_size_per_layer_input,
                eps=config.rms_norm_eps,
            )
            self.register_buffer(
                "per_layer_input_scale",
                torch.rsqrt(torch.tensor(2.0)),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_projection_scale",
                torch.tensor(config.hidden_size**-0.5),
                persistent=False,
            )
        else:
            self.embed_tokens_per_layer = None
            self.embed_scale_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None
            self.per_layer_input_scale = None
            self.per_layer_projection_scale = None

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Gemma4DecoderLayer(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.register_buffer(
            "normalizer",
            torch.tensor(config.hidden_size**0.5),
            persistent=False,
        )

        ple_dim = self.hidden_size_per_layer_input
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size

        def _make_empty_intermediate_tensors(
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> IntermediateTensors:
            tensors: dict[str, torch.Tensor] = {
                "hidden_states": torch.zeros(
                    (batch_size, hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "residual": torch.zeros(
                    (batch_size, hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }
            if ple_dim and ple_dim > 0:
                tensors["per_layer_inputs"] = torch.zeros(
                    (batch_size, num_layers, ple_dim),
                    dtype=dtype,
                    device=device,
                )
            return IntermediateTensors(tensors)

        self.make_empty_intermediate_tensors = _make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Global Embedding Dump
        save_tensor(input_ids, "input_ids", module="model")
        embed_out = self.embed_tokens(input_ids) * self.normalizer
        save_tensor(embed_out, "embedding_output", module="model")
        return embed_out

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens_per_layer is None:
            return None

        per_layer_inputs_mask = torch.logical_and(
            input_ids >= 0,
            input_ids < self.vocab_size_per_layer_input,
        )
        per_layer_inputs_tokens = torch.where(
            per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids)
        )

        per_layer_embeds = self.embed_tokens_per_layer(per_layer_inputs_tokens)
        save_tensor(per_layer_embeds, "ple_embeds_raw", module="model")
        
        per_layer_embeds = per_layer_embeds * self.embed_scale_per_layer
        save_tensor(per_layer_embeds, "ple_embeds_scaled", module="model")
        
        per_layer_embeds = per_layer_embeds.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        save_tensor(per_layer_embeds, "ple_embeds_reshaped", module="model")
        return per_layer_embeds

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.per_layer_model_projection is None:
            return None

        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        save_tensor(per_layer_projection, "ple_proj_raw", module="model")
        
        per_layer_projection = per_layer_projection * self.per_layer_projection_scale
        save_tensor(per_layer_projection, "ple_proj_scaled", module="model")
        
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        save_tensor(per_layer_projection, "ple_proj_reshaped", module="model")
        
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        save_tensor(per_layer_projection, "ple_proj_normed", module="model")

        if per_layer_inputs is None:
            return per_layer_projection

        combined = (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale
        save_tensor(combined, "ple_combined", module="model")
        return combined

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
                per_layer_inputs = self.project_per_layer_inputs(
                    hidden_states, per_layer_inputs
                )
            else:
                hidden_states = self.embed_input_ids(input_ids)
                per_layer_embeds = self.get_per_layer_inputs(input_ids)
                per_layer_inputs = self.project_per_layer_inputs(
                    hidden_states, per_layer_embeds
                )
            residual = None
            save_tensor(hidden_states, "initial_hidden_states", module="model")
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            per_layer_inputs = intermediate_tensors.get("per_layer_inputs")

        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            actual_layer_idx = self.start_layer + layer_idx
            if per_layer_inputs is not None:
                layer_per_input = per_layer_inputs[
                    :, actual_layer_idx, :
                ]
            else:
                layer_per_input = None
            
            # Model层面每层输入Dump
            save_tensor(hidden_states, f"layer{actual_layer_idx}_input", module="model")
            
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                per_layer_input=layer_per_input,
                **kwargs,
            )
            
            # Model层面每层输出Dump
            save_tensor(hidden_states, f"layer{actual_layer_idx}_output", module="model")
            
            if actual_layer_idx == 0:
                save_tensor(hidden_states, "LAYER0_FOCUS_MODEL_OUTPUT", module="model")

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                    "per_layer_inputs": per_layer_inputs,
                }
            )

        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        
        # Global最终输出Dump
        save_tensor(hidden_states, "model_final_output", module="model")
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # 保持原有load_weights逻辑不变，省略重复代码
        loaded_params = set()
        return loaded_params


class Gemma4ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, MixtureOfExperts):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = _get_text_config(vllm_config.model_config.hf_config)
        quant_config = vllm_config.quant_config
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Gemma4Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            soft_cap=getattr(config, "final_logit_softcapping", None),
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.expert_weights: list[list[torch.Tensor]] = []
        self.moe_layers: list[nn.Module] = []
        example_moe: Gemma4MoE | None = None
        for layer in self.model.layers:
            if hasattr(layer, "moe") and isinstance(layer.moe, Gemma4MoE):
                example_moe = layer.moe
                self.moe_layers.append(layer.moe.experts)
        self.num_moe_layers = len(self.moe_layers)
        if example_moe is not None:
            self.num_logical_experts = example_moe.num_experts
            self.num_physical_experts = example_moe.num_experts
            self.num_local_physical_experts = example_moe.num_experts
            self.num_routed_experts = example_moe.num_experts
        else:
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_redundant_experts = 0

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # 新增：Logits层Dump
        logits = self.logits_processor(self.lm_head, hidden_states)
        save_tensor(hidden_states, "logits_input", module="lm_head")
        save_tensor(logits, "logits_output", module="lm_head")
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # 保持原有load_weights逻辑不变，省略重复代码
        loaded_params = set()
        return loaded_params
