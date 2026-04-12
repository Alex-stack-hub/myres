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
"""Gemma 4模型在vLLM中的实现。"""

# 导入必要的库和模块
from collections.abc import Iterable  # 导入可迭代对象类型，用于类型提示
from dataclasses import replace  # 用于创建数据类的副本，保持不可变性
from itertools import islice  # 用于切片迭代器，支持从迭代器中获取指定范围的元素

import regex as re  # 正则表达式库，用于处理字符串模式匹配
import torch  # PyTorch库，深度学习框架
from torch import nn  # PyTorch神经网络模块，提供各种神经网络层

# 导入vLLM相关模块
from vllm.compilation.decorators import support_torch_compile  # 支持Torch编译的装饰器，用于优化模型性能
from vllm.config import CacheConfig, VllmConfig  # 缓存配置和vLLM配置类
from vllm.distributed import (
    get_pp_group,  # 获取流水线并行组，用于分布式训练和推理
    get_tensor_model_parallel_rank,  # 获取张量模型并行排名，用于确定当前进程在张量并行中的位置
    get_tensor_model_parallel_world_size,  # 获取张量模型并行世界大小，用于确定张量并行的进程数量
)
from vllm.forward_context import get_forward_context  # 获取前向传播上下文，用于传递训练/推理过程中的元数据
from vllm.logger import init_logger  # 初始化日志记录器，用于记录模型运行信息
from vllm.model_executor.layers.activation import GeluAndMul  # GELU激活函数的优化实现
from vllm.model_executor.layers.attention import Attention  # 注意力机制的实现
from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear  # 融合专家混合模型的实现
from vllm.model_executor.layers.layernorm import RMSNorm  # RMS归一化层的实现
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,  # 列并行线性层，用于分布式模型的线性层实现
    MergedColumnParallelLinear,  # 合并列并行线性层，优化计算效率
    QKVParallelLinear,  # QKV并行线性层，用于注意力机制中的QKV投影
    ReplicatedLinear,  # 复制线性层，在所有进程中复制相同的权重
    RowParallelLinear,  # 行并行线性层，用于分布式模型的线性层实现
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor  # Logits处理器，用于处理模型输出的logits
from vllm.model_executor.layers.quantization import QuantizationConfig  # 量化配置类，用于模型量化
from vllm.model_executor.layers.rotary_embedding import get_rope  # 获取旋转位置编码，用于注意力机制中的位置编码
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,  # 并行语言模型头，用于分布式模型的输出层
    VocabParallelEmbedding,  # 词汇并行嵌入，用于分布式模型的词嵌入
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,  # 默认权重加载器，用于加载模型权重
    maybe_remap_kv_scale_name,  # 可能重映射KV缩放名称，用于权重加载时的名称映射
)
from vllm.sequence import IntermediateTensors  # 中间张量类，用于存储模型中间计算结果
from vllm.v1.attention.backends.utils import KVSharingFastPrefillMetadata  # KV共享快速预填充元数据，用于优化预填充阶段的性能

# 导入接口和工具函数
from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP  # 接口类，定义模型需要实现的方法
from .utils import (
    AutoWeightsLoader,  # 自动权重加载器，用于自动加载模型权重
    extract_layer_index,  # 提取层索引，从参数名称中提取层索引
    is_pp_missing_parameter,  # 检查是否缺少流水线并行参数，用于权重加载
    make_layers,  # 创建层，用于批量创建模型层
    maybe_prefix,  # 可能添加前缀，用于参数名称的前缀处理
)

# 初始化日志记录器
logger = init_logger(__name__)  # 初始化当前模块的日志记录器


def _get_text_config(config):
    """获取Gemma4模型的文本配置。

    Gemma4检查点使用architectures=["Gemma4ForConditionalGeneration"]，
    这会产生一个带有嵌套text_config的Gemma4Config。此函数
    无论是否嵌套，都会透明地返回文本配置。

    Args:
        config: 模型配置对象，可能是嵌套的Gemma4Config

    Returns:
        文本配置对象，可能是直接的配置或嵌套的text_config
    """
    # 检查配置对象是否有text_config属性
    if hasattr(config, "text_config"):
        # 如果有text_config属性，返回嵌套的文本配置
        return config.text_config
    # 如果没有text_config属性，直接返回配置对象
    return config


class Gemma4MLP(nn.Module):
    """Gemma4的多层感知机实现。

    实现了Gemma4模型中的多层感知机结构，包含门控和上投影层的合并，以及下投影层。
    使用GELU激活函数，并支持模型量化。
    """
    
    def __init__(
        self,
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        hidden_activation: str,  # 隐藏层激活函数
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数命名前缀
    ) -> None:
        """初始化Gemma4MLP。

        Args:
            hidden_size: 隐藏层大小
            intermediate_size: 中间层大小
            hidden_activation: 隐藏层激活函数
            quant_config: 量化配置
            prefix: 参数命名前缀
        """
        super().__init__()
        # 合并的门控和上投影层，使用列并行线性层实现
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,  # 输入大小
            [intermediate_size] * 2,  # 输出大小（门控和上投影）
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.gate_up_proj",  # 参数命名前缀
        )
        # 下投影层，使用行并行线性层实现
        self.down_proj = RowParallelLinear(
            intermediate_size,  # 输入大小
            hidden_size,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.down_proj",  # 参数命名前缀
        )
        # 检查激活函数是否正确，Gemma4必须使用gelu_pytorch_tanh
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma4使用`gelu_pytorch_tanh`作为隐藏层激活函数。"
                "请将`hidden_act`和`hidden_activation`设置为`gelu_pytorch_tanh`。"
            )
        # 激活函数，使用GeluAndMul实现，近似方法为tanh
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播函数。
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量
        """
        # 通过门控和上投影层，获取门控和上投影的输出
        gate_up, _ = self.gate_up_proj(x)
        # 应用激活函数，对门控和上投影的输出进行非线性变换
        x = self.act_fn(gate_up)
        # 通过下投影层，将中间层输出映射回隐藏层大小
        x, _ = self.down_proj(x)
        # 返回最终输出
        return x


class Gemma4Router(nn.Module):
    """Gemma4混合专家模型的路由器，在投影前预处理输入。

    应用RMSNorm（无学习权重），root_size缩放
    (hidden_size^{-0.5})，然后在投影到专家logits之前应用学习的每维度缩放。

    此预处理仅应用于路由器的输入，而不应用于专家MLP的输入。
    """

    def __init__(
        self,
        config,  # 模型配置
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数命名前缀
    ) -> None:
        """初始化Gemma4Router。

        Args:
            config: 模型配置
            quant_config: 量化配置
            prefix: 参数命名前缀
        """
        super().__init__()
        # 保存隐藏层大小
        self.hidden_size = config.hidden_size

        # RMSNorm无学习权重 - 仅纯归一化
        self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps, has_weight=False)
        # 每维度学习缩放，在归一化和root_size之后应用
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        # 常数1/sqrt(hidden_size)缩放因子，注册为缓冲区
        self.register_buffer(
            "root_size",
            torch.tensor(self.hidden_size**-0.5),
            persistent=False,  # 不保存到状态字典
        )
        # 投影到专家logits；跨TP复制以确保一致的路由
        # GateLinear支持bf16 W/A → fp32输出，这很重要
        # 因为topk内核通常需要fp32来稳定路由。
        self.proj = GateLinear(
            self.hidden_size,  # 输入大小
            config.num_experts,  # 专家数量
            bias=False,  # 不使用偏置
            out_dtype=torch.float32,  # 输出数据类型
            prefix=f"{prefix}.proj",  # 参数命名前缀
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播函数，返回原始路由器logits [T, E]。
        
        Args:
            x: 输入张量
            
        Returns:
            路由器logits张量
        """
        # 应用RMS归一化
        x = self.norm(x)
        # 应用root_size缩放，将输入缩放到合适的范围
        x = x * self.root_size.to(x.dtype)
        # 应用学习的缩放，对每个维度进行自适应调整
        x = x * self.scale.to(x.dtype)
        # 投影到专家logits，获取每个专家的得分
        router_logits, _ = self.proj(x)
        # 返回路由器logits
        return router_logits


class Gemma4MoE(nn.Module):
    """使用vLLM的FusedMoE实现的Gemma4混合专家模型。

    使用自定义路由包装FusedMoE。路由器投影是外部的（Gemma4Router）
    - 此类仅处理专家调度。

    Gemma4路由：对所有专家进行softmax → 选择top-k → 重新归一化。
    per_expert_scale被折叠到路由权重中，以确保与FusedMoE的融合内核在数学上正确。
    """

    def __init__(
        self,
        config,  # 模型配置
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数命名前缀
    ) -> None:
        """初始化Gemma4MoE。

        Args:
            config: 模型配置
            quant_config: 量化配置
            prefix: 参数命名前缀
        """
        super().__init__()
        # 保存隐藏层大小和专家数量
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts

        # 每专家输出缩放，折叠到路由权重中，以便
        # FusedMoE的融合内核计算：Σ_e (expert_e * w_e * scale_e)
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

        # Gemma4路由：对所有专家进行softmax → 选择top-k → 重新归一化。
        # FusedMoE的内置fused_topk对softmax的处理方式不同，因此
        # 需要自定义路由函数以确保数值正确性。
        per_expert_scale = self.per_expert_scale

        def routing_function(
            hidden_states: torch.Tensor,  # 隐藏状态
            gating_output: torch.Tensor,  # 门控输出
            topk: int,  # 选择的专家数量
            renormalize: bool,  # 是否重新归一化
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """自定义路由函数，实现Gemma4的路由逻辑。

            Args:
                hidden_states: 隐藏状态
                gating_output: 门控输出
                topk: 选择的专家数量
                renormalize: 是否重新归一化

            Returns:
                (topk_weights, topk_ids): 选择的专家权重和索引
            """
            # 选择top-k专家
            _, topk_ids = torch.topk(gating_output, k=topk, dim=-1)
            # 计算所有专家的softmax概率
            router_probabilities = torch.nn.functional.softmax(gating_output, dim=-1)
            # 创建指示符张量，标记选中的专家
            indicator = torch.nn.functional.one_hot(
                topk_ids, num_classes=gating_output.size(-1)
            ).sum(dim=-2)
            # 计算门控权重，只保留选中专家的概率
            gate_weights = indicator * router_probabilities
            # 计算重新归一化因子
            renorm_factor = torch.sum(gate_weights, dim=-1, keepdim=True)
            renorm_factor = torch.where(renorm_factor > 0.0, renorm_factor, 1.0)
            # 计算调度权重，确保权重和为1
            dispatch_weights = gate_weights / renorm_factor

            # 获取top-k权重
            topk_weights = dispatch_weights.gather(1, topk_ids)

            # 将per_expert_scale折叠到路由权重中
            expert_scales = per_expert_scale[topk_ids].to(topk_weights.dtype)
            topk_weights = topk_weights * expert_scales
            return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

        # 带有自定义Gemma4路由的FusedMoE专家
        self.experts = FusedMoE(
            num_experts=config.num_experts,  # 专家数量
            top_k=config.top_k_experts,  # 每个token选择的专家数量
            hidden_size=config.hidden_size,  # 隐藏层大小
            intermediate_size=getattr(
                config,
                "moe_intermediate_size",
                getattr(config, "expert_intermediate_size", None),
            ),  # 专家中间层大小
            reduce_results=True,  # 是否减少结果
            renormalize=True,  # 是否重新归一化
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.experts",  # 参数命名前缀
            custom_routing_function=routing_function,  # 自定义路由函数
            activation="gelu",  # 激活函数
        )

    def forward(self, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        """前向传播函数。
        
        Args:
            x: 输入张量
            router_logits: 路由器logits
            
        Returns:
            输出张量
        """
        # 调用FusedMoE的前向传播，使用自定义路由函数
        return self.experts(x, router_logits)


class Gemma4Attention(nn.Module):
    """Gemma4的注意力机制实现。

    支持全注意力和滑动窗口注意力，以及KV缓存共享。
    使用Q/K/V归一化和旋转位置编码（RoPE）。
    """

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
    ) -> None:
        """初始化Gemma4Attention。

        Args:
            config: 模型配置
            hidden_size: 隐藏层大小
            num_heads: 注意力头数量
            num_kv_heads: KV注意力头数量
            head_dim: 头维度
            max_position_embeddings: 最大位置嵌入
            use_k_eq_v: 是否使用K=V
            cache_config: 缓存配置
            quant_config: 量化配置
            attn_logits_soft_cap: 注意力logits软上限
            prefix: 参数命名前缀
        """
        super().__init__()
        # 保存配置和参数
        self.config = config
        self.hidden_size = hidden_size
        self.use_k_eq_v = use_k_eq_v

        # 获取张量并行大小和排名
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = num_heads
        # 确保头数量能被张量并行大小整除
        assert self.total_num_heads % tp_size == 0
        # 计算每个张量并行进程的头数量
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        # 处理KV头数量与张量并行大小的关系
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        # 计算每个张量并行进程的KV头数量
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        # 计算Q和KV的大小
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Gemma4使用scaling=1.0
        # 与Gemma2/3不同，这里不使用query_pre_attn_scalar；
        # Q/K归一化与可学习权重隐式处理缩放。
        self.scaling = 1.0

        # QKVParallelLinear正确处理所有层类型的GQA
        # k_eq_v层通过_weight_iterator重映射将K权重加载到K和V槽中
        # 不需要结构差异
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # 输出投影
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Q/K归一化：output = norm(x) * weight（可学习的每头缩放）
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # V归一化：无学习缩放（仅纯归一化）
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)

        # 确定层类型和滑动窗口
        layer_idx = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"
        sliding_window = config.sliding_window if self.is_sliding else None

        # 基于层类型初始化RoPE
        # Gemma4对滑动注意力和全注意力使用不同的RoPE参数
        if layer_type in config.rope_parameters:
            # 每层类型的rope配置（字典格式）
            # rope_parameters已经包含每层类型的正确partial_rotary_factor
            # （全注意力为1.0，滑动注意力为1.0）
            # 不要用global_partial_rotary_factor覆盖 — 该配置键对Gemma4不需要
            # 配置使用每层rope_parameters
            rope_parameters = dict(config.rope_parameters[layer_type])
        else:
            # 旧配置格式回退
            rope_parameters = dict(config.rope_parameters.copy())
            if self.is_sliding:
                rope_parameters["rope_theta"] = getattr(
                    config, "rope_local_base_freq", 10000.0
                )

        # KV共享：最后`num_kv_shared_layers`层与相同类型的早期层共享KV缓存
        kv_sharing_target_layer_name = None
        self.is_kv_shared_layer = False
        num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)
        if num_kv_shared_layers > 0:
            first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared_layers
            if layer_idx >= first_kv_shared_layer_idx:
                self.is_kv_shared_layer = True
                # 找到相同注意力类型的最后一个非共享层
                prev_layers = config.layer_types[:first_kv_shared_layer_idx]
                current_layer_type = config.layer_types[layer_idx]
                kv_shared_layer_index = (
                    len(prev_layers) - 1 - prev_layers[::-1].index(current_layer_type)
                )
                if kv_shared_layer_index >= 0:
                    if ".layers." in prefix:
                        param_name_before_layers = prefix.split(".layers.")[0]
                    else:
                        raise ValueError(
                            "Gemma4Attention的前缀格式意外: "
                            f"'{prefix}'. 预期包含'.layers.'。"
                        )
                    kv_sharing_target_layer_name = (
                        f"{param_name_before_layers}.layers."
                        f"{kv_shared_layer_index}.self_attn.attn"
                    )

        # 初始化旋转位置编码
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

        # 初始化注意力层
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
        """前向传播函数。

        Args:
            positions: 位置张量
            hidden_states: 隐藏状态张量
            **kwargs: 其他参数

        Returns:
            注意力输出张量
        """
        # 统一QKV路径（适用于k_eq_v和标准层）
        # 对于k_eq_v，K权重被加载到qkv_proj的K和V槽中，因此V == K自动成立
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Q归一化（始终应用）
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        if not self.is_kv_shared_layer:
            # 非共享：应用K归一化 + RoPE，V归一化
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k = self.k_norm(k)
            k = k.flatten(-2, -1)
            q, k = self.rotary_emb(positions, q, k)

            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = self.v_norm(v)
            v = v.flatten(-2, -1)
        else:
            # 共享：仅对Q应用RoPE
            q = self.rotary_emb(positions, q, k)[0]

        # 计算注意力输出
        attn_output = self.attn(q, k, v)
        # 应用输出投影
        output, _ = self.o_proj(attn_output)

        return output


class Gemma4DecoderLayer(nn.Module):
    """Gemma4的解码器层实现。

    包含注意力机制、MLP、混合专家模型（MoE）和每层嵌入（PLE）等组件。
    支持全注意力和滑动窗口注意力，以及KV缓存共享。
    """

    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """初始化Gemma4DecoderLayer。

        Args:
            config: 模型配置
            cache_config: 缓存配置
            quant_config: 量化配置
            prefix: 参数命名前缀
        """
        super().__init__()
        # 保存隐藏层大小和每层输入大小
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", 0
        )

        # 提取层索引
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx

        # Gemma4对滑动注意力和全注意力使用不同的头维度
        layer_type = config.layer_types[layer_idx]
        self.is_full_attention = layer_type == "full_attention"
        if self.is_full_attention:
            head_dim = getattr(config, "global_head_dim", config.head_dim)
        else:
            head_dim = config.head_dim

        # 确定全注意力层是否使用k_eq_v
        # (笔记本变体：无v_proj，在全注意力层上重用K作为V)
        use_k_eq_v = self.is_full_attention and getattr(
            config, "attention_k_eq_v", False
        )

        # 对于k_eq_v全注意力层，使用num_global_key_value_heads作为KV头计数
        if use_k_eq_v:
            num_kv_heads = getattr(
                config, "num_global_key_value_heads", config.num_key_value_heads
            )
        else:
            num_kv_heads = config.num_key_value_heads

        # 初始化自注意力层
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
        )

        # 从配置计算每层的中间大小
        # 当use_double_wide_mlp设置时，KV共享层的中间大小加倍
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

        # 初始化MLP
        self.mlp = Gemma4MLP(
            hidden_size=self.hidden_size,
            intermediate_size=layer_intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        # 层归一化：output = norm(x) * weight
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

        # MoE（混合专家）— 路由器 + 与MLP并行的专家块
        self.enable_moe_block = getattr(config, "enable_moe_block", False) or getattr(
            config, "use_second_mlp_block", False
        )
        if self.enable_moe_block:
            self.router = Gemma4Router(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.router",
            )
            self.moe = Gemma4MoE(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.moe",
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

        # 每层嵌入（PLE）组件 — 存在于每个解码器层中
        if (
            self.hidden_size_per_layer_input is not None
            and self.hidden_size_per_layer_input > 0
        ):
            # 门控：将hidden_states投影到每层维度用于门控
            self.per_layer_input_gate = ReplicatedLinear(
                self.hidden_size,
                self.hidden_size_per_layer_input,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.per_layer_input_gate",
                return_bias=False,
            )
            # 投影：将门控后的每层输入投影回隐藏大小
            self.per_layer_projection = ReplicatedLinear(
                self.hidden_size_per_layer_input,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.per_layer_projection",
                return_bias=False,
            )
            # PLE后归一化：output = norm(x) * weight
            self.post_per_layer_input_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

        # 层标量（从检查点加载）— 应用于所有文本层
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        per_layer_input: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播函数。

        Args:
            positions: 位置张量
            hidden_states: 隐藏状态张量
            residual: 残差张量
            per_layer_input: 每层输入张量
            **kwargs: 其他参数

        Returns:
            (hidden_states, None): 输出隐藏状态和残差
        """
        # Gemma4残差模式：
        # 1. input_norm(x) → attn → post_attn_norm → ADD residual
        # 2. pre_ff_norm → mlp → post_ff_norm → ADD residual
        residual = hidden_states

        # 输入归一化
        hidden_states = self.input_layernorm(residual)

        # 自注意力计算
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )

        # 注意力后归一化和残差连接
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states

        # MLP无条件运行（MoE和非MoE使用相同的输入）
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # 如果启用了MoE块
        if self.enable_moe_block:
            # MLP输出归一化
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

            # 路由器和MoE专家看到残差（MLP前状态），
            # 与HF transformers前向路径匹配
            router_logits = self.router(residual)
            hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
            hidden_states_2 = self.moe(hidden_states_2, router_logits)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            # 合并MLP和MoE输出
            hidden_states = hidden_states_1 + hidden_states_2

        # 前馈后归一化和残差连接
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        # 如果配置了PLE（每层嵌入），应用它
        if per_layer_input is not None and self.per_layer_input_gate is not None:
            # 计算门控
            gate = self.per_layer_input_gate(hidden_states)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            # 应用门控到每层输入
            gated_per_layer = gate * per_layer_input
            # 投影回隐藏大小
            per_layer_contribution = self.per_layer_projection(gated_per_layer)
            # 归一化
            per_layer_contribution = self.post_per_layer_input_norm(
                per_layer_contribution
            )
            # 添加到隐藏状态
            hidden_states = hidden_states + per_layer_contribution

        # 应用层标量（所有文本层）
        hidden_states = hidden_states * self.layer_scalar

        # 返回隐藏状态和None（Gemma4直接在隐藏状态中包含残差）
        return hidden_states, None


def _run_decoder_layers(
    decoder_layers: list[Gemma4DecoderLayer],
    layer_idx_start: int,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    per_layer_inputs: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """运行解码器层切片并提取PLE。

    按顺序执行解码器层列表中的所有层，处理每层的输入和输出，
    并支持每层嵌入（PLE）的提取和应用。

    Args:
        decoder_layers: 解码器层列表
        layer_idx_start: 起始层索引
        positions: 位置张量
        hidden_states: 隐藏状态张量
        per_layer_inputs: 每层输入张量
        **kwargs: 其他参数

    Returns:
        处理后的隐藏状态张量
    """
    # 初始化残差为None
    residual = None
    # 遍历解码器层
    for idx, layer in enumerate(decoder_layers):
        # 计算当前层的实际索引
        layer_idx = idx + layer_idx_start
        # 提取当前层的每层输入
        layer_per_input = (
            per_layer_inputs[:, layer_idx, :] if per_layer_inputs is not None else None
        )
        # 调用层的前向传播
        hidden_states, residual = layer(
            positions,
            hidden_states,
            residual,
            per_layer_input=layer_per_input,
            **kwargs,
        )
    # 返回处理后的隐藏状态
    return hidden_states


@support_torch_compile(
    enable_if=lambda vllm_config: vllm_config.cache_config.kv_sharing_fast_prefill
)
class Gemma4SelfDecoderLayers(nn.Module):
    """编译包装器：嵌入 + 非KV共享层（YOCO前半部分）。

    拥有嵌入和PLE模块，使它们位于编译图内。Gemma4Model将嵌入方法委托给这里。
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layers: list[Gemma4DecoderLayer],
        layer_idx_start: int,
        embed_tokens: VocabParallelEmbedding,
        normalizer: torch.Tensor,
        embed_tokens_per_layer: VocabParallelEmbedding | None,
        embed_scale_per_layer: torch.Tensor | None,
        per_layer_model_projection: ColumnParallelLinear | None,
        per_layer_projection_norm: RMSNorm | None,
        per_layer_input_scale: torch.Tensor | None,
        per_layer_projection_scale: torch.Tensor | None,
    ):
        """初始化Gemma4SelfDecoderLayers。

        Args:
            vllm_config: vLLM配置
            prefix: 参数命名前缀
            decoder_layers: 解码器层列表
            layer_idx_start: 起始层索引
            embed_tokens: 词汇并行嵌入
            normalizer: 归一化器
            embed_tokens_per_layer: 每层词汇嵌入
            embed_scale_per_layer: 每层嵌入缩放
            per_layer_model_projection: 每层模型投影
            per_layer_projection_norm: 每层投影归一化
            per_layer_input_scale: 每层输入缩放
            per_layer_projection_scale: 每层投影缩放
        """
        super().__init__()
        # 保存解码器层和起始索引
        self.decoder_layers = decoder_layers
        self.layer_idx_start = layer_idx_start

        # 获取文本配置
        config = _get_text_config(vllm_config.model_config.hf_config)
        self.config = config
        # 获取每层输入大小和词汇大小
        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", 0
        )
        self.vocab_size_per_layer_input = getattr(
            config, "vocab_size_per_layer_input", config.vocab_size
        )

        # 共享对Gemma4Model拥有的模块的引用 — 必须在这个nn.Module内部，
        # 以便torch.compile捕获它们。
        self.embed_tokens = embed_tokens
        self.normalizer = normalizer
        self.embed_tokens_per_layer = embed_tokens_per_layer
        self.embed_scale_per_layer = embed_scale_per_layer
        self.per_layer_model_projection = per_layer_model_projection
        self.per_layer_projection_norm = per_layer_projection_norm
        self.per_layer_input_scale = per_layer_input_scale
        self.per_layer_projection_scale = per_layer_projection_scale

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """将输入ID嵌入为向量。

        Args:
            input_ids: 输入ID张量

        Returns:
            嵌入向量
        """
        return self.embed_tokens(input_ids) * self.normalizer

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """从embed_tokens_per_layer获取每层嵌入。

        Returns:
            每层嵌入 (num_tokens, num_layers, hidden_size_per_layer_input)
        """
        if self.embed_tokens_per_layer is None:
            return None
        # 创建掩码，过滤有效的输入ID
        per_layer_inputs_mask = torch.logical_and(
            input_ids >= 0,
            input_ids < self.vocab_size_per_layer_input,
        )
        # 替换无效的输入ID为0
        per_layer_inputs_tokens = torch.where(
            per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids)
        )
        # 获取每层嵌入
        per_layer_embeds = self.embed_tokens_per_layer(per_layer_inputs_tokens)
        # 应用缩放
        per_layer_embeds = per_layer_embeds * self.embed_scale_per_layer
        # 重塑为正确的形状
        return per_layer_embeds.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """投影输入嵌入并与每层输入组合。

        步骤：
        1. 投影输入嵌入：hidden_size → total_ple_dim
        2. 按hidden_size^{-0.5}缩放
        3. 重塑为 (num_tokens, num_layers, per_layer_dim)
        4. 用per_layer_projection_norm归一化
        5. 组合：(projection + per_layer_inputs) * 1/sqrt(2)

        Args:
            inputs_embeds: 输入嵌入
            per_layer_inputs: 每层输入

        Returns:
            组合后的每层输入
        """
        if self.per_layer_model_projection is None:
            return None
        # 投影输入嵌入
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        # 应用缩放
        per_layer_projection = per_layer_projection * self.per_layer_projection_scale
        # 重塑为正确的形状
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        # 归一化
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        # 如果没有每层输入，直接返回投影
        if per_layer_inputs is None:
            return per_layer_projection
        # 组合投影和每层输入
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """前向传播函数。

        Args:
            input_ids: 输入ID张量
            positions: 位置张量
            inputs_embeds: 输入嵌入张量
            per_layer_inputs: 每层输入张量
            **kwargs: 其他参数

        Returns:
            (hidden_states, per_layer_inputs): 隐藏状态和每层输入
        """
        if inputs_embeds is not None:
            # 使用提供的输入嵌入
            hidden_states = inputs_embeds
            # 投影每层输入
            per_layer_inputs = self.project_per_layer_inputs(
                hidden_states, per_layer_inputs
            )
        else:
            # 从输入ID获取嵌入
            hidden_states = self.embed_input_ids(input_ids)
            # 获取每层嵌入
            per_layer_embeds = self.get_per_layer_inputs(input_ids)
            # 投影每层输入
            per_layer_inputs = self.project_per_layer_inputs(
                hidden_states, per_layer_embeds
            )

        # 运行解码器层
        hidden_states = _run_decoder_layers(
            self.decoder_layers,
            self.layer_idx_start,
            positions,
            hidden_states,
            per_layer_inputs,
            **kwargs,
        )
        return hidden_states, per_layer_inputs


@support_torch_compile(
    enable_if=lambda vllm_config: vllm_config.cache_config.kv_sharing_fast_prefill
)
class Gemma4CrossDecoderLayers(nn.Module):
    """交叉解码器层（YOCO后半部分，KV共享）。"""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layers: list[Gemma4DecoderLayer],
        layer_idx_start: int,
    ):
        """初始化Gemma4CrossDecoderLayers。

        Args:
            vllm_config: vLLM配置
            prefix: 参数命名前缀
            decoder_layers: 解码器层列表
            layer_idx_start: 起始层索引
        """
        super().__init__()
        # 保存解码器层和起始索引
        self.decoder_layers = decoder_layers
        self.layer_idx_start = layer_idx_start

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        per_layer_inputs: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """前向传播函数。

        Args:
            positions: 位置张量
            hidden_states: 隐藏状态张量
            per_layer_inputs: 每层输入张量
            **kwargs: 其他参数

        Returns:
            处理后的隐藏状态张量
        """
        # 运行解码器层
        return _run_decoder_layers(
            self.decoder_layers,
            self.layer_idx_start,
            positions,
            hidden_states,
            per_layer_inputs,
            **kwargs,
        )


@support_torch_compile(
    enable_if=lambda vllm_config: not vllm_config.cache_config.kv_sharing_fast_prefill
)
class Gemma4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _get_text_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        # PLE config values (default to 0 if not present — disables PLE)
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

        # Per-Layer Embedding (PLE) components
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
            # Scaled embedding factor (from config, not hardcoded)
            # Register as buffer so it moves to GPU with the model
            # and interacts correctly with torch.compile AOT caching.
            self.register_buffer(
                "embed_scale_per_layer",
                torch.tensor(self.hidden_size_per_layer_input**0.5),
                persistent=False,
            )
            # Projection: hidden_size → total_ple_dim
            # ColumnParallelLinear with gather_output=True
            self.per_layer_model_projection = ColumnParallelLinear(
                config.hidden_size,
                total_ple_dim,
                bias=False,
                gather_output=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.per_layer_model_projection",
            )
            # PLE projection norm: output = norm(x) * weight
            self.per_layer_projection_norm = RMSNorm(
                self.hidden_size_per_layer_input,
                eps=config.rms_norm_eps,
            )
            # Scale factor for combining projection + per_layer_inputs
            # Register as buffer so it moves to GPU with the model
            # and interacts correctly with torch.compile AOT caching.
            self.register_buffer(
                "per_layer_input_scale",
                torch.rsqrt(torch.tensor(2.0)),
                persistent=False,
            )
            # Scaled projection: multiply output by hidden_size**-0.5.
            # Register as buffer for GPU placement and torch.compile.
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
        # Final norm: output = norm(x) * weight
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Embedding scale = sqrt(hidden_size)
        # Downcast to model dtype (bfloat16 etc.) for numerical parity
        self.register_buffer(
            "normalizer",
            torch.tensor(config.hidden_size**0.5),
            persistent=False,
        )

        # --- You Only Cache Once (YOCO) split for fast prefill ---
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )

        from vllm.compilation.backends import set_model_tag

        # Layers 0..(K-1) are self-decoder layers in YOCO
        with set_model_tag("self_decoder"):
            self.self_decoder = Gemma4SelfDecoderLayers(
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_decoder",
                decoder_layers=self.layers[:first_kv_shared_layer_idx],
                layer_idx_start=0,
                embed_tokens=self.embed_tokens,
                normalizer=self.normalizer,
                embed_tokens_per_layer=getattr(self, "embed_tokens_per_layer", None),
                embed_scale_per_layer=getattr(self, "embed_scale_per_layer", None),
                per_layer_model_projection=getattr(
                    self, "per_layer_model_projection", None
                ),
                per_layer_projection_norm=getattr(
                    self, "per_layer_projection_norm", None
                ),
                per_layer_input_scale=getattr(self, "per_layer_input_scale", None),
                per_layer_projection_scale=getattr(
                    self, "per_layer_projection_scale", None
                ),
            )
        # Layers K..(N-1) are cross-decoder layers in YOCO
        with set_model_tag("cross_decoder"):
            self.cross_decoder = Gemma4CrossDecoderLayers(
                vllm_config=vllm_config,
                prefix=f"{prefix}.cross_decoder",
                decoder_layers=self.layers[first_kv_shared_layer_idx:],
                layer_idx_start=first_kv_shared_layer_idx,
            )

        self.fast_prefill_enabled = cache_config.kv_sharing_fast_prefill

        if self.fast_prefill_enabled:
            # Allocate static buffers for CUDAGraph
            max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            device = next(self.parameters()).device
            self.positions = torch.zeros(
                max_num_tokens, dtype=torch.int64, device=device
            )
            self.hidden_states = torch.zeros(
                (max_num_tokens, config.hidden_size),
                dtype=self.embed_tokens.weight.dtype,
                device=device,
            )
            if (
                self.hidden_size_per_layer_input
                and self.hidden_size_per_layer_input > 0
            ):
                self.per_layer_inputs = torch.zeros(
                    (
                        max_num_tokens,
                        config.num_hidden_layers,
                        self.hidden_size_per_layer_input,
                    ),
                    dtype=self.embed_tokens.weight.dtype,
                    device=device,
                )
            else:
                self.per_layer_inputs = None

        # Custom factory that includes per_layer_inputs for PLE-enabled PP.
        # per_layer_inputs has shape (batch, num_layers, per_layer_dim),
        # which differs from the standard (batch, hidden_size) shape,
        # so we can't use the default factory.
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
        return self.self_decoder.embed_input_ids(input_ids)

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """Get per-layer embeddings from embed_tokens_per_layer.

        Returns:
            Per-layer embeddings (num_tokens, num_layers,
            hidden_size_per_layer_input)
        """
        return self.self_decoder.get_per_layer_inputs(input_ids)

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Project inputs_embeds and combine with per_layer_inputs.

        Steps:
        1. Project inputs_embeds: hidden_size → total_ple_dim
        2. Scale by hidden_size^{-0.5}
        3. Reshape to (num_tokens, num_layers, per_layer_dim)
        4. Normalize with per_layer_projection_norm
        5. Combine: (projection + per_layer_inputs) * 1/sqrt(2)
        """
        return self.self_decoder.project_per_layer_inputs(
            inputs_embeds, per_layer_inputs
        )

    def fast_prefill_forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        logits_indices_padded, num_logits_indices = None, None
        attn_metadata = get_forward_context().attn_metadata

        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            layer_attn_metadata = attn_metadata[
                self.layers[-1].self_attn.attn.layer_name
            ]
            if isinstance(layer_attn_metadata, KVSharingFastPrefillMetadata):
                logits_indices_padded = layer_attn_metadata.logits_indices_padded
                num_logits_indices = layer_attn_metadata.num_logits_indices

        batch_size = positions.size(0)
        self.positions[:batch_size].copy_(positions)
        self_decoder_hidden_states, per_layer_inputs = self.self_decoder(
            input_ids=input_ids,
            positions=self.positions[:batch_size],
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            **kwargs,
        )

        if logits_indices_padded is None:
            logits_indices_padded = torch.arange(
                batch_size,
                dtype=positions.dtype,
                device=positions.device,
            )

        # NOTE: Keep .clone() until fix in
        # https://github.com/vllm-project/vllm/pull/22282
        hidden_states = self_decoder_hidden_states.clone()

        num_padded = logits_indices_padded.size(0)
        self.positions[:num_padded].copy_(positions[logits_indices_padded])
        self.hidden_states[:num_padded].copy_(
            self_decoder_hidden_states[logits_indices_padded]
        )
        if self.per_layer_inputs is not None and per_layer_inputs is not None:
            self.per_layer_inputs[:num_padded].copy_(
                per_layer_inputs[logits_indices_padded]
            )

        # Update batch_descriptor so the cross-decoder's piecewise
        # CUDAGraphWrapper dispatches to the correct (reduced) batch size.
        forward_context = get_forward_context()
        orig_batch_desc = forward_context.batch_descriptor
        if orig_batch_desc is not None:
            forward_context.batch_descriptor = replace(
                orig_batch_desc, num_tokens=num_padded
            )

        cross_per_layer = (
            self.per_layer_inputs[:num_padded]
            if self.per_layer_inputs is not None
            else None
        )
        cross_hidden_states = self.cross_decoder(
            self.positions[:num_padded],
            self.hidden_states[:num_padded],
            cross_per_layer,
            **kwargs,
        )

        # Restore the original batch_descriptor
        forward_context.batch_descriptor = orig_batch_desc

        if num_logits_indices is not None:
            assert num_logits_indices > 0
            hidden_states[logits_indices_padded[:num_logits_indices]] = (
                cross_hidden_states[:num_logits_indices]
            )
        else:
            hidden_states = cross_hidden_states

        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if self.fast_prefill_enabled:
            hidden_states = self.fast_prefill_forward(
                input_ids,
                positions,
                inputs_embeds,
                per_layer_inputs,
                **kwargs,
            )
            hidden_states = self.norm(hidden_states)
            return hidden_states

        # Normal (non-fast-prefill) path with PP support
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
                # When called from the multimodal wrapper, raw PLE
                # embeddings are pre-computed and passed explicitly.
                # Project them through per_layer_model_projection.
                per_layer_inputs = self.project_per_layer_inputs(
                    hidden_states, per_layer_inputs
                )
            else:
                hidden_states = self.embed_input_ids(input_ids)
                # Compute per-layer inputs for PLE
                per_layer_embeds = self.get_per_layer_inputs(input_ids)
                per_layer_inputs = self.project_per_layer_inputs(
                    hidden_states, per_layer_embeds
                )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            per_layer_inputs = intermediate_tensors.get("per_layer_inputs")

        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            # Extract the per-layer embedding for this specific layer
            if per_layer_inputs is not None:
                actual_layer_idx = self.start_layer + layer_idx
                layer_per_input = per_layer_inputs[
                    :, actual_layer_idx, :
                ]  # (num_tokens, per_layer_dim)
            else:
                layer_per_input = None
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                per_layer_input=layer_per_input,
                **kwargs,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                    "per_layer_inputs": per_layer_inputs,
                }
            )
        # Gemma4 incorporates residual into hidden_states directly
        # Apply norm without residual fusion when possible.
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # MoE expert weight mapping: checkpoint can have either:
        #   1. 3D packed tensors (exploded in _weight_iterator to per-expert 2D)
        #   2. Already per-expert 2D weights (if quantized)
        # Map to FusedMoE parameters:
        #   moe.experts.{id}.gate_proj → FusedMoE w1 (shard of w13)
        #   moe.experts.{id}.up_proj   → FusedMoE w3 (shard of w13)
        #   moe.experts.{id}.down_proj → FusedMoE w2
        #
        # Use prefix matching to handle both weights and
        # quantization scale parameters. The param_name is a prefix ending
        # in underscore, and weight_name ends with a dot, so that:
        #   "experts.0.gate_proj.weight_scale" -> "experts.w13_weight_scale"
        #   "experts.0.gate_proj.weight" -> "experts.w13_weight"
        num_experts = getattr(self.config, "num_experts", None) or 0
        expert_params_mapping = [
            # (param_name, weight_name, expert_id, shard_id)
            (
                "experts.w13_"
                if proj_name in ["gate_proj", "up_proj"]
                else "experts.w2_",
                f"experts.{expert_id}.{proj_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, proj_name in [
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ]
        ]
        params_dict = dict(self.named_parameters())
        # Include buffers (e.g. layer_scalar) so they can be loaded too
        params_dict.update(dict(self.named_buffers()))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            if name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
                remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                if remapped_name is not None and remapped_name in params_dict:
                    param = params_dict[remapped_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(remapped_name)
                    continue

            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                stacked_name = name.replace(shard_name, param_name)
                # k_eq_v layers use separate q_proj/k_proj instead of
                # packed qkv_proj. If the stacked param doesn't exist,
                # skip this mapping and fall through to direct load.
                if stacked_name not in params_dict:
                    continue
                if is_pp_missing_parameter(stacked_name, self):
                    continue
                param = params_dict[stacked_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(stacked_name)
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    # Match both:
                    #  - Bare weights: "experts.0.down_proj" (from 3D explosion)
                    #  - With suffix: "experts.0.down_proj.weight_scale" (2D quantized)
                    # weight_name has trailing dot, so check with and without it
                    weight_name_base = weight_name.rstrip(".")
                    if weight_name in name:
                        # Has suffix (e.g., .weight_scale)
                        moe_name = name.replace(weight_name, param_name)
                    elif name.endswith(weight_name_base):
                        # Bare weight (no suffix)
                        moe_name = name.replace(
                            weight_name_base, param_name.rstrip("_") + "_weight"
                        )
                    else:
                        continue
                    if moe_name not in params_dict:
                        continue
                    if is_pp_missing_parameter(moe_name, self):
                        continue
                    param = params_dict[moe_name]
                    # Expert weights are already in the correct
                    # orientation for FusedMoE after _weight_iterator:
                    #   gate/up: [I, H] → w1/w3 expects [I, H]
                    #   down:    [H, I] → w2 expects [H, I]
                    # Scales and other quantization params may be 1D or scalar.
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        moe_name,  # Pass mapped name (handles both weights and scales)
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(moe_name)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class Gemma4ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, MixtureOfExperts):
    # Note: qkv_proj packing applies to non-k_eq_v layers (sliding
    # attention and full attention without k_eq_v). k_eq_v layers use
    # separate q_proj + k_proj without packing.
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

        # --- MixtureOfExperts protocol ---
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
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Checkpoint weight names use "language_model." prefix (from the
        # Gemma4ForConditionalGeneration wrapper). Strip it to map to our
        # model tree which is just "model.*".
        def _weight_iterator():
            use_k_eq_v = getattr(self.config, "attention_k_eq_v", False)
            # Build set of k_eq_v layer indices (full_attention layers
            # when attention_k_eq_v is enabled). These layers have k_proj
            # but no v_proj in checkpoint — we duplicate k_proj as v_proj.
            k_eq_v_layer_indices: set[int] = set()
            if use_k_eq_v:
                for idx, lt in enumerate(self.config.layer_types):
                    if lt == "full_attention":
                        k_eq_v_layer_indices.add(idx)

            for name, weight in weights:
                # Remap "language_model." → "" to match our model tree.
                # Checkpoint: model.language_model.layers.X.*
                # Our model:  model.layers.X.*
                name = name.replace("language_model.", "")

                # Remap new HF checkpoint naming to internal vLLM
                # naming: HF moved per_expert_scale to router and
                # renamed moe → experts in the MoE block.
                name = name.replace(
                    ".router.per_expert_scale",
                    ".moe.per_expert_scale",
                )
                if ".experts.gate_up_proj" in name:
                    name = name.replace(
                        ".experts.gate_up_proj",
                        ".moe.gate_up_proj",
                    )
                elif ".experts.down_proj" in name:
                    name = name.replace(
                        ".experts.down_proj",
                        ".moe.down_proj",
                    )

                # Remap individual 2D expert weights:
                # .experts.{id}.{proj} → .moe.experts.{id}.{proj}
                # (This handles per-expert 2D quantized weights)
                name = re.sub(r"\.experts\.(\d+)\.", r".moe.experts.\1.", name)

                # MoE expert weights: checkpoint stores as 3D packed
                # tensors.  Explode into per-expert 2D weights for
                # FusedMoE weight_loader.
                #
                # Checkpoint format:
                #   moe.gate_up_proj: [E, 2*I, H]  (fused gate + up)
                #   moe.down_proj:    [E, H, I]
                #
                # FusedMoE expects per-expert:
                #   w1 (gate): [I, H]   — first half of gate_up
                #   w3 (up):   [I, H]   — second half of gate_up
                #   w2 (down): [H, I]   — as-is from checkpoint
                #
                # No transpose needed: checkpoint orientation already
                # matches FusedMoE's expected layout.
                if "moe.gate_up_proj" in name and weight.dim() == 3:
                    num_experts = weight.size(0)
                    intermediate_size = weight.size(1) // 2
                    for expert_id in range(num_experts):
                        gate_weight = weight[expert_id, :intermediate_size, :]
                        up_weight = weight[expert_id, intermediate_size:, :]
                        base = name.replace("moe.", f"moe.experts.{expert_id}.")
                        yield base.replace("gate_up_proj", "gate_proj"), gate_weight
                        yield base.replace("gate_up_proj", "up_proj"), up_weight
                    continue

                if "moe.down_proj" in name and weight.dim() == 3:
                    num_experts = weight.size(0)
                    for expert_id in range(num_experts):
                        expert_name = name.replace("moe.", f"moe.experts.{expert_id}.")
                        yield expert_name, weight[expert_id]
                    continue

                # k_eq_v layers: checkpoint has k_proj but no v_proj.
                # QKVParallelLinear expects both, so duplicate k_proj
                # as v_proj so V gets identical weights to K.
                # ONLY for full_attention layers — sliding layers have
                # their own real v_proj weights.
                if "self_attn.k_proj" in name and k_eq_v_layer_indices:
                    m = re.search(r"layers\.(\d+)\.", name)
                    if m and int(m.group(1)) in k_eq_v_layer_indices:
                        yield name, weight
                        yield name.replace("k_proj", "v_proj"), weight.clone()
                        continue

                yield name, weight

        # Skip multimodal weights — handled by the multimodal wrapper.
        # Also skip lm_head when weights are tied.
        skip = [
            "audio_tower.",
            "vision_tower.",
            "embed_audio.",
            "embed_vision.",
        ]
        if self.config.tie_word_embeddings:
            skip.append("lm_head.")

        loader = AutoWeightsLoader(self, skip_substrs=skip)
        return loader.load_weights(_weight_iterator())
