#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
from collections.abc import Callable

import torch
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.experts_selector import zero_experts_compute
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult, _MoECommMethods
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.quant_type import QuantType

from .experts_selector import select_experts
from .moe_comm_method import AllGatherCommImpl310


class AscendUnquantizedFusedMoEMethod310(UnquantizedFusedMoEMethod):
    """Ascend 310P平台的未量化融合MoE方法实现"""

    def __init__(self, moe: FusedMoEConfig = None):
        """初始化Ascend未量化融合MoE方法

        Args:
            moe: MoE配置对象
        """
        super().__init__(moe=moe)

    def process_weights_after_loading(self, layer):
        """加载权重后处理

        Args:
            layer: MoE层对象
        """
        # 调用父类的权重后处理方法
        super().process_weights_after_loading(layer)

        # 处理融合的gate_up_proj权重（列并行）
        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)
        # 处理down_proj权重（行并行）
        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: torch.Tensor | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """应用MoE计算

        Args:
            layer: MoE层对象
            x: 输入隐藏状态
            use_grouped_topk: 是否使用分组topk
            top_k: 每个token选择的专家数量
            router_logits: 路由器logits
            renormalize: 是否重新归一化
            topk_group: topk分组大小
            num_expert_group: 专家分组数量
            custom_routing_function: 自定义路由函数
            scoring_func: 评分函数
            e_score_correction_bias: 专家评分修正偏置
            global_num_experts: 全局专家数量
            expert_map: 专家映射
            apply_router_weight_on_input: 是否在输入上应用路由器权重
            **kwargs: 其他参数

        Returns:
            计算后的隐藏状态
        """
        # 获取零专家数量和类型
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)

        # 选择专家
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=global_num_experts,
        )

        # 处理零专家计算
        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=global_num_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=x,
            )

        # 将topk权重转换为与输入相同的数据类型
        topk_weights = topk_weights.to(x.dtype)

        # 获取MoE通信方法
        moe_comm_method = _EXTRA_CTX.moe_comm_method
        # 执行融合专家计算
        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                quant_type=QuantType.NONE,
                dynamic_eplb=False,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
            ),
        )
        # 加上零专家结果
        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result
        # 返回最终隐藏状态
        return final_hidden_states


class AscendFusedMoE310(FusedMoE):
    """Ascend 310P平台的融合MoE实现"""

    def __init__(self, *args, **kwargs):
        """初始化Ascend融合MoE

        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        """
        # 调用父类初始化方法
        super().__init__(*args, **kwargs)

        # 保存全局专家数量
        self.global_num_experts = kwargs["num_experts"]

        # 根据是否有量化配置选择量化方法
        if self.quant_config is None:
            self.quant_method = AscendUnquantizedFusedMoEMethod310(self.moe_config)
        else:
            self.quant_method = self.quant_config.get_quant_method(self, self.layer_name)

        # 确保量化方法不为None
        assert self.quant_method is not None

        # 设置分布式组
        self.moe_config.tp_group = get_tp_group()  # 张量并行组
        self.moe_config.dp_group = get_dp_group()  # 数据并行组
        self.moe_config.ep_group = get_ep_group()  # 专家并行组
        self.moe_config.supports_eplb = False  # 不支持专家并行负载均衡

        # 初始化MoE相关变量
        self.global_expert_map = None
        self.local_expert_map = None
        # 如果专家并行大小大于1，初始化专家映射
        if self.moe_config.ep_size > 1:
            self.global_expert_map, self.local_expert_map = self.init_experts_map(self.moe_config)
        # 计算本地专家数量
        self.local_num_experts = (
            torch.sum(self.local_expert_map != -1).item()
            if self.local_expert_map is not None
            else self.global_num_experts
        )

        # 更新MoE配置
        self.moe_config.num_experts = self.global_num_experts
        self.moe_config.num_local_experts = self.local_num_experts
        self.moe_config.global_redundant_expert_num = 0

        # 准备量化参数
        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": self.hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": self.params_dtype,
            "weight_loader": self.weight_loader,
        }

        # 创建权重
        self.quant_method.create_weights(layer=self, **moe_quant_params)
        # 获取量化类型
        self.quant_type = self.get_quant_type()

        # 注册AllGather通信实现
        _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl310(self.moe_config)
        # 初始化运行器
        self.runner = self._init_runner()

    def _init_runner(self):
        """初始化MoE运行器

        Returns:
            AscendMoERunner实例
        """
        from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

        return AscendMoERunner(
            layer=self,
            moe_config=self.moe_config,
            router=self.router,
            routed_input_transform=self._routed_input_transform,
            gate=self.gate,
            shared_experts=self.shared_experts,
            quant_method=self.quant_method,
            reduce_results=self.reduce_results,
            enable_dbo=self.vllm_config.parallel_config.enable_dbo,
        )

    def init_experts_map(self, moe_config):
        """
        初始化MoE（混合专家）模型的专家映射。

        该函数为专家并行组中的每个rank创建全局专家索引和本地专家索引之间的映射。
        它将总专家数分配给不同的rank，并创建全局和本地专家映射，这些映射在MoE计算期间
        用于确定哪些专家由哪个rank处理。

        Args:
            moe_config: 包含MoE参数的配置对象，包括专家数量、专家并行大小和专家并行rank。

        Returns:
            tuple: 包含以下内容的元组：
                   - global_expert_map: 所有rank的专家映射堆栈
                   - local_expert_map: 当前rank的专家映射（已转移到NPU）
        """
        # 获取专家数量和专家并行大小
        n_experts = moe_config.num_experts
        ep_size = moe_config.ep_size
        # 创建专家索引
        all_experts = torch.arange(n_experts, dtype=torch.int32)
        # 将专家分成ep_size个组
        experts_groups = all_experts.chunk(ep_size)
        # 初始化全局专家映射和本地专家映射
        global_expert_map = []
        local_expert_map = None
        # 为每个rank创建专家映射
        for rankid in range(ep_size):
            # 创建一个全为-1的专家映射
            expert_map = torch.full((n_experts,), -1, dtype=torch.int32)
            # 获取当前rank的本地专家
            local_experts = experts_groups[rankid]
            # 为本地专家分配本地索引
            expert_map[local_experts] = torch.arange(local_experts.shape[0], dtype=torch.int32)
            # 添加到全局专家映射
            global_expert_map.append(expert_map)
            # 如果是当前rank，保存本地专家映射并转移到NPU
            if rankid == moe_config.ep_rank:
                local_expert_map = expert_map.npu()
        # 返回全局和本地专家映射
        return torch.stack(global_expert_map), local_expert_map

    def get_quant_type(self) -> QuantType:
        """获取量化类型

        Returns:
            QuantType: 量化类型
        """
        # 获取量化方法
        quant_method = self.quant_method
        # 如果量化方法没有quant_method属性或quant_method为None，返回NONE
        if not hasattr(quant_method, "quant_method") or quant_method.quant_method is None:
            return QuantType.NONE

        # 获取量化方法的具体实现
        method = quant_method.quant_method
        # 获取量化类型
        quant_type = getattr(method, "quant_type", QuantType.NONE)
        # 检查是否支持该量化类型
        if quant_type not in [QuantType.NONE, QuantType.W8A8]:
            raise RuntimeError("Only Unquant and W8A8 is supported.")
        # 返回量化类型
        return quant_type

    def forward_impl(  # type: ignore[override]
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        """前向传播实现

        Args:
            hidden_states: 隐藏状态
            router_logits: 路由器logits

        Returns:
            计算后的隐藏状态
        """
        # 确保量化方法不为None
        assert self.quant_method is not None
        # 确保routed_scaling_factor为1.0
        assert self.routed_scaling_factor == 1.0, "routed_scaling_factor != 1.0 is not supported."

        # 准备输入
        prepare_output = _EXTRA_CTX.moe_comm_method.prepare(
            hidden_states=hidden_states, router_logits=router_logits, quant_type=self.quant_type
        )
        # 获取准备后的输出
        hidden_states = prepare_output.hidden_states
        router_logits = prepare_output.router_logits
        pertoken_scale = prepare_output.pertoken_scale
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape

        # 矩阵乘法
        fused_experts_results: FusedExpertsResult = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            use_grouped_topk=self.use_grouped_topk,
            top_k=self.top_k,
            router_logits=router_logits,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            global_num_experts=self.global_num_experts,
            expert_map=self.local_expert_map,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            pertoken_scale=pertoken_scale,
        )

        # 最终处理
        routed_out = _EXTRA_CTX.moe_comm_method.finalize(
            hidden_states=fused_experts_results.routed_out,
            reduce_results=self.reduce_results,
            padded_hidden_states_shape=padded_hidden_states_shape,
        )

        # 返回处理后的输出
        return routed_out


class AscendSharedFusedMoE310(SharedFusedMoE, AscendFusedMoE310):
    """Ascend 310P平台的共享融合MoE实现"""

    def __init__(
        self,
        shared_experts: torch.nn.Module,
        gate: torch.nn.Module | None = None,
        use_overlapped: bool = True,
        routed_input_transform: torch.nn.Module | None = None,
        **kwargs,
    ):
        """初始化Ascend共享融合MoE

        Args:
            shared_experts: 共享专家模块
            gate: 门控模块
            use_overlapped: 是否使用重叠计算
            routed_input_transform: 路由输入变换
            **kwargs: 其他参数
        """
        # 调用AscendFusedMoE310的初始化方法
        AscendFusedMoE310.__init__(self, **kwargs)
        # 设置路由输入变换
        self._routed_input_transform = routed_input_transform
        # 设置共享专家
        self._shared_experts = shared_experts
        # 设置是否使用重叠计算
        self.use_overlapped = use_overlapped
        # 初始化共享专家流
        self.shared_expert_stream = None
        # 设置门控
        self._gate = gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            hidden_states: 隐藏状态
            router_logits: 路由器logits

        Returns:
            共享专家输出和融合专家输出的元组
        """
        # 如果没有共享专家
        if self._shared_experts is None:
            # 调用AscendFusedMoE310的forward方法
            fused_out = AscendFusedMoE310.forward(
                self,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            # 共享专家输出为None
            shared_out = None
            # 返回共享专家输出和融合专家输出
            return shared_out, fused_out
        # 如果有共享专家
        shared_out, fused_out = AscendFusedMoE310.forward(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        # 返回共享专家输出和融合专家输出
        return shared_out, fused_out

    def _forward_shared_experts(self, hidden_states: torch.Tensor):
        """前向传播共享专家

        Args:
            hidden_states: 隐藏状态

        Returns:
            共享专家的输出
        """
        # 如果没有共享专家，返回None
        if self._shared_experts is None:
            return None
        # 执行共享专家的第一部分
        part1_out = self._shared_experts_part1(hidden_states)
        # 执行共享专家的第二部分
        shared_out = self._shared_experts_part2(hidden_states, part1_out)
        # 返回共享专家输出
        return shared_out

    def forward_impl(  # type: ignore[override]
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ):
        """前向传播实现

        Args:
            hidden_states: 隐藏状态
            router_logits: 路由器logits

        Returns:
            如果有共享专家，返回共享专家输出和融合专家输出的元组；否则返回融合专家输出
        """
        # 调用AscendFusedMoE310的forward_impl方法
        routed_out = AscendFusedMoE310.forward_impl(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        # 如果没有共享专家，返回routed_out
        if self._shared_experts is None:
            return routed_out
        # 执行共享专家前向传播
        shared_out = self._forward_shared_experts(hidden_states)
        # 返回共享专家输出和融合专家输出
        return shared_out, routed_out
