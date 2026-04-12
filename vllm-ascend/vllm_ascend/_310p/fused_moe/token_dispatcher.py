# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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


import torch
from vllm.distributed.parallel_state import get_ep_group

from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEAllGatherCombineMetadata, MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.token_dispatcher import MoETokenDispatchOutput, TokenDispatcherWithAllGather


class TokenDispatcherWithAllGather310(TokenDispatcherWithAllGather):
    """Ascend 310P平台的Token调度器实现"""

    def __init__(self, **kwargs):
        """初始化TokenDispatcherWithAllGather310

        Args:
            **kwargs: 关键字参数
        """
        # 调用父类初始化方法
        super().__init__(**kwargs)

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        """执行token调度

        Args:
            token_dispatch_input: Token调度输入

        Returns:
            MoETokenDispatchOutput: Token调度输出
        """
        # 从输入中提取参数
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        expert_map = token_dispatch_input.routing.expert_map
        apply_router_weight_on_input = token_dispatch_input.routing.apply_router_weight_on_input
        # 保存原始形状
        restore_shape = hidden_states.shape

        # 计算token数量
        num_tokens = hidden_states.shape[:-1].numel()
        # 如果在输入上应用路由器权重
        if apply_router_weight_on_input:
            # 确保topk_weights的维度为2
            assert topk_weights.dim() == 2, "`topk_weights` should be in shape (num_tokens, topk)"
            # 获取topk值
            _, topk = topk_weights.shape
            # 确保topk为1
            assert topk == 1, "Only support topk=1 when `apply_router_weight_on_input` is True"
            # 在输入上应用路由器权重
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        # 如果存在专家映射
        if expert_map is not None:
            # 创建掩码，标记有效的专家
            mask = expert_map[topk_ids] != -1
            # 应用掩码到topk_weights
            topk_weights = topk_weights * mask
            # 计算当前rank的专家范围
            first_expert_idx = get_ep_group().rank_in_group * self.num_experts_local
            last_expert_idx = first_expert_idx + self.num_experts_local
        else:
            # 如果没有专家映射，使用默认范围
            first_expert_idx = 0
            last_expert_idx = self.num_experts_local

        # 初始化MoE路由
        sorted_hidden_states, expanded_row_idx, expert_tokens = self.moe_init_routing(
            hidden_states,
            topk_ids,
            active_num=num_tokens * self.top_k,
            active_expert_range=[first_expert_idx, last_expert_idx],
        )
        # 将expert_tokens转换为int64类型
        expert_tokens = expert_tokens.to(torch.int64)
        # 设置组列表类型为1（计数模式）
        group_list_type = 1  # `count` mode

        # 返回Token调度输出
        return MoETokenDispatchOutput(
            hidden_states=sorted_hidden_states,
            group_list=expert_tokens,
            group_list_type=group_list_type,
            combine_metadata=MoEAllGatherCombineMetadata(
                topk_weights=topk_weights,
                expanded_row_idx=expanded_row_idx,
                restore_shape=restore_shape,
            ),
        )

    def moe_init_routing(self, x, expert_idx, active_num, active_expert_range):
        """
        初始化混合专家（MoE）模型的路由，通过根据分配的专家组织token并准备数据结构，
        以实现高效的专家计算。

        Args:
            x (torch.Tensor): 包含token表示的输入张量
            expert_idx (torch.Tensor): 包含每个token的专家索引的张量
            active_num (int): 活跃专家的数量或None
            active_expert_range (tuple): 活跃专家的范围（开始，结束）

        Returns:
            tuple: 包含以下内容的元组：
                   - expanded_x: 活跃专家的输入张量子集
                   - expanded_row_idx: token位置的映射索引
                   - expert_tokens_count: 分配给每个专家的token计数
        """
        # 获取int32的最大值
        MAX_INT32 = torch.iinfo(torch.int32).max
        # 解析活跃专家范围
        expert_start, expert_end = active_expert_range
        # 获取输入张量的行数
        num_rows = x.shape[0]
        # 获取每个token的专家数量
        k = expert_idx.shape[-1]
        # 将专家索引展平
        expert_idx_flat = expert_idx.flatten()
        # 创建掩码，标记在活跃专家范围内的token
        mask = (expert_idx_flat >= expert_start) & (expert_idx_flat < expert_end)
        # 计算实际的专家总数
        actual_expert_total_num = mask.sum().item()
        # 将不在活跃专家范围内的token的专家索引设置为MAX_INT32
        expert_idx_flat = torch.where(
            ~mask, torch.full_like(expert_idx_flat, MAX_INT32, dtype=torch.int32), expert_idx_flat
        )
        # 对专家索引进行排序
        sorted_idx = torch.argsort(expert_idx_flat, stable=True)
        # 获取排序后的专家索引
        sorted_expert_idx = expert_idx_flat[sorted_idx]
        # 创建扩展行索引，初始值为-1
        expanded_row_idx = torch.full((num_rows * k,), -1, dtype=torch.int32, device=expert_idx.device)
        # 为活跃专家的token设置行索引
        expanded_row_idx[sorted_idx[:actual_expert_total_num]] = torch.arange(
            actual_expert_total_num, dtype=torch.int32, device=expert_idx.device
        )
        # 计算每个专家的token数量
        expert_tokens_count = torch.bincount(
            sorted_expert_idx[:actual_expert_total_num] - expert_start, minlength=expert_end - expert_start
        )
        # 确定活跃token数量
        active_num = min(active_num or actual_expert_total_num, actual_expert_total_num)
        # 获取排序后的输入张量
        expanded_x = x[sorted_idx[:active_num] // k]

        # 返回结果
        return expanded_x, expanded_row_idx, expert_tokens_count
