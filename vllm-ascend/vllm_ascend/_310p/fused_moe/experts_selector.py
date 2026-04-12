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

from vllm_ascend.ops.fused_moe.experts_selector import _native_select_experts


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: torch.Tensor | None = None,
    global_num_experts: int = -1,
):
    """
    融合专家选择函数

    Args:
        hidden_states: 隐藏状态，形状为 (num_tokens, hidden_size)
        router_logits: 路由器logits，形状为 (num_tokens, hidden_size)
        top_k: 选择的top k专家数量
        use_grouped_topk: 是否在选择top-k之前对专家进行分组
        renormalize: 是否重新归一化路由权重
        topk_group: 要从中选择的专家组数量
        num_expert_group: 每个组中的专家数量
        custom_routing_function: 自定义路由函数
        scoring_func: 使用的评分函数
        e_score_correction_bias: 应用于专家评分的修正偏置
        global_num_experts: 全局专家数量

    Returns:
        topk_weights: 路由权重，形状为 (num_tokens, top_k)
        topk_ids: 选择的专家ID，形状为 (num_tokens, top_k)
    """
    # 调用原生专家选择函数
    topk_weights, topk_ids = _native_select_experts(
        hidden_states=hidden_states,
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
    # 返回选择的专家权重和ID
    return topk_weights, topk_ids
