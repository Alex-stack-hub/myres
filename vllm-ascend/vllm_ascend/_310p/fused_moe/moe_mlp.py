# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.


import torch
import torch_npu

from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput


def quant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
) -> torch.Tensor:
    """应用量化MLP计算

    Args:
        hidden_states: 隐藏状态
        w1: 第一层权重（量化）
        w1_scale: 第一层权重的缩放因子
        w2: 第二层权重（量化）
        w2_scale: 第二层权重的缩放因子
        group_list: 组列表
        group_list_type: 组列表类型

    Returns:
        计算后的隐藏状态
    """
    # 如果组列表类型为1，将其转换为累积和格式
    if group_list_type == 1:
        # 将组列表从计数格式转换为累积和格式
        group_list = torch.cumsum(group_list, dim=0)

    # 执行量化分组矩阵乘法并反量化
    hidden_states = torch_npu.npu_quant_grouped_matmul_dequant(
        x=hidden_states, quantized_weight=w1, weight_scale=w1_scale, group_list=group_list, quant_mode="pertoken"
    )
    # 应用SiLU激活函数
    hidden_states = torch_npu.npu_swiglu(hidden_states)
    # 执行第二次量化分组矩阵乘法并反量化
    hidden_states = torch_npu.npu_quant_grouped_matmul_dequant(
        x=hidden_states, quantized_weight=w2, weight_scale=w2_scale, group_list=group_list, quant_mode="pertoken"
    )
    # 返回计算后的隐藏状态
    return hidden_states


def unquant_apply_mlp(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, group_list: torch.Tensor, group_list_type: int = 1
) -> torch.Tensor:
    """应用未量化MLP计算

    Args:
        hidden_states: 隐藏状态
        w1: 第一层权重（未量化）
        w2: 第二层权重（未量化）
        group_list: 组列表
        group_list_type: 组列表类型

    Returns:
        计算后的隐藏状态
    """
    # 执行分组矩阵乘法，计算gate和up的输出
    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    # 应用SiLU激活函数
    act_out = torch_npu.npu_swiglu(gate_up_out)

    # 执行第二次分组矩阵乘法，计算down的输出
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[act_out],
        weight=[w2],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    # 返回计算后的隐藏状态
    return hidden_states


def unified_apply_mlp(*, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
    """统一应用MLP计算

    Args:
        mlp_compute_input: MLP计算输入

    Returns:
        计算后的隐藏状态
    """
    # 从输入中提取参数
    hidden_states = mlp_compute_input.hidden_states
    w1 = mlp_compute_input.weights.w1
    w2 = mlp_compute_input.weights.w2
    w1_scale = mlp_compute_input.weights.w1_scale
    w2_scale = mlp_compute_input.weights.w2_scale
    group_list = mlp_compute_input.group_list
    group_list_type = mlp_compute_input.group_list_type
    # 确保w1和w2是张量
    assert isinstance(w1, torch.Tensor)
    assert isinstance(w2, torch.Tensor)

    # 根据是否量化选择不同的实现
    if mlp_compute_input.quant.is_quant:
        # 确保w1_scale和w2_scale是张量且不为None
        assert isinstance(w1_scale, torch.Tensor)
        assert isinstance(w2_scale, torch.Tensor)
        assert w1_scale is not None and w2_scale is not None
        # 调用量化MLP应用函数
        return quant_apply_mlp(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            group_list=group_list,
            group_list_type=group_list_type,
        )

    # 调用未量化MLP应用函数
    return unquant_apply_mlp(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        group_list=group_list,
        group_list_type=group_list_type,
    )
