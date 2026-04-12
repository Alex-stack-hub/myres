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
from __future__ import annotations

import torch

from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput

from .moe_mlp import unified_apply_mlp
from .token_dispatcher import TokenDispatcherWithAllGather310


class AllGatherCommImpl310(AllGatherCommImpl):
    """Ascend 310P平台的AllGather通信实现

    该实现与NativeAllGatherCommImpl相同，但使用NPU特定的操作以获得更好的性能。

    该实现应该与所有场景兼容，因此是MoE通信方法的默认实现。
    它使用`torch_npu.npu_moe_init_routing_v2`进行预处理，
    使用`torch_npu.npu_moe_token_unpermute`进行后处理，
    以高效处理token到专家的映射和通信。
    """

    def __init__(self, moe_config):
        """初始化AllGatherCommImpl310

        Args:
            moe_config: MoE配置对象
        """
        # 调用父类初始化方法
        super().__init__(moe_config)
        # 设置是否使用融合操作
        self.use_fusion_ops = False

    def _apply_mlp(self, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
        """应用MLP计算

        Args:
            mlp_compute_input: MLP计算输入

        Returns:
            计算结果
        """
        # 调用统一的MLP应用函数
        return unified_apply_mlp(mlp_compute_input=mlp_compute_input)

    def _get_token_dispatcher(self):
        """获取token调度器

        Returns:
            TokenDispatcherWithAllGather310实例
        """
        # 返回TokenDispatcherWithAllGather310实例
        return TokenDispatcherWithAllGather310(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )
