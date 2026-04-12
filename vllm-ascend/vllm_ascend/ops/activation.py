#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

import torch
from vllm.model_executor.layers.activation import QuickGELU, SiluAndMul, SwigluOAIAndMul

from vllm_ascend.utils import get_weight_prefetch_method


class AscendQuickGELU(QuickGELU):
    """Ascend平台的QuickGELU激活函数实现"""

    def forward_oot(self, x: torch.tensor) -> torch.Tensor:
        """前向传播实现（out-of-tree）

        Args:
            x: 输入张量

        Returns:
            应用GELU激活函数后的输出张量
        """
        # 导入torch_npu模块
        import torch_npu

        # 使用NPU的fast_gelu操作
        out = torch_npu.npu_fast_gelu(x)
        return out


class AscendSiluAndMul(SiluAndMul):
    """Ascend平台的SiluAndMul激活函数实现"""

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播实现（out-of-tree）

        Args:
            x: 输入张量

        Returns:
            应用SiLU和Mul操作后的输出张量
        """
        # 导入torch_npu模块
        import torch_npu

        # 获取权重预取方法
        weight_prefetch_method = get_weight_prefetch_method()
        # 预取MLP下权重
        weight_prefetch_method.maybe_prefetch_mlp_weight_preprocess(weight_prefetch_method.MLP_DOWN, x)
        # 使用NPU的swiglu操作
        out = torch_npu.npu_swiglu(x)
        # 后处理MLP权重预取
        weight_prefetch_method.maybe_prefetch_mlp_weight_postprocess(out)
        return out


class AscendSwigluOAIAndMul:
    """Ascend平台的SwigluOAIAndMul激活函数实现"""

    @staticmethod
    def swiglu_oai_forward(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
        """SwigluOAI前向传播

        Args:
            x: 输入张量
            alpha: alpha参数
            limit: 限制值

        Returns:
            应用SwigluOAI操作后的输出张量
        """

        # 创建最小化的SwigluOAIAndMul类
        class MinimalSwigluOAIAndMul:
            def __init__(self):
                self.alpha = alpha
                self.limit = limit

        # 创建实例
        layer = MinimalSwigluOAIAndMul()
        # 调用原生forward方法
        return SwigluOAIAndMul.forward_native(layer, x)
