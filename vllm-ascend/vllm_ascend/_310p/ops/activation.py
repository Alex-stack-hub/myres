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

# 导入必要的库
import torch  # PyTorch核心库
import torch.nn.functional as F  # PyTorch函数式API
import torch_npu  # NPU相关操作库

# 从基础模块导入AscendSiluAndMul类
from vllm_ascend.ops.activation import AscendSiluAndMul


class AscendSiluAndMul310(AscendSiluAndMul):
    """Ascend 310P平台的SiLU和Mul操作融合实现

    该类继承自AscendSiluAndMul，专门为Ascend 310P平台优化，
    根据输入张量的维度选择最优的实现方式。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量，形状为 (..., 2 * hidden_size)，其中最后一维是隐藏层大小的两倍

        Returns:
            输出张量，形状为 (..., hidden_size)，经过SiLU激活和乘法操作后的结果
        """
        # 检查输入张量的最后一维是否是32的倍数
        # 32是NPU硬件的一个优化维度，对于32的倍数，使用原生操作会更高效
        if x.shape[-1] % 32 == 0:
            # 如果是32的倍数，使用NPU原生的SWIGLU操作
            # SWIGLU是SiLU和Mul的融合操作，性能更优
            out = torch_npu.npu_swiglu(x)
        else:
            # 如果不是32的倍数，手动计算SiLU和Mul操作
            # 计算隐藏层大小，即输入张量最后一维的一半
            h = x.shape[-1] // 2
            # 对前半部分应用SiLU激活函数，然后与后半部分相乘
            # x[..., :h] 取输入张量的前半部分
            # x[..., h:] 取输入张量的后半部分
            out = F.silu(x[..., :h]) * x[..., h:]
        # 返回计算结果
        return out
