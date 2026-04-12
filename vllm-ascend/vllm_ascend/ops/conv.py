#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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


import torch
from vllm.model_executor.layers.conv import Conv2dLayer, Conv3dLayer


class AscendConv2dLayer(Conv2dLayer):
    """Ascend平台的2D卷积层实现"""

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播实现（out-of-tree）

        Args:
            x: 输入张量

        Returns:
            卷积操作后的输出张量
        """
        # 在Ascend NPU上使用aclnn BatchMatMulV2以获得更好的性能
        return self._forward_conv(x)


class AscendConv3dLayer(Conv3dLayer):
    """Ascend平台的3D卷积层实现"""

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播实现（out-of-tree）

        Args:
            x: 输入张量

        Returns:
            卷积操作后的输出张量
        """
        # 在Ascend NPU上使用aclnn BatchMatMulV2以获得更好的性能
        return self._forward_conv(x)
