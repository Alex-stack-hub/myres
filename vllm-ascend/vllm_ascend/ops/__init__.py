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
from vllm.triton_utils import HAS_TRITON

# 导入融合MoE实现
import vllm_ascend.ops.fused_moe.fused_moe  # noqa

# 导入层归一化实现
import vllm_ascend.ops.layernorm  # noqa

# 导入自定义操作注册
import vllm_ascend.ops.register_custom_ops  # noqa

# 如果有Triton支持，导入Triton相关的线性归一化实现
if HAS_TRITON:
    import vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_rope  # noqa
    import vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_mrope
    import vllm_ascend.ops.triton.linearnorm.split_qkv_tp_rmsnorm_rope

# 导入词汇并行嵌入
import vllm_ascend.ops.vocab_parallel_embedding  # noqa

# 从activation模块导入激活函数
from vllm_ascend.ops.activation import AscendQuickGELU, AscendSiluAndMul

# 从rotary_embedding模块导入旋转嵌入
from vllm_ascend.ops.rotary_embedding import AscendDeepseekScalingRotaryEmbedding, AscendRotaryEmbedding


class dummyFusionOp:
    """虚拟融合操作类"""

    # 默认属性
    default = None

    def __init__(self, name=""):
        """初始化虚拟融合操作

        Args:
            name: 操作名称
        """
        self.name = name


def register_dummy_fusion_op() -> None:
    """注册虚拟融合操作

    为PyTorch操作注册表添加虚拟的融合操作，用于在不支持某些操作的环境中提供兼容性。
    """
    # 注册各种虚拟融合操作
    torch.ops._C_ascend.rms_norm = dummyFusionOp(name="rms_norm")
    torch.ops._C_ascend.fused_add_rms_norm = dummyFusionOp(name="fused_add_rms_norm")
    torch.ops._C_ascend.static_scaled_fp8_quant = dummyFusionOp(name="static_scaled_fp8_quant")
    torch.ops._C_ascend.dynamic_scaled_fp8_quant = dummyFusionOp(name="dynamic_scaled_fp8_quant")
    torch.ops._C_ascend.dynamic_per_token_scaled_fp8_quant = dummyFusionOp(name="dynamic_per_token_scaled_fp8_quant")
    torch.ops._C_ascend.rms_norm_static_fp8_quant = dummyFusionOp(name="rms_norm_static_fp8_quant")
    torch.ops._C_ascend.fused_add_rms_norm_static_fp8_quant = dummyFusionOp(name="fused_add_rms_norm_static_fp8_quant")
    torch.ops._C_ascend.rms_norm_dynamic_per_token_quant = dummyFusionOp(name="rms_norm_dynamic_per_token_quant")


# 导出的符号列表
__all__ = ["AscendQuickGELU", "AscendSiluAndMul", "AscendRotaryEmbedding", "AscendDeepseekScalingRotaryEmbedding"]
