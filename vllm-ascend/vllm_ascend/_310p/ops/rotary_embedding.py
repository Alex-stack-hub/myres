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
import torch_npu  # NPU相关操作库

from vllm_ascend.ops.rotary_embedding import (  # 基础旋转嵌入类和获取cos/sin切片的函数
    AscendRotaryEmbedding,
    get_cos_and_sin_slice,
)


def _rope_forward_oot(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    is_neox_style: bool,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """旋转位置编码的前向传播实现

    Args:
        self: AscendRotaryEmbedding310实例
        positions: 位置张量
        query: 查询张量
        key: 键张量
        is_neox_style: 是否使用NeoX风格的旋转编码
        offsets: 偏移量张量（当前不支持）

    Returns:
        应用旋转编码后的query和key张量
    """
    # 保存原始形状，用于后续恢复
    query_shape, key_shape = query.shape, key.shape
    # 确保cos_sin_cache与query在同一设备
    if self.cos_sin_cache.device != query.device:
        self.cos_sin_cache = self.cos_sin_cache.to(query.device)
    # 确保cos_sin_cache与query数据类型一致
    if self.cos_sin_cache.dtype != query.dtype:
        self.cos_sin_cache = self.cos_sin_cache.to(query.dtype)
    # 获取cos和sin切片
    cos, sin = get_cos_and_sin_slice()
    # 检查是否支持偏移量
    if offsets is not None:
        raise NotImplementedError("Batched rotary embedding is currently not supported on NPU.")
    # 设置旋转模式
    rotary_mode = "half" if is_neox_style else "interleave"

    # 处理head_size为128的特殊情况
    if self.head_size == 128 and self.cos_sin_cache.shape[-1] == 128:
        # 重塑query和key为4维张量
        query = query.contiguous().view(1, query.shape[0], -1, self.head_size)
        key = key.contiguous().view(1, key.shape[0], -1, self.head_size)
        # 使用NPU原生的旋转位置编码操作
        query, key = torch_npu.npu_apply_rotary_pos_emb(query, key, cos, sin, rotary_mode=rotary_mode)
    # 处理rotary_dim小于head_size的情况
    elif self.rotary_dim < self.head_size:
        # 获取token数量
        num_tokens = query.shape[0]
        # 重塑query和key
        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)
        # 分离需要旋转和不需要旋转的部分
        q_rot = query[..., : self.rotary_dim]  # 需要旋转的部分
        q_pass = query[..., self.rotary_dim :]  # 不需要旋转的部分
        k_rot = key[..., : self.rotary_dim]  # 需要旋转的部分
        k_pass = key[..., self.rotary_dim :]  # 不需要旋转的部分

        # 处理rotary_dim为64的特殊情况
        if self.rotary_dim == 64:
            # 重塑为4维张量
            q_rot = q_rot.contiguous().view(1, num_tokens, -1, self.rotary_dim)
            k_rot = k_rot.contiguous().view(1, num_tokens, -1, self.rotary_dim)
            # 使用NPU原生的旋转位置编码操作
            q_rot, k_rot = torch_npu.npu_apply_rotary_pos_emb(q_rot, k_rot, cos, sin, rotary_mode=rotary_mode)
        else:
            # 重塑为2维张量
            q_rot = q_rot.contiguous().view(num_tokens, -1)
            k_rot = k_rot.contiguous().view(num_tokens, -1)
            # 使用NPU原生的旋转嵌入操作
            torch_npu._npu_rotary_embedding(
                positions,
                q_rot,
                k_rot,
                self.rotary_dim,
                self.cos_sin_cache,
                is_neox_style,
            )

        # 重塑回原始形状
        q_rot = q_rot.view(num_tokens, -1, self.rotary_dim)
        k_rot = k_rot.view(num_tokens, -1, self.rotary_dim)
        # 拼接旋转和非旋转部分
        query = torch.cat((q_rot, q_pass), dim=-1).reshape(query_shape)
        key = torch.cat((k_rot, k_pass), dim=-1).reshape(key_shape)
    # 处理其他情况
    else:
        # 重塑为2维张量
        query = query.contiguous().view(query.shape[0], -1)
        key = key.contiguous().view(key.shape[0], -1)
        # 使用NPU原生的旋转嵌入操作
        torch_npu._npu_rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            is_neox_style,
        )

    # 恢复原始形状并返回
    return query.view(query_shape), key.view(key_shape)


class AscendRotaryEmbedding310(AscendRotaryEmbedding):
    """Ascend 310P平台的旋转位置编码实现

    继承自AscendRotaryEmbedding，专门为Ascend 310P平台优化，
    实现了forward_oot方法，使用NPU原生的旋转嵌入操作。
    """

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor | None = None,
        is_neox_style_override: bool | None = None,
    ):
        """前向传播

        Args:
            positions: 位置张量
            query: 查询张量
            key: 键张量
            offsets: 偏移量张量（当前不支持）
            is_neox_style_override: 是否覆盖默认的NeoX风格设置

        Returns:
            应用旋转编码后的query和key张量
        """
        # 获取默认的is_neox_style设置
        is_neox_style = self.is_neox_style
        # 如果提供了覆盖值，使用覆盖值
        if is_neox_style_override is not None:
            is_neox_style = is_neox_style_override
        # 调用_rope_forward_oot函数执行旋转编码
        return _rope_forward_oot(self, positions, query, key, is_neox_style, offsets)
