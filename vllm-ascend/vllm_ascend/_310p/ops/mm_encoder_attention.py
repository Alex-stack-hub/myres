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

# 导入必要的库
import einops  # 张量重排库
import torch  # PyTorch核心库
import torch.nn.functional as F  # PyTorch函数式API
import torch_npu  # NPU相关操作库
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention  # 上游的多模态编码器注意力

# 权重填充的最小和最大尺寸
MIN_PAD_SIZE: int = 64  # 权重填充的最小尺寸
MAX_PAD_SIZE: int = 128  # 权重填充的最大尺寸

# 使用seq_lens CPU缓存来避免频繁的设备到主机的复制
# AscendMMEncoderAttention310在每次前向传播时都会将cu_seqlens从NPU复制到CPU，
# 因为_npu_flash_attention_unpad()操作需要CPU上的cu_seqlens（否则会失败）。
# 因此，我们使用seq_lens_cpu_cache来缓存这个张量，因为它在所有层之间共享，
# 但在不同的前向传播步骤中可能会改变。当当前层索引为0时，我们更新缓存，
# 否则直接使用缓存以避免频繁的差异和复制操作，这些操作成本很高。
seq_lens_cpu_cache: torch.Tensor = None


class AscendMMEncoderAttention310(MMEncoderAttention):
    """Ascend 310P平台的多模态编码器注意力实现

    继承自MMEncoderAttention，专门为Ascend 310P平台优化，
    实现了forward_oot方法，使用NPU原生的Flash Attention操作。
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
    ) -> None:
        """初始化多模态编码器注意力层

        Args:
            num_heads: 每个分区的注意力头数量
            head_size: 每个注意力头的隐藏层大小
            scale: 缩放因子
            num_kv_heads: KV头的数量
            prefix: 前缀，仅用于在Attention和MMEncoderAttention之间切换
        """
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )

        # 当head_size在MIN_PAD_SIZE和MAX_PAD_SIZE之间时，启用填充
        self.enable_pad = self.head_size > MIN_PAD_SIZE and self.head_size < MAX_PAD_SIZE
        # 计算缩放值
        self.scale_value = self.head_size**-0.5

    def _reshape_qkv_to_3d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """将query、key、value重塑为3D张量

        Args:
            query: 查询张量
            key: 键张量
            value: 值张量
            bsz: 批次大小
            q_len: 查询序列长度
            kv_len: KV序列长度

        Returns:
            重塑后的query、key、value张量，形状为 (batch_size * seq_len, num_heads, head_size)
        """
        # 重塑query、key、value为3D张量
        query = query.view(bsz * q_len, self.num_heads, self.head_size)
        key = key.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        # 计算每个KV头对应的查询头数量
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        # 处理MQA（多头注意力）和GQA（分组查询注意力）
        if (num_repeat := self.num_queries_per_kv) > 1:
            # 重复KV头以匹配查询头数量
            key = torch.repeat_interleave(key, num_repeat, dim=1)
            value = torch.repeat_interleave(value, num_repeat, dim=1)

        return query, key, value

    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        sequence_lengths: torch.Tensor | None = None,
    ):
        """前向传播

        Args:
            query: 查询张量
            key: 键张量
            value: 值张量
            cu_seqlens: 累积序列长度张量
            max_seqlen: 最大序列长度（仅用于Flash Attention）
            sequence_lengths: 序列长度张量

        Returns:
            注意力输出张量
        """
        # 获取批次大小和查询序列长度
        bsz, q_len = query.size()[:2]
        # 获取KV序列长度
        kv_len = key.size(1)
        # 检查query是否已经是4维张量
        is_reshaped = query.dim() == 4

        # 直接使用seq_lens CPU缓存来避免设备到主机的复制
        if cu_seqlens is None:
            # 如果没有提供cu_seqlens，创建一个CPU上的累积序列长度张量
            cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device="cpu")
        # 计算序列长度并移至CPU
        seq_lens_cpu = torch.diff(cu_seqlens).to("cpu")

        # 重塑q、k、v为3D张量：[b, s, head, head_dim] -> [b * s, head, head_dim]
        q, k, v = self._reshape_qkv_to_3d(query, key, value, bsz, q_len, kv_len)

        # 如果启用填充，将张量填充到MAX_PAD_SIZE
        if self.enable_pad:
            origin_shape = q.shape[-1]
            pad_len = MAX_PAD_SIZE - origin_shape
            # 填充q、k、v到MAX_PAD_SIZE
            q = F.pad(q, (0, pad_len), mode="constant", value=0)
            k = F.pad(k, (0, pad_len), mode="constant", value=0)
            v = F.pad(v, (0, pad_len), mode="constant", value=0)

        # 创建与q形状相同的输出张量
        context_layer = torch.empty_like(q)

        # 使用NPU原生的Flash Attention操作
        # 注意：此操作需要pta版本 >= 2.5.1
        torch_npu._npu_flash_attention_unpad(
            query=q,
            key=k,
            value=v,
            seq_len=seq_lens_cpu,
            scale_value=self.scale_value,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            out=context_layer,
        )

        # 如果启用了填充，裁剪回原始形状
        if self.enable_pad:
            context_layer = context_layer[..., :origin_shape]

        # 根据输入形状重塑输出
        if is_reshaped:
            # 如果输入是4维张量，输出保持4维：[b * s, h, d] -> [b, s, h, d]
            context_layer = einops.rearrange(context_layer, "(b s) h d -> b s h d", b=bsz).contiguous()
        else:
            # 如果输入不是4维张量，输出将头维度合并：[b * s, h, d] -> [b, s, (h d)]
            context_layer = einops.rearrange(context_layer, "(b s) h d -> b s (h d)", b=bsz).contiguous()
        # 返回注意力输出
        return context_layer
