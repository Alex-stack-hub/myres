#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# 文件功能总结：
# Ascend 310P 专用注意力后端实现
# 继承自 AscendAttentionBackend，提供 310P 特定的注意力计算后端
# 主要功能：
# 1. 注册 310P 注意力后端到 vLLM 后端注册表
# 2. 提供 KV 缓存形状计算
# 3. 返回 310P 特定的元数据构建器和掩码构建器
# 4. 获取 310P 特定的注意力实现
#

from typing import Any

import torch_npu
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)

from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310
from vllm_ascend._310p.attention.metadata_builder import AscendAttentionMetadataBuilder310
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend310(AscendAttentionBackend):
    """
    Ascend 310P 专用注意力后端

    继承自 AscendAttentionBackend，为 310P 设备提供特定的注意力计算实现。
    通过 @register_backend 装饰器注册到 vLLM 后端注册表。
    """

    def __init__(self, *args, **kwargs):
        """
        初始化 310P 后端并设置设备特定的掩码构建器

        参数:
            *args: 传递给父类的参数
            **kwargs: 传递给父类的关键字参数
        """
        super().__init__(*args, **kwargs)  # 调用父类初始化

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ):
        """
        确定键值 (KV) 缓存张量的形状

        310P 硬件需要特定的内存对齐以获得最佳性能。
        此方法定义一个 5D 张量形状，其中头部大小维度被拆分以确保对齐到 16 的倍数。

        参数:
            num_blocks (int): 内存块数量
            block_size (int): 每个块的大小
            num_kv_heads (int): KV 头数量
            head_size (int): 每个头的维度大小
            cache_type (str): 缓存类型（默认空字符串）

        返回:
            tuple: 硬件所需的特定 5D 形状 (2, num_blocks, hidden_dim_aligned, block_size, 16)
        """
        # 对齐到 16 的倍数，这是 310P 设备的要求
        return (
            2,
            num_blocks,
            (num_kv_heads * head_size) // 16,
            block_size,
            16,
        )  # 5D 形状: (2, num_blocks, 对齐后的隐藏维度, block_size, 16)

    @staticmethod
    def get_impl_cls():
        """
        返回注意力操作的实现类

        返回:
            AscendAttentionBackendImpl310: 310P 注意力后端实现类
        """
        return AscendAttentionBackendImpl310  # 返回 310P 注意力后端实现类

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        """
        返回 310P 特定的元数据构建器类

        返回:
            AscendAttentionMetadataBuilder310: 310P 元数据构建器类
        """
        return AscendAttentionMetadataBuilder310  # 返回 310P 元数据构建器类

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        """
        返回支持的内核块大小

        返回:
            list[int]: 支持的内核块大小列表 [128, 64]
        """
        return [128, 64]  # 310P 支持的内核块大小: 128 和 64


class AscendAttentionBackendImpl310(AscendAttentionBackendImpl):
    """
    Implementation of attention operations (Prefill, Decode, Chunked Prefill)
    optimized for the Ascend 310P architecture.
    """

    def forward_paged_attention(
        self,
        query: Any,
        attn_metadata: AscendMetadata,
        output: Any | None = None,
    ) -> Any:
        """
        Executes Paged Attention (typically for the decode phase).

        Ensures that the sequence length metadata is on the correct device
        before invoking the base implementation.

        Args:
            query (Any): The query tensor.
            attn_metadata (AscendMetadata): Metadata associated with the attention request.
            output (Any | None): Optional output tensor.

        Returns:
            Any: The result of the attention operation.
        """
        if attn_metadata.seq_lens.device != query.device:
            attn_metadata.seq_lens = attn_metadata.seq_lens.to(
                device=query.device,
                non_blocking=True,
            )

        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        return output

    def forward_prefill_310(self, query, key, value, attn_metadata, output):
        """
        Executes Flash Attention for the prefill phase on 310P.

        This method handles memory alignment padding. If the query shape implies
        padding (aligned_tokens > real_tokens), it adjusts the sequence length
        of the last request to account for the delta, ensuring the NPU operator
        processes the data correctly.

        Args:
            query, key, value: Input tensors.
            attn_metadata (AscendMetadata): Attention metadata containing masks and seq_lens.
            output: Output tensor.

        Returns:
            The output tensor after flash attention.
        """
        real_tokens = int(attn_metadata.seq_lens.sum().item())
        seq_len = attn_metadata.seq_lens
        aligned_tokens = int(query.shape[0])
        delta = aligned_tokens - real_tokens

        # Adjust sequence length if padding (alignment) was applied to the inputs
        if delta:
            seq_len = seq_len.clone()
            seq_len[-1] += delta

        mask = attn_metadata.attn_mask
        torch_npu._npu_flash_attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            seq_len=seq_len,
            scale_value=self.scale,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            out=output,
        )
        return output

    def forward_chunked_prefill_310(self, query, attn_metadata, output):
        """
        Executes SplitFuse (Chunked Prefill) attention on 310P.

        This handles scenarios where the prefill is split into chunks. It prepares
        the necessary metadata (query lengths, block tables) and generates the
        specific splitfuse mask before calling the NPU operator.

        Args:
            query: The query tensor.
            attn_metadata (AscendMetadata): Metadata containing start locations and block tables.
            output: The output tensor.
        """
        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        query = query[:num_actual_tokens]
        output = output[:num_actual_tokens]

        # Calculate query lengths from start locations
        qsl_cpu = attn_metadata.query_start_loc.cpu()
        qlens = qsl_cpu[1:] - qsl_cpu[:-1]

        context_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_tables

        # Generate the specific mask for splitfuse
        mask = AttentionMaskBuilder310.get_splitfuse_mask(attn_metadata, query.device)

        if context_lens.device != query.device:
            context_lens = context_lens.to(query.device, non_blocking=True)

        torch_npu._npu_paged_attention_splitfuse(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            mask=mask,
            block_table=block_table,
            seq_len=qlens,
            context_lens=context_lens,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            out=output,
        )

        return output

    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        """
        Main dispatch method for attention operations.

        Routes the execution to Decode, Prefill, or Chunked Prefill methods
        based on the current attention state found in metadata.

        Args:
            query, key, value: Input tensors (Key/Value usually empty for decode/chunked).
            kv_cache: The KV cache structure.
            attn_metadata: Metadata determining the state (Prefill vs Decode).
            output: Tensor to write results to.

        Returns:
            The output tensor.

        Raises:
            NotImplementedError: If the attention state is not supported on 310P.
        """
        state = attn_metadata.attn_state
        # Condition for PrefillNoCache: No previous tokens have been processed yet
        if state == AscendAttentionState.PrefillNoCache:
            output = self.forward_prefill_310(query, key, value, attn_metadata, output)
        # Condition for DecodeOnly: Pure decoding phase where each request generates one token
        elif state == AscendAttentionState.DecodeOnly:
            output = self.forward_paged_attention(query, attn_metadata, output)
        # Condition for ChunkedPrefill:
        # 1. During speculative decoding scenarios (except mtp)
        # 2. Processing large prefill requests in chunks
        # Condition for PrefillCacheHit: Indicates prefill with some cached tokens already processed
        elif state in [AscendAttentionState.ChunkedPrefill, AscendAttentionState.PrefillCacheHit]:
            output = self.forward_chunked_prefill_310(query, attn_metadata, output)
        # Condition for SpecDecoding: Specified for mtp, which is not supported yet.
        else:
            raise NotImplementedError(f"AscendAttentionState: {state} is not supported for 310P currently.")
        return output
