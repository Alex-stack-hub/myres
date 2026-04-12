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
# 文件功能总结：
# Ascend 310P 专用模型运行器 (Model Runner)
# 继承自 NPUModelRunner，针对 Ascend 310P 硬件特性进行优化
# 主要功能：
# 1. 支持 310P 特定的 ngram 推测解码
# 2. 管理 CUDA 图 (CUDA Graph) 编译和调度
# 3. 处理注意力状态转换 (Prefill/Decode)
# 4. 批量执行和填充策略
# 5. 内存格式管理 (ACL_FORMAT_FRACTAL_NZ)
#

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.logger import logger
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

_NGRAM_GRAPH_UNIFORM_DECODE_QUERY_LEN = 1
_ATTENTION_BLOCK_SIZE_LIMIT = 128 * 128


class NPUModelRunner310(NPUModelRunner):
    """
    Ascend 310P 专用模型运行器

    继承自 NPUModelRunner，针对 Ascend 310P 硬件特性进行优化。
    主要处理 310P 特定的推理逻辑，包括：
    - ngram 推测解码支持
    - CUDA 图编译和调度
    - 注意力状态管理
    - 内存格式配置
    """

    # Inherited from parent runner; annotated here to satisfy strict type checks.
    uniform_decode_query_len: int

    def __init__(self, *args, **kwargs):
        """
        初始化 310P 模型运行器

        参数:
            *args: 传递给父类的参数
            **kwargs: 传递给父类的关键字参数

        功能:
            1. 调用父类初始化
            2. 设置 ACL 内存格式为 FRACTAL_NZ
            3. 如果使用 ngram 推测解码，调整 CUDA 图调度器的查询长度
        """
        super().__init__(*args, **kwargs)
        self._acl_format = ACL_FORMAT_FRACTAL_NZ  # 设置 ACL 内存格式
        if self.speculative_config is not None and self.speculative_config.method == "ngram":
            # 310P ngram 要求解码图的查询长度为 1
            # 保持调度器内部查询长度同步，避免键初始化断言错误
            self.cudagraph_dispatcher.uniform_decode_query_len = _NGRAM_GRAPH_UNIFORM_DECODE_QUERY_LEN

    @contextmanager
    def temporary_modify_uniform_decode_query_len(self):
        """
        临时修改统一解码查询长度的上下文管理器

        说明:
            仅用于 310P ngram 路径，其中调度器使用 q_len=1，
            而运行器默认的 uniform_decode_query_len 保持为 1 + num_spec_tokens。
            TODO: 在上游支持后端特定的解码捕获查询长度后，移除此临时重写。

        功能:
            如果使用 ngram 推测解码，临时将 uniform_decode_query_len 设置为 1，
            并在退出上下文时恢复原值。
        """
        if self.speculative_config is None or self.speculative_config.method != "ngram":
            yield  # 非 ngram 情况，直接返回
            return

        original_uniform_decode_query_len = self.uniform_decode_query_len  # 保存原值
        self.uniform_decode_query_len = _NGRAM_GRAPH_UNIFORM_DECODE_QUERY_LEN  # 临时设置为 1
        try:
            yield  # 执行上下文内的代码
        finally:
            self.uniform_decode_query_len = original_uniform_decode_query_len  # 恢复原值

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = False,
        force_eager: bool = False,
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ):
        """
        确定批量执行和填充策略

        参数:
            num_tokens: 总令牌数
            num_reqs: 请求数量
            num_scheduled_tokens_np: 每个请求的已调度令牌数 (numpy 数组)
            max_num_scheduled_tokens: 最大已调度令牌数
            use_cascade_attn: 是否使用级联注意力
            allow_microbatching: 是否允许微批次 (默认 False)
            force_eager: 是否强制 eager 模式 (默认 False)
            force_uniform_decode: 是否强制统一解码 (默认 None)
            force_has_lora: 是否强制有 LoRA (默认 None)
            force_num_active_loras: 强制活跃 LoRA 数量 (默认 None)
            num_encoder_reqs: 编码器请求数量 (默认 0)

        返回:
            父类的 _determine_batch_execution_and_padding 方法的返回值

        功能:
            1. 根据注意力状态调整 force_eager 标志
            2. 在解码状态下，如果满足条件则强制统一解码
            3. 调用父类方法进行实际的批量执行决策
        """
        # 如果是分块预填充或预填充缓存命中状态，强制使用 eager 模式
        if self.attn_state in (AscendAttentionState.ChunkedPrefill, AscendAttentionState.PrefillCacheHit):
            force_eager = True

        # 如果未指定 force_uniform_decode 且处于仅解码状态
        if force_uniform_decode is None and self.attn_state == AscendAttentionState.DecodeOnly:
            decode_query_len = _NGRAM_GRAPH_UNIFORM_DECODE_QUERY_LEN  # 解码查询长度 = 1
            # 检查是否满足统一解码条件
            if (
                max_num_scheduled_tokens == decode_query_len  # 最大调度令牌数等于解码查询长度
                and num_tokens == max_num_scheduled_tokens * num_reqs  # 总令牌数匹配
                and np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] > 0)  # 所有请求都有已计算的令牌
            ):
                # 尊重显式调用者重写：仅在未设置时强制
                force_uniform_decode = True  # 强制统一解码

        # 调用父类方法进行实际的批量执行决策
        return super()._determine_batch_execution_and_padding(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            use_cascade_attn=use_cascade_attn,
            allow_microbatching=allow_microbatching,
            force_eager=force_eager,
            force_uniform_decode=force_uniform_decode,
            force_has_lora=force_has_lora,
            force_num_active_loras=force_num_active_loras,
            num_encoder_reqs=num_encoder_reqs,
        )

    def _pad_query_start_loc_for_fia(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        batch_desc_num_reqs: int | None = None,
    ) -> int:
        # Keep this aligned with the dispatcher because batch_desc.num_reqs is
        # generated by dispatcher._create_padded_batch_descriptor().
        # For 310P ngram we intentionally set dispatcher q_len=1, while runner's
        # default uniform_decode_query_len may remain 1 + num_spec_tokens.
        uniform_decode_query_len = self.cudagraph_dispatcher.uniform_decode_query_len

        if num_tokens_padded == num_reqs_padded * uniform_decode_query_len:
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            last_loc = self.query_start_loc.np[num_reqs]
            self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1] = (
                self.arange_np[1 : num_reqs_padded + 1 - num_reqs] * uniform_decode_query_len + last_loc
            )
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded

            # Insert a dummy request instead of setting query_start_loc[num_reqs] = num_tokens_padded directly
            self.query_start_loc.np[num_reqs_padded + 1] = num_tokens_padded
            num_reqs_padded = num_reqs_padded + 1

        self.query_start_loc.copy_to_gpu()
        return num_reqs_padded

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        cudagraph_runtime_mode=None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
    ):
        temporary_context = self.temporary_modify_uniform_decode_query_len() if uniform_decode else nullcontext()
        with temporary_context:
            return super()._dummy_run(
                num_tokens=num_tokens,
                with_prefill=with_prefill,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                force_attention=force_attention,
                uniform_decode=uniform_decode,
                is_profile=is_profile,
                create_mixed_batch=create_mixed_batch,
                allow_microbatching=allow_microbatching,
                skip_eplb=skip_eplb,
                remove_lora=remove_lora,
                is_graph_capturing=is_graph_capturing,
                num_active_loras=num_active_loras,
                profile_seq_lens=profile_seq_lens,
            )

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends,
        kv_cache_groups,
    ) -> None:
        # 910B does not need this branch because runner/dispatcher query_len are
        # naturally consistent there. 310P ngram needs temporary alignment.
        with self.temporary_modify_uniform_decode_query_len():
            super()._check_and_update_cudagraph_mode(attention_backends, kv_cache_groups)

    def initialize_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Override the base class method.
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # 310P limitation: KV transfer is not supported
        if self.vllm_config.kv_transfer_config is not None:
            raise ValueError("KV cache transfer is not supported for 310P.")
        if self.use_sparse:
            raise ValueError("Deepseek Sparse Attention is not supported for 310P.")
        if self.model_config.use_mla:
            raise ValueError("MLAAttention is not supported for 310P.")
        # Initialize the memory buffer for KV cache
        kv_caches = self._allocate_kv_cache_tensors(kv_cache_config)
        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        from vllm.v1.worker.utils import bind_kv_cache

        bind_kv_cache(kv_caches, self.compilation_config.static_forward_context, self.kv_caches)
        return kv_caches

    def _allocate_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache size. The buffer needs to be reshaped to the desired shape before being used by
        the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer.
        """
        # init kv cache tensors
        kv_cache: dict[str, list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]] = {}
        # get kv cache spec for each layer
        layer_kv_cache_spec: dict[str, KVCacheSpec] = {}
        for group_kv_cache_spec in kv_cache_config.kv_cache_groups:
            for layer_name in group_kv_cache_spec.layer_names:
                layer_kv_cache_spec[layer_name] = group_kv_cache_spec.kv_cache_spec
        # Allocate kv cache buffers according to the kv_cache_config and kv_cache_spec
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            for idx in range(len(kv_cache_tensor.shared_by)):
                layer_name = kv_cache_tensor.shared_by[idx]
                if layer_name in self.runner_only_attn_layers:
                    continue
                if "linear_attn" in layer_name and layer_name not in kv_cache:
                    cache_spec = layer_kv_cache_spec[layer_name]
                    assert isinstance(cache_spec, MambaSpec)
                    assert kv_cache_tensor.size % cache_spec.page_size_bytes == 0
                    num_blocks = kv_cache_tensor.size // cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks
                    raw_tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
                    state_tensors = []
                    target_idx = 0
                    start_idx = 0
                    for shape, dtype in zip(cache_spec.shapes, cache_spec.dtypes):
                        target_shape = (num_blocks, *shape)
                        target_idx += math.prod(target_shape) * get_dtype_size(dtype)
                        tensor = raw_tensor[start_idx:target_idx].view(dtype).view(target_shape)
                        start_idx = target_idx
                        state_tensors.append(tensor)
                    for layer_name_inner in kv_cache_tensor.shared_by:
                        if "linear_attn" in layer_name_inner:
                            kv_cache[layer_name_inner] = state_tensors
                elif "attn" in layer_name and layer_name not in kv_cache:
                    kv_cache_spec = layer_kv_cache_spec[layer_name]
                    assert isinstance(kv_cache_spec, AttentionSpec)
                    assert kv_cache_tensor.size % kv_cache_spec.page_size_bytes == 0
                    num_blocks = kv_cache_tensor.size // kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks
                    # Page attention operation on 310P limits block_size * head_size <= 128 * 128
                    supported_sizes = [
                        support_size
                        for support_size in self.attn_backend.get_supported_kernel_block_sizes()
                        if support_size * kv_cache_spec.head_size <= _ATTENTION_BLOCK_SIZE_LIMIT
                    ]
                    if supported_sizes:
                        block_size = supported_sizes[0]
                        block_size_chunk = kv_cache_spec.block_size // block_size
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks * block_size_chunk,
                            block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size,
                        )
                    else:
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks, kv_cache_spec.block_size, kv_cache_spec.num_kv_heads, kv_cache_spec.head_size
                        )
                    k_shape = kv_cache_shape[1:]
                    v_shape = k_shape
                    dtype = kv_cache_spec.dtype
                    k_cache = torch_npu.empty_with_format(
                        size=k_shape, dtype=dtype, device=self.device, acl_format=self._acl_format
                    )
                    v_cache = torch_npu.empty_with_format(
                        size=v_shape, dtype=dtype, device=self.device, acl_format=self._acl_format
                    )
                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if "attn" in layer_name_inner and "linear_attn" not in layer_name_inner:
                            kv_cache[layer_name_inner] = (k_cache, v_cache)
        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache.keys()), "Some layers are not correctly initialized"
        return kv_cache

    # Override this function because of tensor.copy_(other) accuracy issue.
    # TODO: This override will be removed after tensor.copy_(other) accuracy issue is resolved.
    def _prepare_input_ids(
        self,
        scheduler_output: SchedulerOutput,
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the NPU from prev_sampled_token_ids.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_common_req_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        indices_match = True
        max_flattened_index = -1
        total_num_spec_tokens = 0
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                draft_len = len(scheduled_spec_tokens.get(req_id, ()))
                total_num_spec_tokens += draft_len
                flattened_index = cu_num_tokens[cur_index].item() - 1
                sample_flattened_indices.append(flattened_index - draft_len)
                spec_flattened_indices.extend(range(flattened_index - draft_len + 1, flattened_index + 1))
                start = prev_index * self.num_spec_tokens
                prev_draft_token_indices.extend(range(start, start + draft_len))
                indices_match &= prev_index == flattened_index
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_common_tokens = len(sample_flattened_indices)
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
        if num_common_tokens < total_without_spec:
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_common_tokens == 0:
            return
        if indices_match and max_flattened_index == (num_common_tokens - 1):
            # NOTE: Override the copy_ function here
            indices = torch.arange(num_common_tokens, device=self.input_ids.gpu.device)
            source = self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0]
            self.input_ids.gpu.index_copy_(0, indices, source)
            if self.enable_prompt_embeds:
                self.is_token_ids.gpu[:num_common_tokens] = True
            return
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[prev_common_req_indices_tensor, 0],
        )
        # Scatter the draft tokens after the sampled tokens are scattered.
        if self._draft_token_ids is None or not spec_flattened_indices:
            return
        assert isinstance(self._draft_token_ids, torch.Tensor)
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )

    def may_reinitialize_input_batch(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
            if not isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec)
        ]

        # Generate kernel_block_sizes that matches each block_size
        # For attention backends that support virtual block splitting,
        # use the supported block sizes from the backend
        # For other backends (like Mamba), use [0] (no splitting)
        self.kernel_block_sizes = []
        for kv_cache_group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            elif isinstance(kv_cache_spec, AttentionSpec):
                try:
                    attn_groups = self.attn_groups[kv_cache_group_id]
                    backend = attn_groups[0].backend
                    # Page attention operation on 310P limits block_size * head_size <= 128 * 128
                    supported_sizes = [
                        support_size
                        for support_size in backend.get_supported_kernel_block_sizes()
                        if support_size * kv_cache_spec.head_size <= _ATTENTION_BLOCK_SIZE_LIMIT
                    ]
                    kernel_block_size_list = supported_sizes if supported_sizes else [self.cache_config.block_size]
                except IndexError:
                    kernel_block_size_list = [self.cache_config.block_size]
                self.kernel_block_sizes.append(kernel_block_size_list)
            else:
                self.kernel_block_sizes.append([0])

        if block_sizes != [self.cache_config.block_size] or self.kernel_block_sizes != [[self.cache_config.block_size]]:
            assert self.offload_config.uva.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details."
            )
            self.input_batch = NPUInputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max(self.model_config.max_model_len, self.max_encoder_len),
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                is_pooling_model=self.is_pooling_model,
                num_speculative_tokens=(
                    self.vllm_config.speculative_config.num_speculative_tokens
                    if self.vllm_config.speculative_config
                    else 0
                ),
                kernel_block_sizes=self.kernel_block_sizes,
            )
