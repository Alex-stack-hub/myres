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

from __future__ import annotations

# 导入必要的库
import torch  # PyTorch核心库
import torch.nn.functional as F  # PyTorch函数式API
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig  # 量化配置
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,  # 默认词汇填充大小
    UnquantizedEmbeddingMethod,  # 未量化嵌入方法
)

from vllm_ascend.ops.vocab_parallel_embedding import (  # 基础Ascend词汇并行嵌入类
    AscendParallelLMHead,
    AscendVocabParallelEmbedding,
)
from vllm_ascend.utils import maybe_trans_nz  # 可能的NZ转换函数


class AscendUnquantizedEmbeddingMethod310(UnquantizedEmbeddingMethod):
    """Ascend 310P平台的未量化嵌入方法

    继承自UnquantizedEmbeddingMethod，专门为Ascend 310P平台优化，
    实现了权重处理和应用方法。
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """加载后处理权重

        Args:
            layer: 嵌入层模块
        """
        # 对权重进行可能的NZ转换
        layer.weight_nz = maybe_trans_nz(layer.weight)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """应用嵌入

        Args:
            layer: 嵌入层模块
            x: 输入张量
            bias: 偏置张量

        Returns:
            应用嵌入后的输出张量
        """
        # 使用处理后的权重进行线性变换
        return F.linear(x, layer.weight_nz, bias)


class AscendVocabParallelEmbedding310(AscendVocabParallelEmbedding):
    """Ascend 310P平台的词汇并行嵌入

    继承自AscendVocabParallelEmbedding，专门为Ascend 310P平台优化，
    使用AscendUnquantizedEmbeddingMethod310作为未量化方法。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        """初始化词汇并行嵌入

        Args:
            num_embeddings: 嵌入数量
            embedding_dim: 嵌入维度
            params_dtype: 参数数据类型
            org_num_embeddings: 原始嵌入数量
            padding_size: 填充大小
            quant_config: 量化配置
            prefix: 前缀
        """
        # 调用父类初始化
        super().__init__(
            num_embeddings, embedding_dim, params_dtype, org_num_embeddings, padding_size, quant_config, prefix
        )
        # 如果没有量化配置，使用AscendUnquantizedEmbeddingMethod310
        if quant_config is None:
            self.quant_method = AscendUnquantizedEmbeddingMethod310()


class AscendParallelLMHead310(AscendParallelLMHead):
    """Ascend 310P平台的并行LM头

    继承自AscendParallelLMHead，专门为Ascend 310P平台优化，
    注册为Atlas 310P的自定义操作。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        """初始化并行LM头

        Args:
            num_embeddings: 嵌入数量
            embedding_dim: 嵌入维度
            bias: 是否使用偏置
            params_dtype: 参数数据类型
            org_num_embeddings: 原始嵌入数量
            padding_size: 填充大小
            quant_config: 量化配置
            prefix: 前缀
        """
        # 调用父类初始化
        super().__init__(
            num_embeddings, embedding_dim, bias, params_dtype, org_num_embeddings, padding_size, quant_config, prefix
        )

        # 如果没有量化配置，使用AscendUnquantizedEmbeddingMethod310
        if quant_config is None:
            self.quant_method = AscendUnquantizedEmbeddingMethod310()
