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
"""
要自定义此文件中类的线性通信组或前向传播，
请在 linear_op.py 中扩展新的线性操作。
此文件中的类不应被修改，包括 AscendQKVParallelLinear、
AscendMergedColumnParallelLinear、AscendMergedColumnParallelLinear、
AscendRowParallelLinear 和 AscendColumnParallelLinear。
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from vllm.config import get_current_vllm_config
from vllm.distributed import divide
from vllm.model_executor.layers.linear import (  # noqa
    WEIGHT_LOADER_V2_SUPPORTED,
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    QuantizeMethodBase,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.ops.linear_op import get_parallel_op, get_replicated_op
from vllm_ascend.utils import enable_sp, maybe_trans_nz


class AscendUnquantizedLinearMethod(UnquantizedLinearMethod):
    """无量化的线性方法"""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """加载后处理权重

        Args:
            layer: 线性层模块
        """
        super().process_weights_after_loading(layer)
        if "conv1d" not in layer.prefix:
            layer.weight.data = maybe_trans_nz(layer.weight.data)


# TODO(realliujiaxu): Remove this class after linear of vllm supports custom comm group
class AscendLinearBase(LinearBase):
    """Ascend平台线性层的基类"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        """初始化AscendLinearBase

        Args:
            input_size: 输入大小
            output_size: 输出大小
            skip_bias_add: 是否跳过偏置加法
            params_dtype: 参数数据类型
            quant_config: 量化配置
            prefix: 前缀
            return_bias: 是否返回偏置
            disable_tp: 是否禁用张量并行
        """
        nn.Module.__init__(self)

        # 保留输入参数
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        if quant_config is None:
            self.quant_method: QuantizeMethodBase | None = AscendUnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        self.return_bias = return_bias
        self.disable_tp = disable_tp


class AscendQKVParallelLinear(QKVParallelLinear):
    """注意力机制QKV变换的线性层

    用于注意力层中查询、键和值向量的线性变换的线性层。
    权重矩阵沿输出维度连接。该层沿头部维度并行化。
    当键/值头部的数量小于查询头部的数量时（例如，多查询/分组查询注意力），
    键/值头部可能被复制，而查询头部被分区。
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ):
        """初始化AscendQKVParallelLinear

        Args:
            hidden_size: 隐藏层大小
            head_size: 头部大小
            total_num_heads: 总头部数量
            total_num_kv_heads: 总键值头部数量
            bias: 是否使用偏置
            skip_bias_add: 是否跳过偏置加法
            params_dtype: 参数数据类型
            quant_config: 量化配置
            prefix: 前缀
            return_bias: 是否返回偏置
            disable_tp: 是否禁用张量并行
            v_head_size: 值头部大小
        """
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.custom_op, _, tp_size = get_parallel_op(disable_tp, prefix, self, "column")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after
        # linear of vllm supports custom comm group
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # 沿最后一个维度划分权重矩阵
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]
        AscendColumnParallelLinear.__init__(
            self,
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        """前向传播

        Args:
            input_: 输入张量

        Returns:
            输出张量，如果skip_bias_add为True，则返回元组(output, bias)
        """
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)


class AscendMergedColumnParallelLinear(MergedColumnParallelLinear):
    """带列并行的打包线性层

    类似于ColumnParallelLinear，但权重矩阵沿输出维度连接。
    加载权重矩阵时，不同的分区会被单独分片。

    在MLP模块中使用MLP张量并行组，
    在其他模块中使用原始的TP组。
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        """初始化AscendMergedColumnParallelLinear

        Args:
            input_size: 输入大小
            output_sizes: 输出大小列表
            bias: 是否使用偏置
            gather_output: 是否收集输出
            skip_bias_add: 是否跳过偏置加法
            params_dtype: 参数数据类型
            quant_config: 量化配置
            prefix: 前缀
            return_bias: 是否返回偏置
            disable_tp: 是否禁用张量并行
        """
        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(disable_tp, prefix, self, "column")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after
        # linear of vllm supports custom comm group
        self.output_sizes = output_sizes
        assert all(output_size % self.tp_size == 0 for output_size in output_sizes)
        AscendColumnParallelLinear.__init__(
            self,
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        """前向传播

        Args:
            input_: 输入张量

        Returns:
            输出张量，如果skip_bias_add为True，则返回元组(output, bias)
        """
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)


class AscendRowParallelLinear(RowParallelLinear):
    """带行并行的线性层
    在MLP模块中使用MLP张量并行组，
    在其他模块中使用原始的TP组。
    """

    # NOTE: SP场景中使用的全局唯一前缀标识符
    unique_prefix_idx = 0

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        """初始化AscendRowParallelLinear

        Args:
            input_size: 输入大小
            output_size: 输出大小
            bias: 是否使用偏置
            input_is_parallel: 输入是否并行
            skip_bias_add: 是否跳过偏置加法
            params_dtype: 参数数据类型
            reduce_results: 是否减少结果
            quant_config: 量化配置
            prefix: 前缀
            return_bias: 是否返回偏置
            disable_tp: 是否禁用张量并行
        """
        # TODO(kunpengW-code): Specifying the prefix in linear layers of some models in the vLLM.
        if enable_sp():
            compilation_config = get_current_vllm_config().compilation_config
            unique_prefix = prefix
            if prefix in compilation_config.static_forward_context:
                unique_prefix = f"{prefix}.unique_prefix{AscendRowParallelLinear.unique_prefix_idx}"
                AscendRowParallelLinear.unique_prefix_idx += 1
            self.unique_prefix = unique_prefix
            compilation_config.static_forward_context[unique_prefix] = self

        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(disable_tp, prefix, self, "row")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after
        # linear of vllm supports custom comm group
        # 沿第一个维度划分权重矩阵
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        AscendLinearBase.__init__(
            self,
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the results can lead to incorrect results")

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

        if self.custom_op is not None:
            self.custom_op.update_attrs()

    def forward(
        self,
        input_,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        """前向传播

        Args:
            input_: 输入张量
            **kwargs: 关键字参数

        Returns:
            输出张量，如果skip_bias_add为True，则返回元组(output, bias)
        """
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)


class AscendColumnParallelLinear(ColumnParallelLinear):
    """带列并行的线性层

    在MLP模块中使用MLP张量并行组，
    在其他模块中使用原始的TP组。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        output_sizes: list[int] | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        """初始化AscendColumnParallelLinear

        Args:
            input_size: 输入大小
            output_size: 输出大小
            bias: 是否使用偏置
            gather_output: 是否收集输出
            skip_bias_add: 是否跳过偏置加法
            params_dtype: 参数数据类型
            quant_config: 量化配置
            output_sizes: 输出大小列表
            prefix: 前缀
            return_bias: 是否返回偏置
            disable_tp: 是否禁用张量并行
        """
        #
        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(disable_tp, prefix, self, "column")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after
        # linear of vllm supports custom comm group
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # 如果是QKV或MergedColumn，使用每个分区的输出大小
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [divide(output_size, self.tp_size) for output_size in self.output_sizes]

        AscendLinearBase.__init__(
            self,
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.gather_output = gather_output

        if output_sizes is None:
            output_sizes = [output_size]

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if bias:
            self.bias = Parameter(torch.empty(self.output_size_per_partition, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

        if self.custom_op is not None:
            self.custom_op.update_attrs()

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        """前向传播

        Args:
            input_: 输入张量

        Returns:
            输出张量，如果skip_bias_add为True，则返回元组(output, bias)
        """
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)


class AscendReplicatedLinear(ReplicatedLinear):
    """Ascend复制线性层

    参数:
        input_size: 线性层的输入维度。
        output_size: 线性层的输出维度。
        bias: 如果为true，添加偏置。
        skip_bias_add: 如果为true，跳过添加偏置，而是返回它。
        params_dtype: 参数的数据类型。
        quant_config: 量化配置。
        prefix: 状态字典中层的名称，包括所有父级
                        (例如 model.layers.0.qkv_proj)
        return_bias: 如果为true，在前向传播中返回偏置和输出。
        disable_tp: 对复制线性层无效。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        """初始化AscendReplicatedLinear

        Args:
            input_size: 输入大小
            output_size: 输出大小
            bias: 是否使用偏置
            skip_bias_add: 是否跳过偏置加法
            params_dtype: 参数数据类型
            quant_config: 量化配置
            prefix: 前缀
            return_bias: 是否返回偏置
            disable_tp: 是否禁用张量并行
        """
        self.custom_op, self.tp_rank, self.tp_size = get_replicated_op(disable_tp, prefix, self)
        # 如果是MergedReplicatedLinear，使用每个分区的输出大小
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = self.output_sizes
        else:
            self.output_partition_sizes = [output_size]

        AscendLinearBase.__init__(
            self,
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        # 所有线性层都支持量化方法
        assert self.quant_method is not None
        self.quant_method.create_weights(
            self,
            self.input_size,
            [self.output_size],
            self.input_size,
            self.output_size,
            self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=self.params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

        if self.custom_op is not None:
            self.custom_op.update_attrs()

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        """前向传播

        Args:
            input_: 输入张量

        Returns:
            输出张量，如果skip_bias_add为True，则返回元组(output, bias)
        """
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)
