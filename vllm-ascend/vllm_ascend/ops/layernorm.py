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
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm, RMSNormGated

from vllm_ascend.ops.triton.layernorm_gated import layer_norm_fwd_npu
from vllm_ascend.utils import enable_custom_op, get_weight_prefetch_method


class AscendRMSNorm(RMSNorm):
    """Ascend平台的RMS归一化实现"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        """初始化AscendRMSNorm

        Args:
            hidden_size: 隐藏层大小
            eps: epsilon值，用于数值稳定性
            var_hidden_size: 方差隐藏层大小
            has_weight: 是否有权重
            dtype: 数据类型
        """
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        vllm_config = get_current_vllm_config()
        self.bias = None
        self.bias_loaded = False

        # 使用anti_method m4的量化会生成非零的norm bias
        if vllm_config.quant_config is not None and any(
            "norm.bias" in name for name in vllm_config.quant_config.quant_description
        ):
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
            self.bias.weight_loader = self._bias_weight_loader

    def _bias_weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """加载偏置权重

        Args:
            param: 参数
            loaded_weight: 加载的权重
        """
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # 有时标量值不被视为具有形状的张量
            # 所以如果param和loaded_weight都是标量，
            # 使用"广播"而不是复制
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
        self.bias_loaded = True

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """前向传播实现（out-of-tree）

        Args:
            x: 输入张量
            residual: 残差张量

        Returns:
            归一化后的张量，如果提供了residual，则返回元组(x, residual)
        """
        import torch_npu

        if residual is not None:
            # 可能对残差进行分块
            residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
            if enable_custom_op():
                # 使用自定义操作
                x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                    x, residual, self.weight, self.bias, self.variance_epsilon
                )
            else:
                # 使用torch_npu操作
                x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
                if self.bias is not None:
                    x.add_(self.bias)
            return x, residual

        # 执行RMS归一化
        x, residual = torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)
        if self.bias_loaded:
            x.add_(self.bias)

        # 可能预取MLP权重后处理
        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mlp_weight_postprocess(x)
        return x


class AscendGemmaRMSNorm(GemmaRMSNorm):
    """Ascend平台的Gemma RMS归一化实现"""

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """前向传播实现（out-of-tree）

        Args:
            x: 输入张量
            residual: 残差张量

        Returns:
            归一化后的张量，如果提供了residual，则返回元组(x, residual)
        """
        import torch_npu

        if residual is not None:
            # 可能对残差进行分块
            residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
            if enable_custom_op():
                # 使用自定义操作
                x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                    x, residual, 1.0 + self.weight, None, self.variance_epsilon
                )
            else:
                # 使用torch_npu操作
                x, _, residual = torch_npu.npu_add_rms_norm(x, residual, 1.0 + self.weight, self.variance_epsilon)
            return x, residual

        # 执行Gemma RMS归一化
        x, _ = torch.ops._C_ascend.npu_gemma_rms_norm(x, self.weight, self.variance_epsilon)
        return x


class LayerNormFn(torch.autograd.Function):
    """层归一化函数"""

    @staticmethod
    def forward(ctx, x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, is_rms_norm=False):
        """前向传播

        Args:
            ctx: 上下文
            x: 输入张量
            weight: 权重
            bias: 偏置
            z: 门控张量
            eps: epsilon值
            group_size: 组大小
            norm_before_gate: 是否在门控前进行归一化
            is_rms_norm: 是否是RMS归一化

        Returns:
            归一化后的张量
        """
        # 保存原始形状
        x_shape_og = x.shape
        # 将输入数据重塑为2D张量
        x = x.reshape(-1, x.shape[-1])
        # 确保内存连续
        if x.stride(-1) != 1:
            x = x.contiguous()
        if z is not None:
            # 确保z的形状与x相同
            assert z.shape == x_shape_og
            # 将z重塑为2D张量
            z = z.reshape(-1, z.shape[-1])
            # 确保内存连续
            if z.stride(-1) != 1:
                z = z.contiguous()
        # 确保权重内存连续
        weight = weight.contiguous()
        if bias is not None:
            # 确保偏置内存连续
            bias = bias.contiguous()
        # 执行层归一化前向计算
        y, mean, rstd = layer_norm_fwd_npu(
            x,
            weight,
            bias,
            eps,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
        # 保存用于反向传播的变量
        ctx.save_for_backward(x, weight, bias, mean, rstd, z)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.group_size = group_size
        ctx.norm_before_gate = norm_before_gate
        ctx.is_rms_norm = is_rms_norm
        # 恢复原始形状
        return y.reshape(x_shape_og)


class AscendRMSNormGated(RMSNormGated):
    """Ascend平台的带门控RMS归一化实现"""

    def __init__(
        self,
        hidden_size,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """初始化AscendRMSNormGated

        Args:
            hidden_size: 隐藏层大小
            eps: epsilon值，用于数值稳定性
            group_size: 组大小，如果为None，则等效于group_size=hidden_size（即只有1个组）
            norm_before_gate: 是否在门控前进行归一化
            device: 设备
            dtype: 数据类型
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(hidden_size, eps, group_size, norm_before_gate, device, dtype)
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        """重置参数"""
        torch.nn.init.ones_(self.weight)

    def forward_oot(self, x, z=None):
        """前向传播实现（out-of-tree）

        Args:
            x: 输入张量
            z: 门控张量

        Returns:
            归一化后的张量
        """
        return LayerNormFn.apply(x, self.weight, self.bias, z, self.eps, self.group_size, self.norm_before_gate, True)
