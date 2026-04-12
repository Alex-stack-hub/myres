# 导入必要的库
import torch  # PyTorch核心库
import torch_npu  # NPU相关操作库
from vllm.model_executor.layers.layernorm import RMSNormGated  # 上游的门控RMS归一化

from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm, AscendRMSNorm  # 基础Ascend归一化类


class AscendRMSNorm310(AscendRMSNorm):
    """Ascend 310P平台的RMS归一化实现

    继承自AscendRMSNorm，专门为Ascend 310P平台优化，
    实现了forward_oot方法，支持残差连接。
    """

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """前向传播，支持残差连接

        Args:
            x: 输入张量
            residual: 残差张量，如果为None则不使用残差连接

        Returns:
            如果提供了residual，返回元组 (输出张量, 更新后的残差张量)
            否则，返回输出张量
        """
        if residual is not None:
            # 使用NPU原生的add_rms_norm操作，同时计算残差和归一化
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            # 如果有偏置，添加偏置
            if self.bias is not None:
                x.add_(self.bias)
            # 返回输出和更新后的残差
            return x, residual

        # 没有残差连接时，直接使用NPU原生的rms_norm操作
        x, _ = torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)
        # 如果有偏置，添加偏置
        if self.bias is not None:
            x.add_(self.bias)
        # 返回输出
        return x


class AscendGemmaRMSNorm310(AscendGemmaRMSNorm):
    """Ascend 310P平台的Gemma RMS归一化实现

    继承自AscendGemmaRMSNorm，专门为Ascend 310P平台优化，
    实现了forward_oot方法，支持残差连接。
    """

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """前向传播，支持残差连接

        Args:
            x: 输入张量
            residual: 残差张量，如果为None则不使用残差连接

        Returns:
            如果提供了residual，返回元组 (输出张量, 更新后的残差张量)
            否则，返回输出张量
        """
        if residual is not None:
            # 保存残差的原始数据类型
            orig_dtype = residual.dtype
            # 添加残差，确保数据类型一致
            x = x + residual.to(x.dtype)
            # 更新残差为新的值，并恢复原始数据类型
            residual = x.to(orig_dtype)
            # 使用NPU原生的rms_norm操作，权重使用1.0 + self.weight
            x, _ = torch_npu.npu_rms_norm(x, 1.0 + self.weight, self.variance_epsilon)
            # 返回输出和更新后的残差
            return x, residual

        # 没有残差连接时，直接使用NPU原生的rms_norm操作
        x, _ = torch_npu.npu_rms_norm(x, 1.0 + self.weight, self.variance_epsilon)
        # 返回输出
        return x


class AscendRMSNormGated310(RMSNormGated):
    """Ascend 310P平台的门控RMS归一化实现

    继承自RMSNormGated，专门为Ascend 310P平台优化，
    实现了forward_oot方法，使用上游的原生实现。
    """

    def forward_oot(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量
            z: 门控张量，如果为None则不使用门控

        Returns:
            输出张量
        """
        # 310P平台不依赖于Triton门控层归一化路径
        # 直接重用上游的原生实现
        return super().forward_native(x, z)
