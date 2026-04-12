# 导入FLA（Flash Attention）相关的函数
from .chunk_gated_delta_rule import chunk_gated_delta_rule_pytorch  # 分块门控Delta规则的PyTorch实现
from .fused_gdn_gating import fused_gdn_gating_pytorch  # 融合GDN门控的PyTorch实现
from .fused_recurrent_gated_delta_rule import (
    fused_recurrent_gated_delta_rule_pytorch,  # 融合循环门控Delta规则的PyTorch实现
)

# 定义模块的公共API，控制从模块导入时的可见性
__all__ = [
    "fused_gdn_gating_pytorch",  # 融合GDN门控函数
    "fused_recurrent_gated_delta_rule_pytorch",  # 融合循环门控Delta规则函数
    "chunk_gated_delta_rule_pytorch",  # 分块门控Delta规则函数
]
