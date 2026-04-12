# 导入必要的库
import torch  # PyTorch核心库


def fused_gdn_gating_pytorch(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """融合GDN门控的PyTorch实现

    这是310P平台的回退实现，不依赖Triton支持

    Args:
        A_log: A参数的对数，形状为 [num_heads]
        a: a参数，形状为 [batch, num_heads]
        b: b参数，形状为 [batch, num_heads]
        dt_bias: dt偏置，形状为 [num_heads]
        beta: softplus函数的beta参数
        threshold: softplus函数的阈值参数

    Returns:
        g: 门控参数，形状为 [1, batch, num_heads]
        beta_output: sigmoid(b)，形状为 [1, batch, num_heads]
    """
    # 获取批次大小和头数
    batch, num_heads = a.shape
    # 删除num_heads变量，因为后面不需要使用
    del num_heads
    # 为了稳定性，将非线性门控计算保持在fp32精度
    compute_dtype = torch.float32
    # 将所有输入转换为fp32
    A_log_f = A_log.to(compute_dtype)
    a_f = a.to(compute_dtype)
    b_f = b.to(compute_dtype)
    dt_bias_f = dt_bias.to(compute_dtype)

    # 扩展A_log和dt_bias以匹配a的形状
    A_log_expanded = A_log_f.unsqueeze(0).expand(batch, -1)
    dt_bias_expanded = dt_bias_f.unsqueeze(0).expand(batch, -1)

    # 计算x = a + dt_bias
    x = a_f + dt_bias_expanded

    # 计算softplus(x)
    beta_x = beta * x
    softplus_x = torch.where(
        beta_x <= threshold,
        (1.0 / beta) * torch.log1p(torch.exp(beta_x)),
        x,
    )

    # 计算g = -exp(A_log) * softplus(x)
    g = -torch.exp(A_log_expanded) * softplus_x

    # 添加序列维度
    g = g.unsqueeze(0)

    # 匹配Triton内核：在fp32中计算sigmoid，然后转换为输入b的数据类型
    beta_output = torch.sigmoid(b_f).to(b.dtype)
    beta_output = beta_output.unsqueeze(0)

    return g, beta_output
