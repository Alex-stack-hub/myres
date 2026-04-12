# 导入必要的库
import torch  # PyTorch核心库


def _maybe_l2norm(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    """条件性地应用L2归一化

    Args:
        x: 输入张量
        enabled: 是否启用L2归一化

    Returns:
        归一化后的张量（如果启用）或原始张量（如果未启用）
    """
    if not enabled:
        return x
    return x / (torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True)) + 1e-6)


def _expand_to_hv(x: torch.Tensor, hv: int) -> torch.Tensor:
    """扩展 [H, ...] 到 [HV, ...]，用于分组值注意力语义

    Args:
        x: 输入张量，形状为 [H, ...]
        hv: 目标头维度大小

    Returns:
        扩展后的张量，形状为 [HV, ...]
    """
    h = x.shape[0]
    if h == hv:
        return x
    if hv % h != 0:
        raise ValueError(f"Cannot expand head dim from {h} to {hv}.")
    return x.repeat_interleave(hv // h, dim=0)


def _infer_num_states(
    default_n: int,
    initial_state: torch.Tensor | None,
    ssm_state_indices: torch.Tensor | None,
) -> int:
    """推断状态数量

    Args:
        default_n: 默认状态数量
        initial_state: 初始状态张量
        ssm_state_indices: SSM状态索引张量

    Returns:
        推断出的状态数量
    """
    if initial_state is not None:
        return initial_state.shape[0]
    if ssm_state_indices is None:
        return default_n
    nonneg = ssm_state_indices[ssm_state_indices >= 0]
    if nonneg.numel() == 0:
        return default_n
    return int(nonneg.max().item()) + 1


def _state_index(
    seq_idx: int,
    tok_idx: int,
    ssm_state_indices: torch.Tensor | None,
) -> int:
    """获取状态索引

    Args:
        seq_idx: 序列索引
        tok_idx:  token索引
        ssm_state_indices: SSM状态索引张量

    Returns:
        状态索引
    """
    if ssm_state_indices is None:
        return seq_idx
    if ssm_state_indices.ndim == 1:
        return int(ssm_state_indices[seq_idx].item())
    return int(ssm_state_indices[seq_idx, tok_idx].item())


def _run_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor | None,
    states: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    ssm_state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    use_initial_state: bool,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GDN delta规则的参考PyTorch循环实现

    形状遵循fla.ops约定：
    q,k: [B, T, H, K]
    v:   [B, T, HV, V]
    g,beta: [B, T, HV] (beta也可以是 [B, T, HV, V])
    states: [N_state, HV, K, V]

    Args:
        q: 查询张量
        k: 键张量
        v: 值张量
        g: 门控张量
        beta: 偏置张量
        states: 状态张量
        scale: 缩放因子
        cu_seqlens: 累积序列长度张量
        ssm_state_indices: SSM状态索引张量
        num_accepted_tokens: 已接受token数量张量
        use_initial_state: 是否使用初始状态
        use_qk_l2norm_in_kernel: 是否在核中使用qk的L2归一化

    Returns:
        输出张量和更新后的状态张量
    """
    B, T, _, Kdim = k.shape
    HV = v.shape[2]
    Vdim = v.shape[-1]

    # 检查可变长度模式的批次大小
    if cu_seqlens is not None and B != 1:
        raise ValueError("Variable-length mode expects batch size B=1.")

    # 创建输出张量
    out = torch.zeros_like(v)

    # 生成序列范围
    if cu_seqlens is None:
        seq_ranges = [(i, 0, T) for i in range(B)]
    else:
        n_seq = len(cu_seqlens) - 1
        seq_ranges = [
            (
                i,
                int(cu_seqlens[i].item()),
                int(cu_seqlens[i + 1].item()),
            )
            for i in range(n_seq)
        ]

    # 处理每个序列
    for seq_idx, start, end in seq_ranges:
        seq_len = end - start
        if seq_len <= 0:
            continue

        # 处理已接受的token数量
        accepted = None
        if num_accepted_tokens is not None:
            accepted = int(num_accepted_tokens[seq_idx].item())
            seq_len = min(seq_len, accepted)
        if seq_len <= 0:
            continue

        # 初始化隐藏状态
        if use_initial_state:
            if ssm_state_indices is None:
                init_state_idx = seq_idx
            else:
                init_tok = (accepted - 1) if accepted is not None else 0
                init_state_idx = _state_index(seq_idx, init_tok, ssm_state_indices)
            if init_state_idx < 0:
                # 匹配Triton行为，处理连续批处理中的无效PAD_SLOT_ID
                continue
            if init_state_idx >= states.shape[0]:
                raise IndexError(f"state_idx {init_state_idx} out of range for states size {states.shape[0]}")
            h_t = states[init_state_idx].transpose(-1, -2).to(torch.float32)
        else:
            h_t = torch.zeros(HV, Vdim, Kdim, dtype=torch.float32, device=q.device)

        # 处理每个token
        for rel_t in range(seq_len):
            tok = start + rel_t

            # 获取当前token的输入
            if cu_seqlens is None:
                q_t = q[seq_idx, tok]
                k_t = k[seq_idx, tok]
                v_t = v[seq_idx, tok]
                g_t = g[seq_idx, tok] if g is not None else None
                beta_t = beta[seq_idx, tok] if beta is not None else None
            else:
                q_t = q[0, tok]
                k_t = k[0, tok]
                v_t = v[0, tok]
                g_t = g[0, tok] if g is not None else None
                beta_t = beta[0, tok] if beta is not None else None

            # 匹配Triton内核数学：先加载到fp32，然后应用l2norm
            q_t = q_t.to(torch.float32)
            k_t = k_t.to(torch.float32)
            q_t = _maybe_l2norm(q_t, use_qk_l2norm_in_kernel)
            k_t = _maybe_l2norm(k_t, use_qk_l2norm_in_kernel)
            v_t = v_t.to(torch.float32)
            q_t = q_t * scale

            # 扩展q和k到HV维度
            q_hv = _expand_to_hv(q_t, HV)
            k_hv = _expand_to_hv(k_t, HV)

            # 应用门控
            if g_t is not None:
                g_t = g_t.to(torch.float32)
                if g_t.ndim == 0:
                    g_t = g_t.expand(HV)
                elif g_t.shape[0] != HV:
                    g_t = _expand_to_hv(g_t.unsqueeze(-1), HV).squeeze(-1)
                h_t = h_t * torch.exp(g_t).view(HV, 1, 1)

            # 计算v_t
            v_t = v_t - torch.sum(h_t * k_hv.unsqueeze(-2), dim=-1)

            # 应用beta
            if beta_t is not None:
                beta_t = beta_t.to(torch.float32)
                if beta_t.ndim == 1:
                    if beta_t.shape[0] != HV:
                        beta_t = _expand_to_hv(beta_t.unsqueeze(-1), HV).squeeze(-1)
                    v_t = v_t * beta_t.view(HV, 1)
                else:
                    if beta_t.shape[0] != HV:
                        beta_t = _expand_to_hv(beta_t, HV)
                    v_t = v_t * beta_t

            # 更新隐藏状态和计算输出
            h_t = h_t + v_t.unsqueeze(-1) * k_hv.unsqueeze(-2)
            o_t = torch.sum(h_t * q_hv.unsqueeze(-2), dim=-1)

            # 更新输出
            if cu_seqlens is None:
                out[seq_idx, tok] = o_t.to(out.dtype)
            else:
                out[0, tok] = o_t.to(out.dtype)

            # 更新状态
            state_idx = _state_index(seq_idx, rel_t, ssm_state_indices)
            if state_idx >= 0:
                if state_idx >= states.shape[0]:
                    raise IndexError(f"state_idx {state_idx} out of range for states size {states.shape[0]}")
                states[state_idx] = h_t.transpose(-1, -2).to(states.dtype)

    return out, states


def fused_recurrent_gated_delta_rule_pytorch(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    inplace_final_state=False,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    use_qk_l2norm_in_kernel=False,
):
    """融合循环门控Delta规则的PyTorch回退实现

    Args:
        q: 查询张量
        k: 键张量
        v: 值张量
        g: 门控张量
        beta: 偏置张量
        initial_state: 初始状态张量
        inplace_final_state: 是否原地更新最终状态
        cu_seqlens: 累积序列长度张量
        ssm_state_indices: SSM状态索引张量
        num_accepted_tokens: 已接受token数量张量
        use_qk_l2norm_in_kernel: 是否在核中使用qk的L2归一化

    Returns:
        输出张量和更新后的状态张量
    """
    B, _, _, Kdim = k.shape
    HV = v.shape[2]
    Vdim = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    # 推断状态数量
    n_states = _infer_num_states(N, initial_state, ssm_state_indices)
    # 初始化状态
    if initial_state is not None:
        states = initial_state if inplace_final_state else initial_state.clone()
    else:
        states = torch.zeros(n_states, HV, Kdim, Vdim, dtype=q.dtype, device=q.device)

    # 计算缩放因子
    scale = Kdim**-0.5
    # 运行循环门控Delta规则
    out, states = _run_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        states=states,
        scale=scale,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_initial_state=initial_state is not None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    return out, states
