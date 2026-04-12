#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# mypy: ignore-errors

from __future__ import annotations

# 导入必要的库
import torch  # PyTorch核心库
import torch.nn.functional as F  # PyTorch函数式API

# 分块大小
CHUNK_SIZE = 64


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """计算L2归一化

    Args:
        x: 输入张量
        dim: 归一化维度
        eps: 防止除零的小值

    Returns:
        归一化后的张量
    """
    return x * torch.rsqrt(x.square().sum(dim=dim, keepdim=True) + eps)


def _expand_qk_to_v_heads(x: torch.Tensor, num_v_heads: int) -> torch.Tensor:
    """扩展q/k头以匹配v头，用于分组值注意力语义

    Args:
        x: 输入张量，形状为 [L, Hqk, D]
        num_v_heads: v头的数量

    Returns:
        扩展后的张量，形状为 [L, Hv, D]
    """
    h_qk = x.shape[1]
    if h_qk == num_v_heads:
        return x
    if num_v_heads % h_qk != 0:
        raise ValueError(f"Invalid grouped heads: Hqk={h_qk}, Hv={num_v_heads}.")
    group_size = num_v_heads // h_qk
    return x.repeat_interleave(group_size, dim=1)


def _iter_seq_ranges(batch_size: int, seq_len: int, cu_seqlens: torch.Tensor | None) -> list[tuple[int, int, int]]:
    """生成序列范围列表

    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        cu_seqlens: 累积序列长度张量

    Returns:
        序列范围列表，每个元素为 (batch_idx, start, end)
    """
    if cu_seqlens is None:
        return [(i, 0, seq_len) for i in range(batch_size)]
    return [(i, int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())) for i in range(len(cu_seqlens) - 1)]


def _normalize_chunk_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """归一化输入为 [B, T, H, D] / [B, T, H] 格式，同时保留TND支持

    Args:
        q: 查询张量
        k: 键张量
        v: 值张量
        g: 门控张量
        beta: 偏置张量
        cu_seqlens: 累积序列长度张量

    Returns:
        归一化后的张量和指示输入是否为TND格式的标志
    """
    input_was_tnd = False

    if q.ndim == 3:
        # 处理TND格式输入
        if cu_seqlens is None:
            raise ValueError("TND inputs require `cu_seqlens` for variable-length layout.")
        if k.ndim != 3 or v.ndim != 3:
            raise ValueError("When q is TND, k and v must also be TND.")
        if g.ndim != 2 or beta.ndim != 2:
            raise ValueError("When q is TND, g and beta must be shape [T, H].")
        # 添加批次维度
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)
        input_was_tnd = True
    elif q.ndim == 4:
        # 处理4D格式输入
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError("When q is 4D, k and v must also be 4D.")
        if g.ndim != 3 or beta.ndim != 3:
            raise ValueError("When q is 4D, g and beta must be shape [B, T, H].")
    else:
        raise ValueError(f"Unsupported q ndim={q.ndim}; expected 3D(TND) or 4D(BTHD).")

    return q, k, v, g, beta, input_was_tnd


def _torch_chunk_gated_delta_rule_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = CHUNK_SIZE,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """分块的PyTorch实现，与Qwen3-Next的PyTorch路径对齐

    路径：transformers/models/qwen3_next/modular_qwen3_next.py::torch_chunk_gated_delta_rule

    形状：
    query/key: [B, T, H, K]
    value:     [B, T, H, V]
    g/beta:    [B, T, H]
    initial_state: [B, H, K, V]

    Args:
        query: 查询张量
        key: 键张量
        value: 值张量
        g: 门控张量
        beta: 偏置张量
        chunk_size: 分块大小
        initial_state: 初始状态张量
        output_final_state: 是否输出最终状态
        use_qk_l2norm_in_kernel: 是否在核中使用qk的L2归一化

    Returns:
        注意力输出和最终状态（如果output_final_state为True）
    """
    # 保存初始数据类型
    initial_dtype = query.dtype
    # 如果需要，对query和key进行L2归一化
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

    # 转换维度顺序并转换为float32以提高计算精度
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    # 获取形状信息
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    # 计算填充大小，确保序列长度是chunk_size的倍数
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    # 对输入进行填充
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))

    # 计算总序列长度和缩放因子
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    # 计算v_beta和k_beta
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # 重塑为分块形式
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    # 创建对角线掩码
    mask_diag = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # 计算分块衰减
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    # 计算注意力权重
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_diag, 0)
    # 累积注意力权重
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    # 添加单位矩阵
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    # 计算value和k_cumdecay
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    # 初始化最后一个循环状态
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )
    # 创建输出张量
    core_attn_out = torch.zeros_like(value)

    # 创建上三角掩码
    mask_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # 处理每个分块
    for i in range(0, total_sequence_length // chunk_size):
        # 获取当前分块的query、key和value
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        # 计算块间注意力
        attn_inter_chunk = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask_upper, 0)
        # 计算v_prime
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        # 计算v_new
        v_new = v_i - v_prime
        # 计算中间状态
        inter_state = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        # 计算核心注意力输出
        core_attn_out[:, :, i] = inter_state + attn_inter_chunk @ v_new
        # 更新最后一个循环状态
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    # 如果不需要输出最终状态，设置为None
    if not output_final_state:
        last_recurrent_state = None

    # 重塑输出并裁剪到原始序列长度
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    # 转换回原始维度顺序和数据类型
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def chunk_gated_delta_rule_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """310P平台的分块门控Delta规则实现，具有vLLM兼容接口

    内部数学运算遵循Transformers的torch_chunk_gated_delta_rule流程

    Args:
        q: 查询张量
        k: 键张量
        v: 值张量
        g: 门控张量
        beta: 偏置张量
        initial_state: 初始状态张量
        output_final_state: 是否输出最终状态
        cu_seqlens: 累积序列长度张量
        head_first: 是否头维度在前（310P不支持）
        use_qk_l2norm_in_kernel: 是否在核中使用qk的L2归一化

    Returns:
        注意力输出和最终状态（如果output_final_state为True）
    """
    # 检查head_first参数
    if head_first:
        raise DeprecationWarning("head_first=True is not supported in 310P fallback.")
    # 归一化输入
    q, k, v, g, beta, input_was_tnd = _normalize_chunk_inputs(q, k, v, g, beta, cu_seqlens)

    # 检查可变长度模式的批次大小
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("Variable-length mode expects batch size B=1.")

    # 获取形状信息
    batch_size, total_tokens, h_qk, k_dim = q.shape
    h_v = v.shape[2]
    v_dim = v.shape[-1]
    # 检查形状匹配
    if k.shape != q.shape:
        raise ValueError("q and k shapes must match.")
    if g.shape != beta.shape or g.shape[:2] != (batch_size, total_tokens) or g.shape[2] != h_v:
        raise ValueError("g/beta must have shape [B, T, Hv] matching v.")

    # 生成序列范围
    seq_ranges = _iter_seq_ranges(batch_size, total_tokens, cu_seqlens)
    # 计算状态数量
    num_states = batch_size if cu_seqlens is None else len(cu_seqlens) - 1
    # 初始化状态
    if initial_state is not None:
        states = initial_state.to(torch.float32).clone()
    else:
        states = torch.zeros(num_states, h_v, k_dim, v_dim, dtype=torch.float32, device=q.device)

    # 创建输出张量
    out = torch.zeros_like(v)
    # 处理每个序列
    for seq_idx, start, end in seq_ranges:
        seq_len = end - start
        if seq_len <= 0:
            continue

        # 计算批次索引
        b_idx = 0 if (cu_seqlens is not None and batch_size == 1) else seq_idx

        # 准备当前序列的输入
        q_seq = _expand_qk_to_v_heads(q[b_idx, start:end], h_v).unsqueeze(0)
        k_seq = _expand_qk_to_v_heads(k[b_idx, start:end], h_v).unsqueeze(0)
        v_seq = v[b_idx, start:end].unsqueeze(0)
        g_seq = g[b_idx, start:end].unsqueeze(0)
        beta_seq = beta[b_idx, start:end].unsqueeze(0)
        init_seq_state = states[seq_idx].unsqueeze(0)

        # 调用分块实现
        out_seq, final_state = _torch_chunk_gated_delta_rule_chunked(
            query=q_seq,
            key=k_seq,
            value=v_seq,
            g=g_seq,
            beta=beta_seq,
            chunk_size=CHUNK_SIZE,
            initial_state=init_seq_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        # 更新输出和状态
        out[b_idx, start:end] = out_seq[0]
        states[seq_idx] = final_state[0]

    # 如果输入是TND格式，移除批次维度
    if input_was_tnd:
        out = out[0]

    # 根据需要返回最终状态
    if output_final_state:
        return out, states
    return out, None
