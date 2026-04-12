# 导入必要的库
import torch  # PyTorch核心库
import torch.nn.functional as F  # PyTorch函数式API
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # 填充槽位ID


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    """因果卷积的PyTorch参考实现

    因果卷积是一种只考虑当前和过去输入的卷积操作，
    常用于序列建模任务中，确保模型不会看到未来的信息。

    Args:
        x: 输入张量，形状为 (batch, dim, seqlen)
        weight: 权重张量，形状为 (dim, width)
        bias: 偏置张量，形状为 (dim,)
        initial_states: 初始状态张量，形状为 (batch, dim, width - 1)，用于处理序列的起始部分
        final_states_out: 最终状态输出张量，形状为 (batch, dim, width - 1)，用于存储计算后的最终状态
        return_final_states: 是否返回最终状态
        activation: 激活函数名称，支持 None、"silu" 或 "swish"

    Returns:
        out: 输出张量，形状为 (batch, dim, seqlen)
        final_states_out: 如果 return_final_states 为 True，则返回最终状态张量，形状为 (batch, dim, width - 1)
    """
    # 检查激活函数是否支持
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    # 保存输入数据类型，用于后续恢复
    dtype_in = x.dtype
    # 将输入转换为权重的数据类型，确保类型一致
    x = x.to(weight.dtype)
    # 获取序列长度
    seqlen = x.shape[-1]
    # 获取权重的维度和宽度
    dim, width = weight.shape

    # 执行因果卷积
    if initial_states is None:
        # 没有初始状态时，使用填充进行卷积
        # padding=width - 1 确保因果性，只考虑当前和过去的输入
        # groups=dim 表示使用深度可分离卷积
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        # 有初始状态时，将初始状态与输入拼接后进行卷积
        # 这样可以避免填充，直接使用初始状态作为历史信息
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    # 截取与输入序列长度相同的输出
    out = out[..., :seqlen]

    # 处理最终状态
    if return_final_states:
        # 计算最终状态，确保长度为 width - 1
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
        if final_states_out is not None:
            # 如果提供了最终状态输出张量，将结果复制到其中
            final_states_out.copy_(final_states)
        else:
            # 否则，创建新的最终状态输出张量
            final_states_out = final_states
    # 应用激活函数并恢复数据类型
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    # 根据是否返回最终状态返回不同的结果
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    conv_states: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    """310P平台的因果卷积函数实现

    专门为Ascend 310P平台优化的因果卷积实现，
    支持变长序列输入和状态缓存。

    Args:
        x: 输入张量，形状为 (dim, cu_seq_len) 用于变长序列
        weight: 权重张量，形状为 (dim, width)
        bias: 偏置张量，形状为 (dim,)
        activation: 激活函数名称，支持 None、"silu" 或 "swish"
        conv_states: 卷积状态张量，形状为 (..., dim, width - 1)
        has_initial_state: 指示是否有初始状态的布尔张量，形状为 (batch,)
        cache_indices: 缓存索引张量，形状为 (batch,)
        query_start_loc: 查询起始位置张量，形状为 (batch + 1,)
        pad_slot_id: 填充槽位ID

    Returns:
        out: 输出张量，形状为 (batch, dim, seqlen)
    """

    # 检查激活函数是否支持
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    # 检查必需的参数
    if query_start_loc is None:
        raise RuntimeError("causal_conv1d_fn requires query_start_loc for varlen inputs.")
    if cache_indices is None:
        raise RuntimeError("causal_conv1d_fn requires cache_indices.")
    if has_initial_state is None:
        raise RuntimeError("causal_conv1d_fn requires has_initial_state.")
    if conv_states is None:
        raise RuntimeError("causal_conv1d_fn requires conv_states.")

    # 确保输入张量内存连续，提高计算效率
    if x.stride(-1) != 1:
        x = x.contiguous()
    # 确保偏置张量内存连续
    bias = bias.contiguous() if bias is not None else None

    # 标准化输入形状为 [dim, total_tokens]
    if x.dim() == 3:
        if x.shape[0] == 1:
            x = x.squeeze(0)
        elif x.shape[1] == 1:
            x = x.squeeze(1).transpose(0, 1)
        else:
            raise RuntimeError(f"Unsupported x shape for causal_conv1d_fn: {tuple(x.shape)}")
    if x.dim() != 2:
        raise RuntimeError(f"Unsupported x ndim for causal_conv1d_fn: {x.dim()}")

    # 处理权重维度
    feature_dim = x.shape[0]
    if weight.shape[0] != feature_dim and weight.shape[1] == feature_dim:
        weight = weight.transpose(0, 1)
    weight = weight.contiguous()
    dim, width = weight.shape
    if dim != feature_dim:
        raise RuntimeError(
            f"causal_conv1d_fn: weight dim mismatch, x dim={feature_dim}, weight.shape={tuple(weight.shape)}"
        )

    # 处理卷积状态
    state_len = width - 1
    if conv_states.shape[-2] != dim and conv_states.shape[-1] == dim:
        conv_states = conv_states.transpose(-1, -2)
    if conv_states.shape[-2] != dim:
        raise RuntimeError(
            f"causal_conv1d_fn: conv_states dim mismatch, "
            f"expected dim={dim}, conv_states.shape={tuple(conv_states.shape)}"
        )
    if conv_states.shape[-1] < state_len:
        raise RuntimeError(f"causal_conv1d_fn: conv_states too short, need >= {state_len}, got {conv_states.shape[-1]}")

    # 计算每个序列的长度并分割输入
    seqlens = (query_start_loc[1:] - query_start_loc[:-1]).tolist()
    splits = torch.split(x, seqlens, dim=-1)

    # 处理每个序列
    out_chunks = []
    for i, x_s in enumerate(splits):
        # 获取缓存索引
        cache_idx = int(cache_indices[i].item())
        # 跳过填充槽位
        if cache_idx == pad_slot_id:
            continue

        # 获取当前序列的状态
        state = conv_states[cache_idx]
        # 准备初始状态
        init_state = state[..., :state_len].unsqueeze(0) if bool(has_initial_state[i].item()) else None
        # 调用参考实现计算输出和最终状态
        out_ref, final_state = causal_conv1d_ref(
            x_s.unsqueeze(0),
            weight,
            bias,
            activation=activation,
            return_final_states=True,
            initial_states=init_state,
        )
        # 更新状态
        state[..., :state_len].copy_(final_state.squeeze(0))
        # 添加输出到结果列表
        out_chunks.append(out_ref.squeeze(0))

    # 处理空输出情况
    if not out_chunks:
        return x.new_zeros((dim, 0))
    # 拼接所有输出
    return torch.cat(out_chunks, dim=-1)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    """310P平台的因果卷积更新实现

    用于更新因果卷积的状态和计算输出，
    支持多种输入形状和变长序列。

    Args:
        x: 输入张量，可以是不同形状
        conv_state: 卷积状态张量，形状为 (..., dim, state_len)
        weight: 权重张量，形状为 (dim, width)
        bias: 偏置张量，形状为 (dim,)
        activation: 激活函数名称，支持 None、"silu"、"swish" 或布尔值
        conv_state_indices: 卷积状态索引张量，形状为 (batch,)
        num_accepted_tokens: 接受的token数量张量，形状为 (batch,)
        query_start_loc: 查询起始位置张量，形状为 (batch + 1,)
        pad_slot_id: 填充槽位ID

    Returns:
        out: 与输入形状相同的输出张量
    """
    # 处理激活函数参数
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    # 保存原始输入数据类型，用于后续恢复
    original_x_dtype = x.dtype
    # 将输入转换为卷积状态的数据类型，确保类型一致
    x = x.to(conv_state.dtype)

    # 处理权重维度
    feature_dim = x.shape[-1] if query_start_loc is None else x.shape[1]
    if weight.shape[0] != feature_dim and weight.shape[1] == feature_dim:
        weight = weight.transpose(0, 1)
    weight = weight.contiguous()
    dim, width = weight.shape
    if dim != feature_dim:
        raise RuntimeError(
            f"causal_conv1d_update: weight dim mismatch, feature_dim={feature_dim}, weight.shape={tuple(weight.shape)}"
        )

    # 处理卷积状态维度
    if conv_state.shape[-2] != dim and conv_state.shape[-1] == dim:
        # 接受 (..., dim, state_len) 和 (..., state_len, dim) 两种输入形状
        conv_state = conv_state.transpose(-1, -2)
    if conv_state.shape[-2] != dim:
        raise RuntimeError(
            f"causal_conv1d_update: conv_state dim mismatch, "
            f"expected dim={dim}, conv_state.shape={tuple(conv_state.shape)}"
        )

    # 检查卷积状态长度
    state_len = width - 1
    if conv_state.shape[-1] < state_len:
        raise RuntimeError(
            f"causal_conv1d_update: conv_state too short, need >= {state_len}, got {conv_state.shape[-1]}"
        )

    # 创建输出张量，初始化为输入的副本
    out = x.clone()

    # 选择状态的辅助函数
    def _select_state(i: int) -> torch.Tensor | None:
        """根据索引选择对应的卷积状态

        Args:
            i: 序列索引

        Returns:
            对应的卷积状态张量，如果是填充槽位则返回 None
        """
        if conv_state_indices is not None:
            idx = int(conv_state_indices[i].item())
            if idx == pad_slot_id:
                return None
            state = conv_state[idx]
        else:
            state = conv_state[i]
        return state

    # 处理单个序列的辅助函数
    def _run_one(seq_tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """处理单个序列的因果卷积

        Args:
            seq_tokens: 序列tokens，形状为 [L, dim]
            state: 卷积状态，形状为 [dim, state_len]

        Returns:
            处理后的序列，形状为 [L, dim]
        """
        # 转换序列形状：[L, dim] -> [1, dim, L]
        x_ref = seq_tokens.transpose(0, 1).unsqueeze(0)
        # 准备初始状态
        init_state = state[..., :state_len].unsqueeze(0)
        # 调用参考实现计算输出和最终状态
        out_ref, final_state = causal_conv1d_ref(
            x_ref,
            weight,
            bias,
            initial_states=init_state,
            return_final_states=True,
            activation=activation,
        )
        # 更新状态
        state[..., :state_len].copy_(final_state.squeeze(0))
        # 转换输出形状：[1, dim, L] -> [L, dim]
        return out_ref.squeeze(0).transpose(0, 1)

    # 处理不同形状的输入
    if query_start_loc is None:
        if x.dim() == 2:
            # 处理形状为 [batch, dim] 的输入
            batch = x.shape[0]
            for i in range(batch):
                state = _select_state(i)
                if state is None:
                    continue
                seq_tokens = x[i : i + 1]
                # 处理接受的token数量
                if num_accepted_tokens is not None:
                    accepted = int(num_accepted_tokens[i].item())
                    if accepted <= 0:
                        continue
                    seq_tokens = seq_tokens[:accepted]
                # 处理单个序列
                out_i = _run_one(seq_tokens, state)
                # 更新输出
                out[i : i + out_i.shape[0]] = out_i
        else:
            # 处理形状为 [batch, seq_len, dim] 的输入
            batch = x.shape[0]
            for i in range(batch):
                state = _select_state(i)
                if state is None:
                    continue
                seq_tokens = x[i]
                # 处理接受的token数量
                if num_accepted_tokens is not None:
                    accepted = int(num_accepted_tokens[i].item())
                    if accepted <= 0:
                        continue
                    seq_tokens = seq_tokens[:accepted]
                # 处理单个序列
                out_i = _run_one(seq_tokens, state)
                # 更新输出
                out[i, : out_i.shape[0]] = out_i
    else:
        # 处理变长序列输入
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        for i in range(batch):
            # 获取序列的起始和结束位置
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end <= start:
                continue
            # 获取状态
            state = _select_state(i)
            if state is None:
                continue
            # 获取序列tokens
            seq_tokens = x[start:end]
            # 处理接受的token数量
            if num_accepted_tokens is not None:
                accepted = int(num_accepted_tokens[i].item())
                if accepted <= 0:
                    continue
                seq_tokens = seq_tokens[:accepted]
            # 处理单个序列
            out_i = _run_one(seq_tokens, state)
            # 更新输出
            out[start : start + out_i.shape[0]] = out_i

    # 恢复原始数据类型并返回
    return out.to(original_x_dtype)
