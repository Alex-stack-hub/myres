def _forward_fia_fullattention(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: AscendMetadata,
    output: torch.Tensor,
):
    # ========== 0. 快速返回：非目标层不dump ==========
    # 注意：这里 self.layer_idx 需要在 __init__ 中初始化并保存
    if not hasattr(self, 'layer_idx') or self.layer_idx != 15:
        # 非目标层，执行原逻辑
        batch_size = 1
        seq_q, num_q_heads, head_dim = query.shape
        seq_kv, num_kv_heads, _ = key.shape
        query_bsh = query.view(batch_size, seq_q, num_q_heads * head_dim)
        key_bsh = key.view(batch_size, seq_kv, num_kv_heads * head_dim)
        value_bsh = value.view(batch_size, seq_kv, num_kv_heads * head_dim)
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query_bsh,
            key_bsh,
            value_bsh,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSH",
            atten_mask=attn_metadata.attn_mask,
            scale=self.scale,
            sparse_mode=3,
        )
        attn_output = attn_output.view(seq_q, self.num_heads, self.head_size)
        output[:seq_q] = attn_output[:seq_q]
        return output

    # ========== 1. 目标层（Layer15）开始dump ==========
    import os
    import logging
    logger = logging.getLogger(__name__)
    device = query.device
    layer_idx = self.layer_idx

    # 辅助函数：直接在字符串前拼层数和设备
    def log_prefix():
        return f"[Layer {layer_idx}] [Device {device}] "

    # ========== 第一步：打印基础信息（层数+设备） ==========
    logger.info("=" * 80)
    logger.info(log_prefix() + "开始执行_forward_fia_fullattention，基础信息：")
    logger.info(log_prefix() + f"当前层数：{layer_idx} | 设备类型：{device} | 是否NPU：{device.type == 'npu'}")

    # ========== 第二步：打印算子输入前的原始Tensor（query/key/value） ==========
    def dump_tensor_info(name: str, tensor: torch.Tensor, do_save: bool = False):
        if tensor is None:
            logger.info(log_prefix() + f"{name}: None")
            return
        
        # 打印基本信息
        info = (log_prefix() + f"{name} | "
                f"Shape: {tuple(tensor.shape)} | Dtype: {tensor.dtype} | "
                f"Device: {tensor.device}")
        logger.info(info)
        
        # 打印统计信息（Mean/Std/Min/Max），处理 bfloat16
        if tensor.numel() > 0:
            # 转 float32 计算统计量，避免 bfloat16 在某些操作上的问题
            stat_tensor = tensor.cpu().detach()
            if stat_tensor.dtype == torch.bfloat16:
                stat_tensor = stat_tensor.to(torch.float32)
            
            mean_val = stat_tensor.mean().item()
            std_val = stat_tensor.std().item() if stat_tensor.numel() > 1 else 0.0
            min_val = stat_tensor.min().item()
            max_val = stat_tensor.max().item()
            
            logger.info(log_prefix() + f"{name} 统计: "
                    f"Mean={mean_val:.6f} | Std={std_val:.6f} | "
                    f"Min={min_val:.6f} | Max={max_val:.6f}")
            
            # 打印前5个元素的采样值，处理 bfloat16
            sample_tensor = tensor.flatten()[:5].cpu().detach()
            if sample_tensor.dtype == torch.bfloat16:
                sample_tensor = sample_tensor.to(torch.float32)
            sample = sample_tensor.tolist()
            logger.info(log_prefix() + f"{name} 前5个元素: {sample}")
        
        # 可选：保存到pt文件（直接保存原始 dtype，不转 float32）
        if do_save:
            save_dir = f"./layer_{layer_idx}_dump"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{name.replace('/', '_').replace(' ', '_')}.pt")
            torch.save(tensor.detach().cpu(), save_path)
            logger.info(log_prefix() + f"{name} 已保存至: {save_path}")

    logger.info(log_prefix() + "算子输入前 - Query/Key/Value原始信息：")
    dump_tensor_info("Query", query, do_save=True)
    dump_tensor_info("Key", key, do_save=True)
    dump_tensor_info("Value", value, do_save=True)
    dump_tensor_info("Output(初始)", output, do_save=False)

    # ========== 第三步：打印attn_metadata关键参数 ==========
    logger.info(log_prefix() + "AttnMetadata关键参数：")
    dump_tensor_info("attn_mask", attn_metadata.attn_mask, do_save=True)
    logger.info(log_prefix() + f"causal: {attn_metadata.causal} | attn_state: {attn_metadata.attn_state}")
    logger.info(log_prefix() + f"seq_lens: {attn_metadata.seq_lens} | num_actual_tokens: {attn_metadata.num_actual_tokens}")

    # ========== 第四步：打印reshape后的Query/Key/Value（BSH格式） ==========
    batch_size = 1
    seq_q, num_q_heads, head_dim = query.shape
    seq_kv, num_kv_heads, _ = key.shape

    query_bsh = query.view(batch_size, seq_q, num_q_heads * head_dim)
    key_bsh = key.view(batch_size, seq_kv, num_kv_heads * head_dim)
    value_bsh = value.view(batch_size, seq_kv, num_kv_heads * head_dim)

    logger.info(log_prefix() + "Reshape后（BSH格式）Tensor信息：")
    dump_tensor_info("Query_BSH", query_bsh, do_save=True)
    dump_tensor_info("Key_BSH", key_bsh, do_save=True)
    dump_tensor_info("Value_BSH", value_bsh, do_save=True)

    # ========== 第五步：打印npu_fused_infer_attention_score算子输入参数 ==========
    infer_kwargs = {
        "num_heads": self.num_heads,
        "num_key_value_heads": self.num_kv_heads,
        "input_layout": "BSH",
        "atten_mask": attn_metadata.attn_mask,
        "scale": self.scale,
        "sparse_mode": 3,
    }
    logger.info(log_prefix() + "npu_fused_infer_attention_score算子输入参数：")
    for k, v in infer_kwargs.items():
        if isinstance(v, torch.Tensor):
            dump_tensor_info(k, v, do_save=False)
        else:
            logger.info(log_prefix() + f"{k}: {v}")

    # ========== 第六步：执行算子并打印输出 ==========
    logger.info(log_prefix() + "开始执行npu_fused_infer_attention_score算子...")
    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
        query_bsh,
        key_bsh,
        value_bsh,
        **infer_kwargs
    )
    logger.info(log_prefix() + "算子执行完成 - 输出attn_output信息：")
    dump_tensor_info("Attn_Output", attn_output, do_save=True)

    # ========== 第七步：打印reshape后赋值给output的信息 ==========
    attn_output = attn_output.view(seq_q, self.num_heads, self.head_size)
    output[:seq_q] = attn_output[:seq_q]

    logger.info(log_prefix() + "赋值给output后最终信息：")
    dump_tensor_info("Output(最终)", output, do_save=True)
    logger.info("=" * 80)

    return output
