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
# This file is a part of the vllm-ascend project.
#
# 文件功能总结：
# Ascend 310P 专用注意力掩码构建器
# 提供 310P 设备上的注意力掩码生成和缓存功能
# 主要功能：
# 1. 生成因果注意力掩码 (causal additive mask)
# 2. 管理注意力掩码缓存
# 3. 支持滑动窗口注意力 (SWA) 掩码
# 4. 处理 310P 特定的内存格式转换
#

import torch
import torch_npu

from vllm_ascend.attention.attention_v1 import AscendMetadata
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, nd_to_nz_2d, nd_to_nz_spec


class AttentionMaskBuilder310:
    chunked_prefill_attn_mask = None
    max_seqlen = 16384

    def __init__(self, device: torch.device, max_seqlen: int):
        """
        初始化 310P 设备的注意力掩码构建器

        参数:
            device (torch.device): 用于分配张量的设备
            max_seqlen (int): 序列的最大长度（包括提示和生成文本）

        功能:
            1. 设置类最大序列长度
            2. 初始化注意力掩码缓存为 None
            3. 保存设备引用
            4. 初始化滑动窗口注意力掩码为 None
        """
        AttentionMaskBuilder310.max_seqlen = max_seqlen  # 设置类最大序列长度
        self.attn_mask_cache = None  # 注意力掩码缓存
        self.device = device  # 设备引用
        self.swa_mask = None  # 滑动窗口注意力掩码

    @staticmethod
    def gen_causal_additive_mask(max_seq_len: int, device: torch.device):
        """
        生成标准的因果下三角注意力掩码

        上三角部分填充负无穷大 (float("-inf")) 以掩盖未来令牌，
        而下三角部分保持为 0。

        参数:
            max_seq_len (int): 掩码的最大序列长度
            device (torch.device): 张量的目标设备

        返回:
            torch.Tensor: 表示因果掩码的 float16 张量

        实现步骤:
            1. 创建下三角布尔矩阵
            2. 获取上三角部分（非下三角）
            3. 创建零矩阵，并将上三角部分填充为负无穷大
        """
        tril = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=device).tril_()  # 创建下三角矩阵
        upper = ~tril  # 获取上三角部分（取反）
        mask = torch.zeros((max_seq_len, max_seq_len), dtype=torch.float16, device=device)  # 创建零矩阵
        mask.masked_fill_(upper, float("-inf"))  # 将上三角部分填充为负无穷大
        return mask

    @classmethod
    def get_splitfuse_mask(cls, attn_metadata: AscendMetadata, device: torch.device):
        """
        生成并格式化 SplitFuse（分块预填充）解码的注意力掩码

        根据查询起始位置和上下文长度计算所需的特定索引，
        从全局分块掩码中选择相关部分，并将结果转换为 NPU 特定的分形格式。

        参数:
            attn_metadata (AscendMetadata): 包含查询起始位置和序列长度的元数据
            device (torch.device): 执行操作的设备

        返回:
            torch.Tensor: 转换为 ACL_FORMAT_FRACTAL_NZ 格式的 splitfuse 注意力掩码

        实现步骤:
            1. 如果分块预填充掩码未初始化，则生成一个
            2. 从元数据中提取查询起始位置和序列长度
            3. 计算每个查询的长度和上下文长度
            4. 构建位置索引列表
            5. 从全局掩码中选择对应位置的行
            6. 转换为 NPU 分形格式
        """
        if cls.chunked_prefill_attn_mask is None:  # 如果分块预填充掩码未初始化
            cls.chunked_prefill_attn_mask = cls.gen_causal_additive_mask(cls.max_seqlen, device)  # 生成因果掩码
        qsl = attn_metadata.query_start_loc.to("cpu", dtype=torch.int32)  # 查询起始位置（转到 CPU）
        qlens = qsl[1:] - qsl[:-1]  # 计算每个查询的长度（相邻位置差）
        q_list = qlens.tolist()  # 转换为 Python 列表
        context_lens = attn_metadata.seq_lens.to("cpu", dtype=torch.int32)  # 上下文长度（转到 CPU）
        c_list = context_lens.tolist()  # 转换为 Python 列表
        # 构建位置列表：对于每个查询，从 (上下文长度 - 查询长度) 到 (上下文长度 - 1)
        pos_list = [p for ql, cl in zip(q_list, c_list) for p in range(cl - ql, cl)]
        position = torch.tensor(pos_list, dtype=torch.int32, device=device)  # 转换为设备上的张量
        splitfuse_mask = cls.chunked_prefill_attn_mask.index_select(0, position)  # 从全局掩码中选择对应行
        # 转换为 NPU 分形格式：先应用 nd_to_nz_spec，然后进行格式转换
        splitfuse_mask_nz = torch_npu.npu_format_cast(nd_to_nz_spec(splitfuse_mask).contiguous(), ACL_FORMAT_FRACTAL_NZ)
        return splitfuse_mask_nz

    def get_swa_mask(self, dtype: torch.dtype, sliding_window):
        """
        生成或获取缓存的滑动窗口注意力 (SWA) 掩码

        此掩码仅允许在特定的局部窗口（对角线带）内进行注意力计算，
        掩盖过去或未来太远的令牌。

        参数:
            dtype (torch.dtype): 掩码的数据类型
            sliding_window (int): 滑动窗口的大小

        返回:
            torch.Tensor: 转换为 ACL_FORMAT_FRACTAL_NZ 格式的 SWA 掩码

        实现步骤:
            1. 确保数据类型为 float16
            2. 如果滑动窗口不为 None 且 SWA 掩码未缓存，则生成新掩码
            3. 创建布尔掩码，标记需要被掩盖的位置
            4. 创建零矩阵，并将需要掩盖的位置填充为负无穷大
            5. 转换为 NPU 分形格式并缓存
        """
        assert dtype == torch.float16  # 确保数据类型为 float16
        if sliding_window is not None and self.swa_mask is None:  # 需要生成新掩码
            mask = torch.ones(self.max_seqlen, self.max_seqlen, dtype=torch.bool)  # 创建全1布尔矩阵
            triu_mask = torch.triu(mask, diagonal=1).to(self.device)  # 上三角部分（对角线以上）
            tril_mask = torch.tril(mask, -sliding_window).to(self.device)  # 下三角部分（滑动窗口以下）
            mask = triu_mask + tril_mask  # 合并需要掩盖的区域
            swa_mask = torch.zeros((self.max_seqlen, self.max_seqlen), dtype=dtype, device=self.device)  # 创建零矩阵
            swa_mask.masked_fill_(mask, float("-inf"))  # 将需要掩盖的区域填充为负无穷大
            self.swa_mask = torch_npu.npu_format_cast(
                nd_to_nz_2d(swa_mask), ACL_FORMAT_FRACTAL_NZ
            )  # 转换为 NPU 分形格式
        return self.swa_mask

    def get_attention_mask(self, model_config) -> torch.Tensor:
        """
        根据模型配置获取适当的注意力掩码

        明确检查 'pooling' 运行器类型，310P 硬件不支持此类型。

        参数:
            model_config: 包含运行器详细信息的配置对象

        返回:
            torch.Tensor: 因果注意力掩码

        抛出:
            NotImplementedError: 如果 runner_type 是 'pooling'

        功能:
            1. 检查 runner_type 是否为 'pooling'，如果是则抛出异常
            2. 调用 _get_causal_mask 获取因果注意力掩码
        """
        if getattr(model_config, "runner_type", None) == "pooling":  # 检查是否为 pooling 类型
            # TODO: pooling 模型即将支持
            raise NotImplementedError("310P does not support runner_type='pooling'")  # 310P 不支持 pooling 类型
        return self._get_causal_mask(self.max_seqlen)  # 获取因果注意力掩码

    def _get_causal_mask(self, max_seq_len: int) -> torch.Tensor:
        """
        内部方法：获取或更新缓存的因果注意力掩码

        如果缓存为空或请求的长度超过缓存长度，
        则生成新掩码并转换为 NPU 分形格式。

        参数:
            max_seq_len (int): 所需的序列长度

        返回:
            torch.Tensor: 缓存的因果掩码，格式为 ACL_FORMAT_FRACTAL_NZ

        功能:
            1. 检查注意力掩码缓存是否为空
            2. 如果为空，则生成新的因果掩码
            3. 将掩码转换为 NPU 分形格式并缓存
            4. 返回缓存的掩码
        """
        if self.attn_mask_cache is None:  # 如果缓存为空
            attn_mask = self.gen_causal_additive_mask(max_seq_len, self.device)  # 生成因果掩码
            self.attn_mask_cache = torch_npu.npu_format_cast(
                nd_to_nz_2d(attn_mask), ACL_FORMAT_FRACTAL_NZ
            )  # 转换为 NPU 分形格式
        return self.attn_mask_cache  # 返回缓存的掩码
