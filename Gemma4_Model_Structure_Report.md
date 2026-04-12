# Gemma 4 模型结构深度分析报告

## 1. Gemma 4 模型家族概述

Gemma 4 是 Google DeepMind 推出的新一代开源语言模型家族，包含四个不同规模的变体，以满足不同硬件和用例的需求：

| 模型名称 | 参数量 | 架构特点 | 适用场景 |
|----------|--------|----------|----------|
| Gemma 4 - E2B | 20亿有效参数 | 密集模型，每层嵌入（Per-Layer Embeddings） | 移动设备、边缘计算 |
| Gemma 4 - E4B | 40亿有效参数 | 密集模型，每层嵌入 | 移动设备、边缘计算 |
| Gemma 4 - 31B | 310亿参数 | 密集模型，标准架构 | 服务器端、高性能计算 |
| Gemma 4 - 26B A4B | 260亿总参数，40亿激活参数 | 混合专家（MoE），稀疏激活 | 高效率推理，服务器端 |

**共同特性**：
- 多模态支持：所有变体均支持图像输入，可处理可变宽高比和分辨率的图像
- 小模型（E2B/E4B）额外支持音频输入
- 交错局部注意力与全局注意力层
- 全局注意力层采用 K=V、p-RoPE 等效率优化技术

## 2. 整体架构设计

Gemma 4 的整体架构延续了 Gemma 3 的设计理念，但在注意力机制、位置编码和多模态处理等方面进行了重要改进。

### 2.1 架构核心组件

![整体架构示意图](A%20Visual%20Guide%20to%20Gemma%204%20-%20by%20Maarten%20Grootendorst_files/37fe0e0b-90b8-445b-b2a8-0a402605cc18_6208x4980.jpg)

Gemma 4 的主要架构组件包括：

1. **文本嵌入层**：将输入 tokens 映射为隐藏状态向量
2. **交错注意力层**：局部注意力（滑动窗口）与全局注意力交替堆叠
3. **前馈网络**：每层包含 MLP 或 MoE 块
4. **视觉编码器**（可选）：处理图像输入，生成视觉 tokens
5. **音频编码器**（仅小模型）：处理音频输入，生成音频 embeddings
6. **输出层**：生成下一个 token 的概率分布

### 2.2 源代码结构概览

主要的实现文件位于 `vllm/vllm/model_executor/models/gemma4.py`，包含以下核心类：

- `Gemma4Model`：主模型类，管理嵌入层和解码器层
- `Gemma4DecoderLayer`：单个解码器层，包含注意力、MLP/MoE 和 PLE 组件
- `Gemma4Attention`：注意力机制实现，支持滑动窗口和全局注意力
- `Gemma4MLP`：标准前馈网络
- `Gemma4MoE`：混合专家实现
- `Gemma4Router`：MoE 路由模块
- `Gemma4ForCausalLM`：因果语言模型封装

## 3. 注意力机制：局部注意力与全局注意力交错

### 3.1 交错层设计

![交错层示意图](A%20Visual%20Guide%20to%20Gemma%204%20-%20by%20Maarten%20Grootendorst_files/4c6e4126-0b68-4276-bf59-6f3a3341b597_3932x3352.jpg)

Gemma 4 采用局部注意力（滑动窗口注意力）与全局注意力交错的设计：

- **局部注意力**：每个 token 仅关注窗口内的前驱 tokens
  - E2B/E4B：滑动窗口大小 512 tokens
  - 26B A4B/31B：滑动窗口大小 1024 tokens
- **全局注意力**：每个 token 关注所有前驱 tokens
- **交错模式**：
  - E2B：4层局部注意力 + 1层全局注意力（4:1）
  - 其他模型：5层局部注意力 + 1层全局注意力（5:1）
- **最后一层总是全局注意力**：确保模型能够捕捉完整的序列上下文

### 3.2 源代码实现

在 `Gemma4DecoderLayer` 中，通过 `layer_type` 配置确定每层的注意力类型：

```python
layer_type = config.layer_types[layer_idx]
self.is_full_attention = layer_type == "full_attention"
if self.is_full_attention:
    head_dim = getattr(config, "global_head_dim", config.head_dim)
else:
    head_dim = config.head_dim
```

`Gemma4Attention` 类根据 `layer_type` 初始化不同的 RoPE 参数和滑动窗口设置：

```python
# Determine layer type and sliding window
layer_idx = extract_layer_index(prefix)
layer_type = config.layer_types[layer_idx]
self.is_sliding = layer_type == "sliding_attention"
sliding_window = config.sliding_window if self.is_sliding else None
```

## 4. 全局注意力效率优化

全局注意力层虽然提供了完整的上下文感知，但计算和内存开销较大。Gemma 4 采用了多种技术来提升其效率。

### 4.1 分组查询注意力（Grouped Query Attention, GQA）

![GQA示意图](A%20Visual%20Guide%20to%20Gemma%204%20-%20by%20Maarten%20Grootendorst_files/46450e20-c6c9-4e8b-b9c4-2aa941210628_4272x2088.jpg)

- **局部注意力层**：2个查询头共享1个KV头
- **全局注意力层**：8个查询头共享1个KV头
- **Key维度加倍**：为了补偿KV头减少可能带来的性能损失，Key的维度被加倍

### 4.2 K=V 技术

在全局注意力层中，Key 和 Value 被设置为相等（K=V），进一步减少 KV 缓存的内存占用。

源代码中通过 `use_k_eq_v` 参数控制：

```python
use_k_eq_v = self.is_full_attention and getattr(
    config, "attention_k_eq_v", False
)
```

对于 k_eq_v 层，检查点中只有 k_proj 权重，v_proj 通过复制 k_proj 获得：

```python
if "self_attn.k_proj" in name and k_eq_v_layer_indices:
    m = re.search(r"layers\.(\d+)\.", name)
    if m and int(m.group(1)) in k_eq_v_layer_indices:
        yield name, weight
        yield name.replace("k_proj", "v_proj"), weight.clone()
        continue
```

### 4.3 部分旋转位置编码（p-RoPE）

![p-RoPE示意图](A%20Visual%20Guide%20to%20Gemma%204%20-%20by%20Maarten%20Grootendorst_files/fea15481-672a-4fe2-b803-5f4d493998d5_4770x2460.png)

传统的 RoPE 对所有维度对应用旋转，其中低频对包含的定位信息很少，反而可能干扰语义表示。p-RoPE 只对前 p% 的维度对应用旋转（Gemma 4 中 p=0.25）。

**优势**：
- 保留低频对的语义信息
- 提高长上下文处理能力
- 减少旋转空间，提升模型泛化性

源代码中通过 `rope_parameters` 配置：

```python
if layer_type in config.rope_parameters:
    rope_parameters = dict(config.rope_parameters[layer_type])
else:
    rope_parameters = dict(config.rope_parameters.copy())
```

### 4.4 KV 共享层（YOCO 优化）

Gemma 4 支持 You Only Cache Once (YOCO) 优化，后 `num_kv_shared_layers` 层与前层共享 KV 缓存，减少预填充阶段的计算。

```python
num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)
if num_kv_shared_layers > 0:
    first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared_layers
    if layer_idx >= first_kv_shared_layer_idx:
        self.is_kv_shared_layer = True
```

## 5. 视觉编码器

### 5.1 视觉 Transformer（ViT）基础

![视觉编码器示意图](A%20Visual%20Guide%20to%20Gemma%204%20-%20by%20Maarten%20Grootendorst_files/39531efc-71c3-44b6-97aa-546b108fb4d6_4770x1242.jpg)

Gemma 4 使用基于 Vision Transformer 的视觉编码器：
- 将输入图像分割为 16×16 像素的 patches
- 每个 patch 通过 Transformer 编码为 embedding
- 不同模型规模的视觉编码器参数量不同：
  - E2B/E4B：1.5亿参数
  - 26B A4B/31B：5.5亿参数

### 5.2 可变宽高比支持

![可变宽高比示意图](A%20Visual%20Guide%20to%20Gemma%204%20-%20by%20Maarten%20Grootendorst_files/3015e48f-a122-4af5-b443-a95681039f9c_4770x2082.png)

传统 ViT 要求输入为正方形，Gemma 4 通过以下技术支持可变宽高比：

1. **自适应调整大小**：保持原始宽高比，必要时填充
2. **2D RoPE**：将 embedding 分为两部分，分别编码宽度和高度位置信息
3. **Patch 池化**：将相邻的 3×3 patches 池化为单个 embedding

### 5.3 可变分辨率支持

![可变分辨率示意图](A%20Visual%20Guide%20to%20Gemma%204%20-%20by%20Maarten%20Grootendorst_files/f26153b1-ec96-4c04-ad84-717bc134b06b_4020x2046.png)

用户可指定软 token 预算（70, 140, 280, 560, 1120 tokens），编码器根据预算调整图像分辨率：

| 预算 | 近似分辨率 | 说明 |
|------|------------|------|
| 70 tokens | 336×336 | 低分辨率，快速处理 |
| 140 tokens | 480×480 | 中等分辨率 |
| 280 tokens | 672×672 | 高分辨率 |
| 560 tokens | 960×960 | 超高分辨率 |
| 1120 tokens | 1344×1344 | 极高分辨率，细节丰富 |

### 5.4 线性投影与归一化

视觉 embeddings 通过线性投影层和 RMSNorm 对齐到语言模型的嵌入空间：

```python
# 伪代码表示
visual_embeddings = vision_encoder(image_patches)
projected_embeddings = linear_projection(visual_embeddings)
normalized_embeddings = rms_norm(projected_embeddings)
```

## 6. 混合专家（Mixture of Experts, MoE）架构

### 6.1 MoE 基本概念

![MoE示意图](A%20Visual%20Guide%20to%20Gemma%204%20-%20by%20Maarten%20Grootendorst_files/eb2f917c-31b6-4461-a881-d1ae4a6d0071_4146x2982.jpg)

Gemma 4 26B A4B 采用 MoE 架构：
- **总参数**：260亿（稀疏参数）
- **激活参数**：40亿（每次推理仅激活部分专家）
- **专家数量**：128个
- **激活专家数**：每次推理激活8个专家 + 1个共享专家

### 6.2 路由机制

`Gemma4Router` 负责将 tokens 路由到合适的专家：

```python
class Gemma4Router(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps, has_weight=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.proj = GateLinear(self.hidden_size, config.num_experts, bias=False)
    
    def forward(self, x):
        x = self.norm(x)
        x = x * self.root_size.to(x.dtype)
        x = x * self.scale.to(x.dtype)
        router_logits, _ = self.proj(x)
        return router_logits
```

路由过程：
1. 对输入进行 RMSNorm 归一化
2. 应用根尺寸缩放（1/√hidden_size）
3. 应用逐维度学习缩放
4. 投影到专家 logits

### 6.3 专家计算

`Gemma4MoE` 使用 `FusedMoE` 实现高效的专家计算：

```python
def routing_function(hidden_states, gating_output, topk, renormalize):
    _, topk_ids = torch.topk(gating_output, k=topk, dim=-1)
    router_probabilities = torch.nn.functional.softmax(gating_output, dim=-1)
    indicator = torch.nn.functional.one_hot(
        topk_ids, num_classes=gating_output.size(-1)
    ).sum(dim=-2)
    gate_weights = indicator * router_probabilities
    # 重归一化...
    return topk_weights, topk_ids
```

### 6.4 共享专家

Gemma 4 MoE 包含一个**共享专家**，其规模是普通专家的3倍，始终被激活，负责处理通用知识。

## 7. 每层嵌入（Per-Layer Embeddings, PLE）

### 7.1 PLE 设计原理

![PLE示意图](A%20Visual%20Guide%20to%20Gemma%204%20-%20by%20Maarten%20Grootendorst_files/3253caa8-0499-42c7-beaf-ff1e828d890e_3012x3510.jpg)

小模型（E2B/E4B）使用 PLE 技术提升参数效率：
- **核心思想**：为每个 token 在每一层准备专门的嵌入向量
- **存储方式**：存储在闪存中，仅在推理开始时加载一次
- **维度**：256维（远小于主嵌入的1536/2560维）
- **参数量**：262,144 tokens × 35层 × 256维 ≈ 23亿参数

### 7.2 源代码实现

在 `Gemma4DecoderLayer` 中，PLE 相关组件：

```python
if (self.hidden_size_per_layer_input is not None
        and self.hidden_size_per_layer_input > 0):
    # Gate: projects hidden_states → per-layer dim for gating
    self.per_layer_input_gate = ReplicatedLinear(
        self.hidden_size,
        self.hidden_size_per_layer_input,
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.per_layer_input_gate",
        return_bias=False,
    )
    # Projection: projects gated per-layer input back → hidden size
    self.per_layer_projection = ReplicatedLinear(
        self.hidden_size_per_layer_input,
        self.hidden_size,
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.per_layer_projection",
        return_bias=False,
    )
    self.post_per_layer_input_norm = RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps
    )
```

前向传播中的 PLE 处理：

```python
if per_layer_input is not None and self.per_layer_input_gate is not None:
    gate = self.per_layer_input_gate(hidden_states)
    gate = torch.nn.functional.gelu(gate, approximate="tanh")
    gated_per_layer = gate * per_layer_input
    per_layer_contribution = self.per_layer_projection(gated_per_layer)
    per_layer_contribution = self.post_per_layer_input_norm(
        per_layer_contribution
    )
    hidden_states = hidden_states + per_layer_contribution
```

### 7.3 优势
- **减少 RAM 使用**：大嵌入表存储在闪存中
- **提升表达力**：每层有专门的嵌入表示
- **保持效率**：仅需一次查找，后续层复用

## 8. 音频编码器（小模型专有）

### 8.1 音频处理流程

![音频编码器示意图](A%20Visual%20Guide%20to%20Gemma%204%20-%20by%20Maarten%20Grootendorst_files/be2618d8-6c06-4e87-bc30-e865ccab1f15_5790x2760.png)

E2B/E4B 模型支持音频输入，处理流程包括：

1. **特征提取**：通过梅尔频谱图将原始音频转换为2D时频表示
2. **分块处理**：将梅尔特征分组为 chunks
3. **下采样**：通过2D卷积层缩短序列长度，生成"软 tokens"
4. **Conformer 编码**：使用类 Transformer 的 Conformer 编码器处理
5. **线性投影**：将音频 embeddings 投影到语言模型嵌入空间

### 8.2 Conformer 架构

Conformer 结合了 Transformer 的自注意力机制和 CNN 的局部特征提取能力：
- **自注意力模块**：捕捉全局依赖
- **卷积模块**：提取局部特征
- **前馈网络**：非线性变换

## 9. 源代码深度分析

### 9.1 模型初始化与配置

`Gemma4Model` 是主模型类，负责初始化所有组件：

```python
class Gemma4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _get_text_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        
        # 初始化嵌入层
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        
        # 初始化 PLE 组件（如果启用）
        if (self.hidden_size_per_layer_input is not None
                and self.hidden_size_per_layer_input > 0):
            # ... PLE 初始化代码
        
        # 创建解码器层
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Gemma4DecoderLayer(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        
        # YOCO 优化：将层分为自解码器和交叉解码器
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        self.self_decoder = Gemma4SelfDecoderLayers(...)
        self.cross_decoder = Gemma4CrossDecoderLayers(...)
```

### 9.2 权重加载机制

`Gemma4Model.load_weights` 方法负责从检查点加载权重，支持：
- QKV 投影的打包/解包
- MoE 专家权重的3D到2D转换
- k_eq_v 层的权重复制
- 量化缩放参数的映射

### 9.3 前向传播流程

前向传播支持两种模式：
1. **标准模式**：顺序执行所有解码器层
2. **快速预填充模式**（YOCO）：分两阶段执行，减少 KV 缓存重复计算

```python
def forward(self, input_ids, positions, intermediate_tensors=None, ...):
    if self.fast_prefill_enabled:
        # 快速预填充路径
        hidden_states = self.fast_prefill_forward(...)
    else:
        # 标准路径
        if get_pp_group().is_first_rank:
            # 处理输入嵌入和 PLE
            hidden_states = self.embed_input_ids(input_ids)
            per_layer_inputs = self.get_per_layer_inputs(input_ids)
            per_layer_inputs = self.project_per_layer_inputs(
                hidden_states, per_layer_inputs
            )
        else:
            # 从中间张量恢复状态
            hidden_states = intermediate_tensors["hidden_states"]
            per_layer_inputs = intermediate_tensors.get("per_layer_inputs")
        
        # 逐层处理
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                per_layer_input=layer_per_input,
                **kwargs,
            )
    
    return hidden_states
```

## 10. 性能与效率分析

### 10.1 计算效率优化

| 优化技术 | 作用 | 影响模型 |
|----------|------|----------|
| 滑动窗口注意力 | 减少注意力计算复杂度 | 所有模型 |
| GQA | 减少 KV 缓存大小 | 所有模型 |
| K=V | 进一步减少 KV 缓存 | 全局注意力层 |
| p-RoPE | 改进长上下文处理 | 全局注意力层 |
| MoE | 稀疏激活，减少计算量 | 26B A4B |
| PLE | 大嵌入表存储于闪存 | E2B/E4B |
| YOCO | 减少预填充阶段计算 | 所有模型 |

### 10.2 内存效率优化

1. **KV 缓存优化**：GQA + K=V 大幅减少全局注意力层的缓存需求
2. **参数稀疏性**：MoE 仅激活部分参数，减少激活内存
3. **存储分离**：PLE 将大嵌入表移至闪存，减少 RAM 占用
4. **共享缓存**：YOCO 允许层间共享 KV 缓存

### 10.3 多模态支持效率

- **自适应图像处理**：根据 token 预算动态调整分辨率
- **2D RoPE**：更好地处理可变宽高比图像
- **统一投影层**：将视觉/音频 embeddings 对齐到文本嵌入空间

## 11. 总结

Gemma 4 模型家族通过创新的架构设计，在保持强大性能的同时，显著提升了计算和内存效率。关键创新包括：

1. **分层注意力设计**：局部与全局注意力交错，平衡效率与效果
2. **多维度效率优化**：GQA、K=V、p-RoPE 等技术协同作用
3. **灵活的 MoE 架构**：26B A4B 模型实现高参数效率
4. **创新的 PLE 技术**：小模型实现大容量嵌入而不占用宝贵 RAM
5. **先进的多模态支持**：可变宽高比/分辨率图像处理，小模型支持音频
6. **系统级优化**：YOCO、快速预填充等提升推理速度

这些设计使 Gemma 4 能够适应从移动设备到数据中心的广泛部署场景，为开源大模型的发展树立了新的标杆。

---

**图表引用**：本文所有图表均来自 Maarten Grootendorst 的《A Visual Guide to Gemma 4》，存储于 `A Visual Guide to Gemma 4 - by Maarten Grootendorst_files/` 目录中。

**源代码分析基于**：`vllm/vllm/model_executor/models/gemma4.py`

**报告生成日期**：2026年4月9日