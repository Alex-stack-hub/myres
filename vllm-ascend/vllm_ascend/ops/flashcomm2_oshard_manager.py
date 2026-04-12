from typing import Any

from vllm.model_executor.models.utils import extract_layer_index

from vllm_ascend.distributed.parallel_state import get_shard_weight_group
from vllm_ascend.ops.layer_shard_linear import (
    is_hidden_layer,
    post_process_after_loading_for_shard_weight_series,
    reach_layer_for_shard_weight_series,
    register_layer_to_shard_weight_series,
)
from vllm_ascend.utils import flashcomm2_enable, o_shard_enable


class Flashcomm2OShardManager:
    """管理FlashComm2 O-Shard特性的分片层

    此类实现了集中管理所有与Flashcomm2OShard层相关的逻辑。
    其主要职责是：
    1.  注册需要O-Sharding的Attention `o_proj`层。
    2.  将这些层存储和管理在一个字典中，映射层索引
        到层对象（`layer_index -> layer`）。
    3.  为外部调用者提供高级API，用于模型初始化、计算和权重加载等关键阶段。

    属性：
        _shard_layers: 存储已注册分片层的字典，
            将层索引（int）映射到其对应的层对象。
    """

    def __init__(self):
        """初始化Flashcomm2OShardManager"""
        self._shard_layers: dict[int, Any] = {}

    def flashcomm2_oshard_enable(self):
        """检查FlashComm2 O-Shard特性是否启用

        Returns:
            bool: 如果FlashComm2和O-Shard都启用，则返回True
        """
        return flashcomm2_enable() and o_shard_enable()

    def register_layer(self, layer: Any, prefetch_step: int = 1):
        """注册层以进行O-Sharding

        此方法首先检查O-Shard特性是否启用，以及提供的层是否符合目标条件（例如，隐藏层）。
        如果符合，它执行两个操作：
        1. 在内部`_shard_layers`字典中缓存层。
        2. 调用底层的`register_layer_to_shared_weight_series`
           函数将其注册用于通信。

        Args:
            layer: 要注册的层对象。
            prefetch_step: 注册层到共享权重系列时使用的预取步骤。
        """
        # 检查层是否为分片目标
        if is_hidden_layer(layer):
            # 提取层索引
            layer_idx = extract_layer_index(layer.prefix)
            # 缓存层
            self._shard_layers[layer_idx] = layer

            # 注册层到共享权重系列
            register_layer_to_shard_weight_series(
                series_name="o_proj", group=get_shard_weight_group(), layer=layer, prefetch_step=prefetch_step
            )

    def get_layer(self, layer_idx: int) -> Any | None:
        """通过索引安全地检索已注册的层

        Args:
            layer_idx: 要检索的层的索引。

        Returns:
            如果找到，返回层对象；否则返回None。
        """
        return self._shard_layers.get(layer_idx)

    def trigger_broadcast_for_layer(self, layer_prefix: str):
        """在模型计算期间触发特定层的广播

        此方法旨在在层的前向传递中调用。
        它从前缀中提取层索引，检索相应的
        已注册层对象，然后在满足所有条件时触发广播操作。

        Args:
            layer_prefix: 当前正在计算的层的名称前缀。
        """
        # 提取层索引
        layer_idx = extract_layer_index(layer_prefix)
        # 获取目标层
        target_layer = self.get_layer(layer_idx)

        # 确保层存在并满足分片条件
        if target_layer and is_hidden_layer(target_layer):
            reach_layer_for_shard_weight_series(target_layer)

    def post_process_after_loading(self):
        """在权重加载后对所有注册层执行后处理

        这应该在模型权重完全加载后调用一次。
        """
        if self._shard_layers:
            # 选择任何层（例如，第一个）来触发分片后处理
            any_layer = next(iter(self._shard_layers.values()))
            post_process_after_loading_for_shard_weight_series(any_layer)


# 创建Flashcomm2OShardManager的全局实例
flashcomm2_oshard_manager = Flashcomm2OShardManager()
