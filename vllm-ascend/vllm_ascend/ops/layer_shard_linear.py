from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.utils import extract_layer_index

from vllm_ascend.distributed.parallel_state import get_shard_weight_group


def dispose_tensor(x: torch.Tensor):
    """释放张量内存

    Args:
        x: 要释放的张量
    """
    x.set_(torch.empty([], device=x.device, dtype=x.dtype))


@dataclass
class LayerMetadata:
    """层的元数据"""

    layer_idx: int  # 层的索引
    layer: LinearBase  # 层对象
    post_method: Callable[[torch.nn.Module], None]  # 量化方法中的`process_weights_after_loading`方法
    weight: torch.Tensor  # 权重张量
    window_idx: int  # 窗口的索引


@dataclass
class ShardWindowMetadata:
    """分片窗口的元数据"""

    weight: torch.Tensor  # 要按层分片的权重张量
    data_layer_idx: int  # 此窗口权重对应的层的索引
    work: torch.distributed.Work | None  # 异步广播工作


@dataclass
class SeriesMetadata:
    """权重分片系列的元数据"""

    group: GroupCoordinator
    start_layer: int
    end_layer: int
    num_layers: int
    prefetch_step: int
    dummy_weight: torch.Tensor  # 用于替换加载的权重矩阵的虚拟权重
    # 系列中的所有层共享同一个虚拟权重张量
    layers: list[LayerMetadata]
    shard_windows: list[ShardWindowMetadata]  # 用于预取的分片窗口。窗口大小为(`prefetch_step` + 1)，
    # 因为只需要存储接下来(`prefetch_step` + 1)层的权重
    window_offset: int  # 下一层的窗口索引

    def is_source(self, layer_idx) -> bool:
        """检查当前设备是否是指定层的源设备

        Args:
            layer_idx: 层索引

        Returns:
            bool: 如果当前设备是源设备，则返回True
        """
        return layer_idx % self.group.world_size == self.group.rank_in_group

    def post_process_after_loading(self):
        """加载后处理

        此方法每个系列只需要调用一次。
        """
        # 如果已经有分片窗口，则直接返回
        if self.shard_windows:
            return

        # 按层索引排序
        self.layers.sort(key=lambda x: x.layer_idx)
        # 设置层数
        self.num_layers = len(self.layers)
        # 确保系列中有层
        assert self.num_layers > 0, "No layers in the series"
        # 确保prefetch_step在有效范围内
        assert self.prefetch_step >= 0 and self.prefetch_step <= max(0, self.num_layers - 2), (
            "prefetch_step must be in [0, num_layers - 2]"
        )
        # 设置起始层和结束层
        self.start_layer = self.layers[0].layer_idx
        self.end_layer = self.layers[-1].layer_idx + 1

        # 处理每个层
        for layer_idx in range(self.start_layer, self.end_layer):
            # 获取当前层
            layer = self.layers[layer_idx - self.start_layer]
            # 确保层索引是连续的
            assert layer.layer_idx == layer_idx, "layer_idx must be consecutive"
            # 检查是否是源设备
            is_source = self.is_source(layer_idx)
            # 如果权重使用虚拟权重，临时创建一个副本，以便post方法调用不会影响其他也使用虚拟权重的层
            if not is_source:
                layer.weight.set_(torch.empty_like(self.dummy_weight))
            # 广播以获取真实权重
            dist.broadcast(
                layer.weight, src=self.group.ranks[layer_idx % self.group.world_size], group=self.group.device_group
            )
            # 调用量化方法中的`process_weights_after_loading`
            layer.post_method(layer.layer)
            # 计算当前步骤
            step = layer_idx - self.start_layer
            # 处理前`prefetch_step`层
            if step < self.prefetch_step:
                # 为前`prefetch_step`层构建窗口。这些权重可以在`forward()`中用于前`prefetch_step`层，因此也克隆权重
                self.shard_windows.append(
                    ShardWindowMetadata(
                        weight=layer.weight.clone().detach(),
                        data_layer_idx=layer_idx,
                        work=None,
                    )
                )
                # 设置窗口索引
                layer.window_idx = step
                # 当层不打算存储在此设备上时，链接到相应窗口的张量
                if not is_source:
                    layer.weight.set_(self.shard_windows[-1].weight)
            else:
                # 为预取构建一个额外的窗口。权重无用，所以只保留形状
                if step == self.prefetch_step:
                    self.shard_windows.append(
                        ShardWindowMetadata(
                            weight=torch.empty_like(layer.weight),
                            data_layer_idx=-1,
                            work=None,
                        )
                    )
                # 当层不打算存储在此设备上时，释放张量
                if not is_source:
                    dispose_tensor(layer.weight)
        # 释放虚拟张量，因为不再需要
        dispose_tensor(self.dummy_weight)

    def reach_layer(self, layer_idx: int):
        """到达层时的处理

        Args:
            layer_idx: 层索引
        """
        # 要预取的层的索引
        next_layer_idx = (layer_idx + self.prefetch_step) % self.num_layers + self.start_layer
        # 获取下一层
        next_layer = self.layers[next_layer_idx - self.start_layer]
        # 存储即将到来的层的权重的窗口索引
        next_layer.window_idx = self.window_offset
        # 获取窗口
        window = self.shard_windows[next_layer.window_idx]
        # 当层不打算存储在此设备上时，链接到相应窗口的张量
        if not self.is_source(next_layer_idx):
            next_layer.weight.set_(window.weight)
        # 通过滚动一步更新`window_offset`
        self.window_offset = (self.window_offset + 1) % (self.prefetch_step + 1)
        # 确保窗口的数据层索引与下一层索引不同
        assert window.data_layer_idx != next_layer_idx
        # 设置窗口的数据层索引
        window.data_layer_idx = next_layer_idx
        # 开始异步广播工作
        window.work = dist.broadcast(
            next_layer.weight,
            src=self.group.ranks[next_layer_idx % self.group.world_size],
            group=self.group.device_group,
            async_op=True,
        )

    def wait_weight(self, layer_idx: int):
        """等待权重

        Args:
            layer_idx: 层索引
        """
        # 确保有分片窗口
        assert self.shard_windows
        # 获取对应窗口
        window = self.shard_windows[self.layers[layer_idx - self.start_layer].window_idx]
        # 确保对应分片窗口中的数据是当前层的
        assert window.data_layer_idx == layer_idx
        # 如果有异步广播工作，等待它完成
        if window.work is not None:
            window.work.wait()
            window.work = None


@dataclass
class LayerExternalMetadata:
    """层的外部元数据"""

    series: SeriesMetadata
    layer_idx: int


# 系列字典，存储系列名称到SeriesMetadata的映射
_series_dict: dict[str, SeriesMetadata] = {}

# 层外部字典，存储层ID到LayerExternalMetadata的映射
_layer_external_dict: dict[int, LayerExternalMetadata] = {}


def _create_forward_wrapper(forward: Callable, series: SeriesMetadata, layer_idx: int) -> Callable:
    """创建前向包装器

    Args:
        forward: 原始前向函数
        series: 系列元数据
        layer_idx: 层索引

    Returns:
        包装后的前向函数
    """

    def wrapped_forward(*args, **kwargs):
        # 等待权重
        series.wait_weight(layer_idx)
        return forward(*args, **kwargs)

    return wrapped_forward


"""
将线性层注册到分片存储系列中。

在并行组中，每个设备存储系列中不同的、不重叠的层子集。
系列中的所有层必须具有相同的结构（同构）。第i层的权重矩阵
存储在设备(i % n)上，其中n是设备数量。

加载模型后，必须调用`post_process_after_loading_for_shard_weight_series(layer)`
对该系列的任何层进行初始化。

在执行期间，每次到达新层时，必须调用`reach_layer_for_shard_weight_series(layer)`
为该层预取权重。参数`prefetch_step`是一个非负整数k，用于管理
异步权重预取。每次调用`reach_layer_for_shard_weight_series(current_layer)`方法将
触发对系列中`current_layer`之后第k层权重的异步预取。

注意：层作为循环缓冲区管理。要预取的层的索引由以下公式确定：
- start_layer是系列中第一层的索引（包含）。
- end_layer是系列中最后一层的索引（不包含）。因此，系列包括所有索引在
  范围[start_layer, end_layer)内的层。
- total_layers = end_layer - start_layer
- prefetch_layer_idx = (layer_idx + prefetch_step) % total_layers + start_layer

为了保存当前层和k个预取层的权重，将为此系列创建一个(k + 1)个分片张量缓冲区的池。

参数：
    series_name: 此名称标识该层属于哪个系列。
    group: 用于处理异步通信的组协调器。建议为每个新系列创建一个新的组
        协调器。
    layer: 要注册的线性层对象。
    prefetch_step: 管理异步权重预取的整数。设置为0或1可以覆盖大多数情况。
"""


def register_layer_to_shard_weight_series(
    series_name: str,
    group: GroupCoordinator,
    layer: LinearBase,
    prefetch_step: int = 1,
):
    """注册层到分片权重系列

    Args:
        series_name: 系列名称
        group: 组协调器
        layer: 线性层对象
        prefetch_step: 预取步骤
    """
    global _series_dict
    # 如果系列不存在，创建一个新的
    if series_name not in _series_dict:
        _series_dict[series_name] = SeriesMetadata(
            group=group,
            start_layer=0,
            end_layer=0,
            num_layers=0,
            prefetch_step=prefetch_step,
            dummy_weight=torch.empty_like(layer.weight),
            layers=[],
            shard_windows=[],
            window_offset=prefetch_step,
        )
    # 获取系列
    series = _series_dict[series_name]
    # 确保层有量化方法
    assert layer.quant_method is not None
    # 提取层索引
    layer_idx = extract_layer_index(layer.prefix)
    # 添加层元数据
    series.layers.append(
        LayerMetadata(
            layer_idx=layer_idx,
            layer=layer,
            post_method=layer.quant_method.process_weights_after_loading,
            weight=layer.weight,
            window_idx=-1,
        )
    )
    # 丢弃原始的`process_weights_after_loading`方法，使其不会被其他人调用
    layer.quant_method.process_weights_after_loading = lambda layer: None
    # 当层不打算存储在此设备上时，释放张量并跳过权重加载
    if not series.is_source(layer_idx):
        dispose_tensor(layer.weight)
        layer.weight.weight_loader = lambda *args, **kwargs: None
    # 包装前向函数
    layer.forward = _create_forward_wrapper(layer.forward, series, layer_idx)
    # 存储层外部元数据
    global _layer_external_dict
    _layer_external_dict[id(layer)] = LayerExternalMetadata(
        series=series,
        layer_idx=layer_idx,
    )


def post_process_after_loading_for_shard_weight_series(layer: LinearBase):
    """分片权重系列加载后处理

    Args:
        layer: 线性层对象
    """
    # 获取层外部元数据
    ext = _layer_external_dict[id(layer)]
    # 调用系列的post_process_after_loading方法
    ext.series.post_process_after_loading()


def reach_layer_for_shard_weight_series(layer: LinearBase):
    """到达分片权重系列的层

    Args:
        layer: 线性层对象
    """
    # 获取层外部元数据
    ext = _layer_external_dict[id(layer)]
    # 调用系列的reach_layer方法
    ext.series.reach_layer(ext.layer_idx)


def wait_layer_for_shard_weight_series(layer: LinearBase):
    """等待分片权重系列的层

    Args:
        layer: 线性层对象
    """
    # 获取层外部元数据
    ext = _layer_external_dict[id(layer)]
    # 调用系列的wait_weight方法
    ext.series.wait_weight(ext.layer_idx)


@lru_cache(maxsize=1)
def get_current_model_num_hidden_layers() -> int:
    """获取当前模型的隐藏层数

    Returns:
        int: 隐藏层数
    """
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    return vllm_config.model_config.get_total_num_hidden_layers()


def is_hidden_layer(layer: LinearBase) -> bool:
    """检查是否是隐藏层

    Args:
        layer: 线性层对象

    Returns:
        bool: 如果是隐藏层，则返回True
    """
    # 获取隐藏层数
    num_hidden_layers = get_current_model_num_hidden_layers()
    # 提取层索引
    layer_idx = extract_layer_index(layer.prefix)
    # 检查层索引是否小于隐藏层数
    return layer_idx < num_hidden_layers


def register_all_layers_to_shard_weight_series(
    layer_sharding: list[LinearBase],
):
    """注册所有层到分片权重系列

    Args:
        layer_sharding: 要注册的层列表
    """
    # 遍历所有层
    for curr_layer in layer_sharding or []:
        # 如果是隐藏层
        if is_hidden_layer(curr_layer):
            # 提取层名称
            layer_name = curr_layer.prefix.split(".")[-1]
            # 注册层到分片权重系列
            register_layer_to_shard_weight_series(
                series_name=layer_name,
                group=get_shard_weight_group(),
                layer=curr_layer,
                prefetch_step=1,
            )
