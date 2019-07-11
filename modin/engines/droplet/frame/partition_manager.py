from modin.engines.base.frame.partition_manager import BaseFrameManager
from .axis_partition import (
PandasOnDropletFrameColumnPartition,
PandasOnDropletFrameRowPartition
)
from .partition import PandasOnDropletFramePartition


class DropletFrameManager(BaseFrameManager):

    _partition_class = PandasOnDropletFramePartition
    _column_partitions_class = PandasOnDropletFrameColumnPartition
    _row_partition_class = PandasOnDropletFrameRowPartition

    def __init__(self, partitions):
        self.partitions = partitions
