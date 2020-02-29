from modin.engines.base.frame.partition_manager import BaseFrameManager
from .axis_partition import (
    PandasOnMPIFrameColumnPartition,
    PandasOnMPIFrameRowPartition,
)
from .partition import PandasOnMPIFramePartition


class MPIFrameManager(BaseFrameManager):
    """This class implements the interface in `BaseFrameManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = PandasOnMPIFramePartition
    _column_partitions_class = PandasOnMPIFrameColumnPartition
    _row_partition_class = PandasOnMPIFrameRowPartition
