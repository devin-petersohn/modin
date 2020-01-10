from modin.engines.base.frame.partition_manager import BaseFrameManager
from modin.engines.cloudburst.frame.partition import PandasOnCloudburstFramePartition

class CloudburstFrameManager(BaseFrameManager):
    # This object uses DropletRemotePartition objects as the underlying store.
    _partition_class = PandasOnCloudburstFramePartition
    _column_partitions_class = None
    _row_partition_class = None
