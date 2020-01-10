from modin.engines.cloudburst.frame.partition_manager import CloudburstFrameManager
from modin.engines.base.frame.data import BasePandasFrame


class PandasOnCloudburstFrame(BasePandasFrame):

    _frame_mgr_cls = CloudburstFrameManager
