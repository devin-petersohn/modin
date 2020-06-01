from ..frame.partition_manager import PandasOnRayFrameManager
from modin.engines.base.array.data import ModinArray


class ModinonRayArray(ModinArray):

    _frame_mgr_cls = PandasOnRayFrameManager
