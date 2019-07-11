from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modin.engines.base.io import BaseIO
from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.engines.droplet.frame.partition_manager import DropletFrameManager


class PandasOnDropletIO(BaseIO):

    frame_mgr_cls = DropletFrameManager
    query_compiler_cls = PandasQueryCompiler
