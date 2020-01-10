from modin.engines.base.io import BaseIO
from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.engines.cloudburst.frame.data import PandasOnCloudburstFrame
from modin.engines.cloudburst.frame.partition import (
    PandasOnCloudburstFramePartition,
)
from modin.engines.cloudburst.task_wrapper import CloudburstTask
from modin.engines.base.io import CSVReader
from modin.backends.pandas.parsers import PandasCSVParser


class PandasOnCloudburstIO(BaseIO):
    frame_cls = PandasOnCloudburstFrame
    query_compiler_cls = PandasQueryCompiler
    build_args = dict(
        frame_cls=PandasOnCloudburstFrame,
        frame_partition_cls=PandasOnCloudburstFramePartition,
        query_compiler_cls=PandasQueryCompiler,
    )
    read_csv = type("", (CloudburstTask, PandasCSVParser, CSVReader), build_args).read
