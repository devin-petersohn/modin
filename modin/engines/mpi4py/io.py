from modin.engines.base.io import BaseIO
from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.engines.mpi4py.frame.data import PandasOnMPIFrame
from modin.engines.mpi4py.frame.partition import PandasOnMPIFramePartition
from modin.engines.base.io import (
    CSVReader,
    JSONReader,
    ParquetReader,
    FeatherReader,
    SQLReader,
)
from modin.backends.pandas.parsers import (
    PandasCSVParser,
    PandasJSONParser,
    PandasParquetParser,
    PandasFeatherParser,
    PandasSQLParser,
)
from modin.engines.mpi4py.task_wrapper import MPITask


class PandasOnMPIIO(BaseIO):

    frame_cls = PandasOnMPIFrame
    query_compiler_cls = PandasQueryCompiler
    build_args = dict(
        frame_cls=PandasOnMPIFrame,
        frame_partition_cls=PandasOnMPIFramePartition,
        query_compiler_cls=PandasQueryCompiler,
    )

    read_csv = type("", (MPITask, PandasCSVParser, CSVReader), build_args).read
    read_json = type("", (MPITask, PandasJSONParser, JSONReader), build_args).read
    read_parquet = type(
        "", (MPITask, PandasParquetParser, ParquetReader), build_args
    ).read
    # Blocked on pandas-dev/pandas#12236. It is faster to default to pandas.
    # read_hdf = type("", (DaskTask, PandasHDFParser, HDFReader), build_args).read
    read_feather = type(
        "", (MPITask, PandasFeatherParser, FeatherReader), build_args
    ).read
    read_sql = type("", (MPITask, PandasSQLParser, SQLReader), build_args).read
