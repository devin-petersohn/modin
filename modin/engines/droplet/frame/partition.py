import pandas

from modin.data_management.utils import length_fn_pandas, width_fn_pandas
from modin.engines.base.frame.partition import BaseFramePartition
from modin.engines.droplet.utils import remote_apply


class PandasOnDropletFramePartition(BaseFramePartition):

    def __init__(self, future, length=None, width=None, call_queue=[]):
        self.future = future
        self._length_cache = length
        self._width_cache = width
        self.call_queue = call_queue

    def get(self):
        self.drain_call_queue()
        return self.future.get()

    def apply(self, func, **kwargs):
        call_queue = self.call_queue + [(func, kwargs)]
        return PandasOnDropletFramePartition(remote_apply(*call_queue, self.future))

    def add_to_apply_calls(self, func, **kwargs):
        return PandasOnDropletFramePartition(
            self.future,
            self._length_cache,
            self._width_cache,
            self.call_queue + [(func, kwargs)]
        )

    def drain_call_queue(self):
        if len(self.call_queue) == 0:
            return
        self.future = self.apply(lambda x: x).future
        self.call_queue = []

    def mask(self, row_indices=None, col_indices=None):
        new_obj = self.add_to_apply_calls(
            lambda df: pandas.DataFrame(df.iloc[row_indices, col_indices])
        )
        return new_obj

    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Note: If the underlying object is a Pandas DataFrame, this will likely
            only need to call `get`

        Returns:
            A Pandas DataFrame.
        """
        dataframe = self.get()
        assert type(dataframe) is pandas.DataFrame or type(dataframe) is pandas.Series

        return dataframe

    def to_numpy(self):
        """Convert the object stored in this partition to a NumPy Array.

        Returns:
            A NumPy Array.
        """
        return self.apply(lambda df: df.values).get()

    @classmethod
    def put(cls, obj):
        return cls(obj).apply(lambda x: x)

    @classmethod
    def preprocess_func(cls, func):
        return func

    @classmethod
    def length_extraction_fn(cls):
        """The function to compute the length of the object in this partition.

        Returns:
            A callable function.
        """
        return length_fn_pandas

    @classmethod
    def width_extraction_fn(cls):
        """The function to compute the width of the object in this partition.

        Returns:
            A callable function.
        """
        return width_fn_pandas

    _length_cache = None
    _width_cache = None

    def length(self):
        if self._length_cache is None:
            self._length_cache = self.apply(lambda x: len(x))
        return self._length_cache

    def width(self):
        if self._width_cache is None:
            self._width_cache = self.apply(lambda x: len(x.columns))
        return self._width_cache

    @classmethod
    def empty(cls):
        return cls(pandas.DataFrame())
