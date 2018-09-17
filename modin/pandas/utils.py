from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas

import collections
import numpy as np
import ray
import time
import gc

from ..data_management.factories import BaseFactory

_NAN_BLOCKS = {}
_MEMOIZER_CAPACITY = 1000  # Capacity per function


class LRUCache(object):
    """A LRUCache implemented with collections.OrderedDict

    Notes:
        - OrderedDict will record the order each item is inserted.
        - The head of the queue will be LRU items.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, key):
        """Retrieve item from cache and re-insert it to the back of the queue
        """
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def __setitem__(self, key, value):
        if key in self.cache:
            self.cache.pop(key)

        if len(self.cache) >= self.capacity:
            # Pop oldest items at the beginning of the queue
            self.cache.popitem(last=False)

        self.cache[key] = value


class memoize(object):
    """A basic memoizer that cache the input and output of the remote function

    Notes:
        - How is this implemented?
          This meoizer is implemented by adding a caching layer to the remote
          function's remote attribute. When user call f.remote(*args), we will
          first check against the cache, and then call the ray remote function
          if we can't find the return value in the cache.
        - When should this be used?
          This should be used when we anticipate temporal locality for the
          function. For example, we can reasonally assume users will perform
          columnar operation repetitively over time (like sum() or loc[]).
        - Caveat
          Don't use this decorator if the any argument to the remote function
          will mutate. Following snippet will fail
          ```py
              @memoize
              @ray.remote
              def f(obj):
                ...

              mutable_obj = [1]
              oid_1 = f.remote(mutable_obj) # will be cached

              mutable_obj.append(3)
              oid_2 = f.remote(mutable_obj) # cache hit!

              oid_1 == oid_2 # True!
           ```
           In short, use this function sparingly. The ideal case is that all
           inputs are ray ObjectIDs because they are immutable objects.
        - Future Development
          - Fix the mutability bug
          - Dynamic cache size (Fixed as 1000 for now)
    """

    def __init__(self, f):
        # Save of remote function
        self.old_remote_func = f.remote
        self.cache = LRUCache(capacity=_MEMOIZER_CAPACITY)

    def remote(self, *args):
        """Return cached result if the arguments are cached
        """
        args = tuple(args)

        if args in self.cache:
            cached_result = self.cache[args]
            return cached_result

        result = self.old_remote_func(*args)
        self.cache[args] = result
        return result


def post_task_gc(func):
    """Perform garbage collection after the task is executed.

    Usage:
        ```
        @ray.remote
        @post_task_gc
        def memory_hungry_op():
            ...
        ```
    Note:
        - This will invoke the GC for the entire process. Expect
          About 100ms latency.
        - We have a basic heuristic in place to balance of trade-off between
          speed and memory. If the task takes more than 500ms to run, we
          will do the GC.
    """

    def wrapped(*args):
        start_time = time.time()

        result = func(*args)

        duration_s = time.time() - start_time
        duration_ms = duration_s * 1000
        if duration_ms > 500:
            gc.collect()

        return result

    return wrapped


def _get_nan_block_id(n_row=1, n_col=1, transpose=False):
    """A memory efficient way to get a block of NaNs.

    Args:
        n_rows(int): number of rows
        n_col(int): number of columns
        transpose(bool): if true, swap rows and columns
    Returns:
        ObjectID of the NaN block
    """
    global _NAN_BLOCKS
    if transpose:
        n_row, n_col = n_col, n_row
    shape = (n_row, n_col)
    if shape not in _NAN_BLOCKS:
        arr = np.tile(np.array(np.NaN), shape)
        _NAN_BLOCKS[shape] = ray.put(pandas.DataFrame(data=arr))
    return _NAN_BLOCKS[shape]


def from_pandas(df):
    """Converts a pandas DataFrame to a Ray DataFrame.
    Args:
        df (pandas.DataFrame): The pandas DataFrame to convert.

    Returns:
        A new Ray DataFrame object.
    """
    from .dataframe import DataFrame

    return DataFrame(data_manager=BaseFactory.from_pandas(df))


def to_pandas(df):
    """Converts a Ray DataFrame to a pandas DataFrame/Series.
    Args:
        df (modin.DataFrame): The Ray DataFrame to convert.
    Returns:
        A new pandas DataFrame.
    """
    return df._data_manager.to_pandas()


"""
Indexing Section
    Generate View Copy Helpers
    Function list:
        - `extract_block` (ray.remote function, move to EOF)
        - `_generate_block`
        - `_repartition_coord_df`
    Call Dependency:
        - _generate_block calls extract_block remote
    Pipeline:
        - Repartition the dataframe by npartition
        - Use case:
              The dataframe is a DataFrameView, the two coord_dfs only
              describe the subset of the block partition data. We want
              to create a new copy of this subset and re-partition
              the new dataframe.
"""


def _generate_blocks(old_row, new_row, old_col, new_col,
                     block_partition_2d_oid_arr):
    """
    Given the four coord_dfs:
        - Old Row Coord df
        - New Row Coord df
        - Old Col Coord df
        - New Col Coord df
    and the block partition array, this function will generate the new
    block partition array.
    """

    # We join the old and new coord_df to find out which chunk in the old
    # partition belongs to the chunk in the new partition. The new coord df
    # should have the same index as the old coord df in order to align the
    # row/column. This is guaranteed by _repartition_coord_df.
    def join(old, new):
        return new.merge(
            old, left_index=True, right_index=True, suffixes=('_new', '_old'))

    row_grouped = join(old_row, new_row).groupby('partition_new')
    col_grouped = join(old_col, new_col).groupby('partition_new')

    oid_lst = []
    for row_idx, row_lookup in row_grouped:
        for col_idx, col_lookup in col_grouped:
            oid = extract_block.remote(
                block_partition_2d_oid_arr,
                row_lookup,
                col_lookup,
                col_name_suffix='_old')
            oid_lst.append(oid)
    return np.array(oid_lst).reshape(len(row_grouped), len(col_grouped))


# Indexing
#  Generate View Copy Helpers
# END


def _mask_block_partitions(blk_partitions, row_metadata, col_metadata):
    """Return the squeezed/expanded block partitions as defined by
    row_metadata and col_metadata.

    Note:
        Very naive implementation. Extract one scaler at a time in a double
        for loop.
    """
    col_df = col_metadata._coord_df
    row_df = row_metadata._coord_df

    result_oids = []
    shape = (len(row_df.index), len(col_df.index))

    for _, row_partition_data in row_df.iterrows():
        for _, col_partition_data in col_df.iterrows():
            row_part = row_partition_data.partition
            col_part = col_partition_data.partition
            block_oid = blk_partitions[row_part, col_part]

            row_idx = row_partition_data['index_within_partition']
            col_idx = col_partition_data['index_within_partition']

            result_oid = extractor.remote(block_oid, [row_idx], [col_idx])
            result_oids.append(result_oid)
    return np.array(result_oids).reshape(shape)


def _inherit_docstrings(parent, excluded=[]):
    """Creates a decorator which overwrites a decorated class' __doc__
    attribute with parent's __doc__ attribute. Also overwrites __doc__ of
    methods and properties defined in the class with the __doc__ of matching
    methods and properties in parent.

    Args:
        parent (object): Class from which the decorated class inherits __doc__.
        excluded (list): List of parent objects from which the class does not
            inherit docstrings.

    Returns:
        function: decorator which replaces the decorated class' documentation
            parent's documentation.
    """

    def decorator(cls):
        if parent not in excluded:
            cls.__doc__ = parent.__doc__
        for attr, obj in cls.__dict__.items():
            parent_obj = getattr(parent, attr, None)
            if parent_obj in excluded or \
                    (not callable(parent_obj) and
                     not isinstance(parent_obj, property)):
                continue
            if callable(obj):
                obj.__doc__ = parent_obj.__doc__
            elif isinstance(obj, property) and obj.fget is not None:
                p = property(obj.fget, obj.fset, obj.fdel, parent_obj.__doc__)
                setattr(cls, attr, p)

        return cls

    return decorator


def _fix_blocks_dimensions(blocks, axis):
    """Checks that blocks is 2D, and adds a dimension if not.
    """
    if blocks.ndim < 2:
        return np.expand_dims(blocks, axis=axis ^ 1)
    return blocks


@ray.remote
def _deploy_func(func, dataframe, *args):
    """Deploys a function for the _map_partitions call.
    Args:
        dataframe (pandas.DataFrame): The pandas DataFrame for this partition.
    Returns:
        A futures object representing the return value of the function
        provided.
    """
    if len(args) == 0:
        return func(dataframe)
    else:
        return func(dataframe, *args)


@ray.remote
def extractor(df_chunk, row_loc, col_loc):
    """Retrieve an item from remote block
    """
    # We currently have to do the writable flag trick because a pandas bug
    # https://github.com/pandas-dev/pandas/issues/17192
    try:
        row_loc.flags.writeable = True
        col_loc.flags.writeable = True
    except AttributeError:
        # Locators might be scaler or python list
        pass
    # Python2 doesn't allow writable flag to be set on this object. Copying
    # into a list allows it to be used by iloc.
    except ValueError:
        row_loc = list(row_loc)
        col_loc = list(col_loc)
    return df_chunk.iloc[row_loc, col_loc]


@ray.remote
def writer(df_chunk, row_loc, col_loc, item):
    """Make a copy of the block and write new item to it
    """
    df_chunk = df_chunk.copy()
    df_chunk.iloc[row_loc, col_loc] = item
    return df_chunk


@memoize
@ray.remote
def _blocks_to_series(*partition):
    """Used in indexing, concatenating blocks in a flexible way
    """
    if len(partition) == 0:
        return pandas.Series()

    partition = [pandas.Series(p.squeeze()) for p in partition]
    series = pandas.concat(partition)
    return series


@ray.remote
def extract_block(blk_partitions, row_lookup, col_lookup, col_name_suffix):
    """
    This function extracts a single block from blk_partitions using
    the row_lookup and col_lookup.

    Pass in col_name_suffix='_old' when operate on a joined df.
    """

    def apply_suffix(s):
        return s + col_name_suffix

    # Address Arrow Error:
    #   Buffer source array is read-only
    row_lookup = row_lookup.copy()
    col_lookup = col_lookup.copy()

    df_columns = []
    for row_idx, row_df in row_lookup.groupby(apply_suffix('partition')):
        this_column = []
        for col_idx, col_df in col_lookup.groupby(apply_suffix('partition')):
            block_df_oid = blk_partitions[row_idx, col_idx]
            block_df = ray.get(block_df_oid)
            chunk = block_df.iloc[row_df[apply_suffix(
                'index_within_partition')], col_df[apply_suffix(
                    'index_within_partition')]]
            this_column.append(chunk)
        df_columns.append(pandas.concat(this_column, axis=1))
    final_df = pandas.concat(df_columns)
    final_df.index = pandas.RangeIndex(0, final_df.shape[0])
    final_df.columns = pandas.RangeIndex(0, final_df.shape[1])

    return final_df
