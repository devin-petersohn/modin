from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas

from pandas.compat import string_types
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (_get_dtype_from_object, is_list_like)

from .partitioning.partition_collections import BlockPartitions, RayBlockPartitions
from .partitioning.remote_partition import RayRemotePartition


class PandasDataManager(object):
    """This class implements the logic necessary for operating on partitions
        with a Pandas backend. This logic is specific to Pandas.
    """

    def __init__(self, block_partitions_object, index, columns, dtypes=None):
        assert isinstance(block_partitions_object, BlockPartitions)
        self.data = block_partitions_object
        self.index = index
        self.columns = columns
        if dtypes is not None:
            self._dtype_cache = dtypes

    # dtypes
    _dtype_cache = None

    def _get_dtype(self):
        if self._dtype_cache is None:
            map_func = lambda df: df.dtypes

            def func(row):
                return find_common_type(row.values)

            self._dtype_cache = self.data.full_reduce(map_func, lambda df: df.apply(func, axis=0), 0)
            self._dtype_cache.index = self.columns
        return self._dtype_cache

    def _set_dtype(self, dtypes):
        self._dtype_cache = dtypes

    dtypes = property(_get_dtype, _set_dtype)

    # Index and columns objects
    # These objects are currently not distributed.
    # Note: These are more performant as pandas.Series objects than they are as
    # pandas.DataFrame objects.
    #
    # _index_cache is a pandas.Series that holds the index
    _index_cache = None
    # _columns_cache is a pandas.Series that holds the columns
    _columns_cache = None

    def _get_index(self):
        return self._index_cache.index

    def _get_columns(self):
        return self._columns_cache.index

    def _set_index(self, new_index):
        if self._index_cache is not None:
            self._index_cache.index = new_index
        else:
            self._index_cache = pandas.Series(index=new_index)

    def _set_columns(self, new_columns):
        if self._columns_cache is not None:
            self._columns_cache.index = new_columns
        else:
            self._columns_cache = pandas.Series(index=new_columns)

    columns = property(_get_columns, _set_columns)
    index = property(_get_index, _set_index)

    # END Index, columns, and dtypes objects

    def compute_index(self, axis, data_object, compute_diff=True):
        """Computes the index after a number of rows have been removed.

        Note: In order for this to be used properly, the indexes must not be
            changed before you compute this.

        Args:
            axis: The axis to extract the index from.
            data_object: The new data object to extract the index from.
            compute_diff: True to use `self` to compute the index from self
                rather than data_object. This is used when the dimension of the
                index may have changed, but the deleted rows/columns are
                unknown

        Returns:
            A new pandas.Index object.
        """
        index_obj = self.index if not axis else self.columns
        old_blocks = self.data if compute_diff else None
        new_indices = data_object.get_indices(axis=axis, index_func=lambda df: pandas_index_extraction(df, axis), old_blocks=old_blocks)

        return index_obj[new_indices] if compute_diff else new_indices
    # END Index and columns objects

    # Internal methods
    # These methods are for building the correct answer in a modular way.
    # Please be careful when changing these!
    def _prepare_method(self, pandas_func, **kwargs):
        """Prepares methods given various metadata.

        :param pandas_func:
        :param kwargs:
        :return:
        """
        if self._is_transposed:
            return lambda df: pandas_func(df.T, **kwargs)
        else:
            return lambda df: pandas_func(df, **kwargs)
    # END Internal methods

    # Metadata modification methods
    def add_prefix(self, prefix):
        cls = type(self)
        new_column_names = self.columns.map(lambda x: str(prefix) + str(x))
        return cls(self.data, self.index, new_column_names, self.dtypes)

    def add_suffix(self, suffix):
        cls = type(self)
        new_column_names = self.columns.map(lambda x: str(x) + str(suffix))
        return cls(self.data, self.index, new_column_names, self.dtypes)
    # END Metadata modification methods

    # Copy
    # For copy, we don't want a situation where we modify the metadata of the
    # copies if we end up modifying something here. We copy all of the metadata
    # to prevent that.
    def copy(self):
        cls = type(self)
        return cls(self.data.copy(), self.index.copy(), self.columns.copy(), self.dtypes.copy())

    # Append/Concat/Join (Not Merge)
    # The append/concat/join operations should ideally never trigger remote
    # compute. These operations should only ever be manipulations of the
    # metadata of the resulting object. It should just be a simple matter of
    # appending the other object's blocks and adding np.nan columns for the new
    # columns, if needed. If new columns are added, some compute may be
    # required, though it can be delayed.
    #
    # Currently this computation is not delayed, and it may make a copy of the
    # DataFrame in memory. This can be problematic and should be fixed in the
    # future. TODO: Delay reindexing
    def _join_index_objects(self, axis, other_index, how, sort=True):
        """Joins a pair of index objects (columns or rows) by a given strategy.

        :param other_index:
        :param axis: The axis index object to join (0 for columns, 1 for index)
        :param how:
        :return:
        """
        if isinstance(other_index, list):
            joined_obj = self.columns if not axis else self.index
            # TODO: revisit for performance
            for obj in other_index:
                joined_obj = joined_obj.join(obj, how=how)

            return joined_obj
        if not axis:
            return self.columns.join(other_index, how=how, sort=sort)
        else:
            return self.index.join(other_index, how=how, sort=sort)

    def concat(self, axis, other, **kwargs):
        ignore_index = kwargs.get("ignore_index", False)
        if axis == 0:
            if isinstance(other, list):
                return self._append_list_of_managers(other, ignore_index)
            else:
                return self._append_data_manager(other, ignore_index)
        else:
            if isinstance(other, list):
                return self._join_list_of_managers(other, **kwargs)
            else:
                return self._join_data_manager(other, **kwargs)

    def _append_data_manager(self, other, ignore_index):
        assert isinstance(other, type(self)), \
            "This method is for data manager objects only"
        cls = type(self)

        joined_columns = self._join_index_objects(0, other.columns, 'outer')
        to_append = other.reindex(1, joined_columns).data
        new_self = self.reindex(1, joined_columns).data

        new_data = new_self.concat(0, to_append)
        new_index = self.index.append(other.index) if not ignore_index else pandas.RangeIndex(len(self.index) + len(other.index))

        return cls(new_data, new_index, joined_columns)

    def _append_list_of_managers(self, others, ignore_index):
        assert isinstance(others, list), \
            "This method is for lists of DataManager objects only"
        assert all(isinstance(other, type(self)) for other in others), \
            "Different Manager objects are being used. This is not allowed"
        cls = type(self)

        joined_columns = self._join_index_objects(0, [other.columns for other in others], 'outer')

        to_append = [other.reindex(1, joined_columns).data for other in others]
        new_self = self.reindex(1, joined_columns).data

        new_data = new_self.concat(0, to_append)
        new_index = self.index.append([other.index for other in others]) if not ignore_index else pandas.RangeIndex(len(self.index) + sum([len(other.index) for other in others]))

        return cls(new_data, new_index, joined_columns)

    def _join_data_manager(self, other, **kwargs):
        assert isinstance(other, type(self)), \
            "This method is for data manager objects only"
        cls = type(self)

        # Uses join's default value (though should not revert to default)
        how = kwargs.get("how", "left")
        sort = kwargs.get("sort", False)
        lsuffix = kwargs.get("lsuffix", "")
        rsuffix = kwargs.get("rsuffix", "")

        joined_index = self._join_index_objects(1, other.index, how, sort=sort)

        to_join = other.reindex(0, joined_index).data
        new_self = self.reindex(0, joined_index).data

        new_data = new_self.concat(1, to_join)

        # This stage is to efficiently get the resulting columns, including the
        # suffixes.
        self_proxy = pandas.DataFrame(columns=self.columns)
        other_proxy = pandas.DataFrame(columns=other.columns)
        new_columns = self_proxy.join(other_proxy, lsuffix=lsuffix, rsuffix=rsuffix).columns

        return cls(new_data, joined_index, new_columns)

    def _join_list_of_managers(self, others, **kwargs):
        assert isinstance(others, list), \
            "This method is for lists of DataManager objects only"
        assert all(isinstance(other, type(self)) for other in others), \
            "Different Manager objects are being used. This is not allowed"
        cls = type(self)

        # Uses join's default value (though should not revert to default)
        how = kwargs.get("how", "left")
        sort = kwargs.get("sort", False)
        lsuffix = kwargs.get("lsuffix", "")
        rsuffix = kwargs.get("rsuffix", "")

        assert isinstance(others, list), \
            "This method is for lists of DataManager objects only"
        assert all(isinstance(other, type(self)) for other in others), \
            "Different Manager objects are being used. This is not allowed"
        cls = type(self)

        joined_index = self._join_index_objects(1, [other.index for other in others], how, sort=sort)

        to_join = [other.reindex(0, joined_index).data for other in others]
        new_self = self.reindex(0, joined_index).data

        new_data = new_self.concat(1, to_join)

        # This stage is to efficiently get the resulting columns, including the
        # suffixes.
        self_proxy = pandas.DataFrame(columns=self.columns)
        others_proxy = [pandas.DataFrame(columns=other.columns) for other in others]
        new_columns = self_proxy.join(others_proxy, lsuffix=lsuffix, rsuffix=rsuffix).columns

        return cls(new_data, joined_index, new_columns)
    # END Append/Concat/Join

    # Inter-Data operations (e.g. add, sub)
    # These operations require two DataFrames and will change the shape of the
    # data if the index objects don't match. An outer join + op is performed,
    # such that columns/rows that don't have an index on the other DataFrame
    # result in NaN values.
    def inter_manager_operations(self, other, how_to_join, func):
        cls = type(self)

        assert isinstance(other, type(self)), \
            "Must have the same DataManager subclass to perform this operation"

        joined_index = self._join_index_objects(1, other.index, how_to_join, sort=False)
        new_columns = self._join_index_objects(0, other.columns, how_to_join, sort=False)

        reindexed_other = other.reindex(0, joined_index).data
        reindexed_self = self.reindex(0, joined_index).data

        # THere is an interesting serialization anomaly that happens if we do
        # not use the columns in `inter_data_op_builder` from here (e.g. if we
        # pass them in). Passing them in can cause problems, so we will just
        # use them from here.
        self_cols = self.columns
        other_cols = other.columns

        def inter_data_op_builder(left, right, self_cols, other_cols, func):
            left.columns = self_cols
            right.columns = other_cols
            result = func(left, right)
            result.columns = pandas.RangeIndex(len(result.columns))
            return result

        new_data = reindexed_self.inter_data_operation(1, lambda l, r: inter_data_op_builder(l, r, self_cols, other_cols, func), reindexed_other)

        return cls(new_data, joined_index, new_columns)

    def _inter_df_op_handler(self, func, other, **kwargs):
        """Helper method for inter-DataFrame and scalar operations"""
        axis = kwargs.get("axis", 0)

        if isinstance(other, type(self)):
            return self.inter_manager_operations(other, "outer", lambda x, y: func(x, y, **kwargs))
        else:
            return self.scalar_operations(axis, other, lambda df: func(df, other, **kwargs))

    def where(self, cond, other, **kwargs):
        cls = type(self)

        assert isinstance(cond, type(self)), \
            "Must have the same DataManager subclass to perform this operation"

        if isinstance(other, type(self)):
            # Note: Currently we are doing this with two maps across the entire
            # data. This can be done with a single map, but it will take a
            # modification in the `BlockPartition` class.
            # If this were in one pass it would be ~2x faster.
            # TODO rewrite this to take one pass.
            def where_builder_first_pass(cond, other, **kwargs):
                return cond.where(cond, other, **kwargs)

            def where_builder_second_pass(df, new_other, **kwargs):
                return df.where(new_other == True, new_other, **kwargs)

            # We are required to perform this reindexing on everything to
            # shuffle the data together
            reindexed_cond = cond.reindex(0, self.index).data
            reindexed_other = other.reindex(0, self.index).data
            reindexed_self = self.reindex(0, self.index).data

            first_pass = reindexed_cond.inter_data_operation(1, lambda l, r: where_builder_first_pass(l, r, **kwargs), reindexed_other)
            final_pass = reindexed_self.inter_data_operation(1, lambda l, r: where_builder_second_pass(l, r, **kwargs), first_pass)
            return cls(final_pass, self.index, self.columns)
        else:
            axis = kwargs.get("axis", 0)

            def where_builder_series(df, cond, other, **kwargs):
                return df.where(cond, other, **kwargs)

            reindexed_self = self.reindex(axis, self.index if not axis else self.columns).data
            reindexed_cond = cond.reindex(axis, self.index if not axis else self.columns).data

            new_data = reindexed_self.inter_data_operation(axis, lambda l, r: where_builder_series(l, r, other, **kwargs), reindexed_cond)
            return cls(new_data, self.index, self.columns)

    def update(self, other, **kwargs):
        assert isinstance(other, type(self)), \
            "Must have the same DataManager subclass to perform this operation"

        def update_builder(df, other, **kwargs):
            df.update(other, **kwargs)
            return df

        return self._inter_df_op_handler(update_builder, other, **kwargs)

    def add(self, other, **kwargs):
        #TODO: need to write a prepare_function for inter_df operations
        func = pandas.DataFrame.add
        return self._inter_df_op_handler(func, other, **kwargs)

    def div(self, other, **kwargs):
        func = pandas.DataFrame.div
        return self._inter_df_op_handler(func, other, **kwargs)

    def eq(self, other, **kwargs):
        func = pandas.DataFrame.eq
        return self._inter_df_op_handler(func, other, **kwargs)

    def floordiv(self, other, **kwargs):
        func = pandas.DataFrame.floordiv
        return self._inter_df_op_handler(func, other, **kwargs)

    def ge(self, other, **kwargs):
        func = pandas.DataFrame.ge
        return self._inter_df_op_handler(func, other, **kwargs)

    def gt(self, other, **kwargs):
        func = pandas.DataFrame.gt
        return self._inter_df_op_handler(func, other, **kwargs)

    def le(self, other, **kwargs):
        func = pandas.DataFrame.le
        return self._inter_df_op_handler(func, other, **kwargs)

    def lt(self, other, **kwargs):
        func = pandas.DataFrame.lt
        return self._inter_df_op_handler(func, other, **kwargs)

    def mod(self, other, **kwargs):
        func = pandas.DataFrame.mod
        return self._inter_df_op_handler(func, other, **kwargs)

    def mul(self, other, **kwargs):
        func = pandas.DataFrame.mul
        return self._inter_df_op_handler(func, other, **kwargs)

    def ne(self, other, **kwargs):
        func = pandas.DataFrame.ne
        return self._inter_df_op_handler(func, other, **kwargs)

    def pow(self, other, **kwargs):
        func = pandas.DataFrame.pow
        return self._inter_df_op_handler(func, other, **kwargs)

    def sub(self, other, **kwargs):
        func = pandas.DataFrame.sub
        return self._inter_df_op_handler(func, other, **kwargs)

    def truediv(self, other, **kwargs):
        func = pandas.DataFrame.truediv
        return self._inter_df_op_handler(func, other, **kwargs)

    # END Inter-Data operations

    # Single Manager scalar operations (e.g. add to scalar, list of scalars)
    def scalar_operations(self, axis, scalar, func):
        if isinstance(scalar, list):
            cls = type(self)
            new_data = self.map_across_full_axis(axis, func)
            return cls(new_data, self.index, self.columns)
        else:
            return self.map_partitions(func)
    # END Single Manager scalar operations

    # Reindex/reset_index (may shuffle data)
    def reindex(self, axis, labels, **kwargs):
        cls = type(self)

        # To reindex, we need a function that will be shipped to each of the
        # partitions.
        def reindex_builer(df, axis, old_labels, new_labels, **kwargs):
            if axis:
                df.columns = old_labels
                new_df = df.reindex(columns=new_labels, **kwargs)
                # reset the internal columns back to a RangeIndex
                new_df.columns = pandas.RangeIndex(len(new_df.columns))
                return new_df
            else:
                df.index = old_labels
                new_df = df.reindex(index=new_labels, **kwargs)
                # reset the internal index back to a RangeIndex
                new_df.reset_index(inplace=True, drop=True)
                return new_df

        old_labels = self.columns if axis else self.index

        new_index = self.index if axis else labels
        new_columns = labels if axis else self.columns

        func = self._prepare_method(lambda df: reindex_builer(df, axis, old_labels, labels, **kwargs))

        # The reindex can just be mapped over the axis we are modifying. This
        # is for simplicity in implementation. We specify num_splits here
        # because if we are repartitioning we should (in the future).
        # Additionally this operation is often followed by an operation that
        # assumes identical partitioning. Internally, we *may* change the
        # partitioning during a map across a full axis.
        new_data = self.map_across_full_axis(axis, func)
        return cls(new_data, new_index, new_columns)

    def reset_index(self, **kwargs):
        cls = type(self)
        drop = kwargs.get("drop", False)
        new_index = pandas.RangeIndex(len(self.index))

        if not drop:
            new_column_name = "index" if "index" not in self.columns else "level_0"
            new_columns = self.columns.insert(0, new_column_name)
            result = self.insert(0, new_column_name, self.index)
            return cls(result.data, new_index, new_columns)
        else:
            # The copies here are to ensure that we do not give references to
            # this object for the purposes of updates.
            return cls(self.data.copy(), new_index, self.columns.copy(), self.dtypes.copy())
    # END Reindex/reset_index

    # Transpose
    # For transpose, we aren't going to immediately copy everything. Since the
    # actual transpose operation is very fast, we will just do it before any
    # operation that gets called on the transposed data. See _prepare_method
    # for how the transpose is applied.
    #
    # Our invariants assume that the blocks are transposed, but not the
    # data inside. Sometimes we have to reverse this transposition of blocks
    # for simplicity of implementation.
    #
    # _is_transposed, 0 for False or non-transposed, 1 for True or transposed.
    _is_transposed = 0

    def transpose(self, *args, **kwargs):
        cls = type(self)
        new_data = self.data.transpose(*args, **kwargs)
        # Switch the index and columns and transpose the
        new_manager = cls(new_data, self.columns, self.index)
        # It is possible that this is already transposed
        new_manager._is_transposed = self._is_transposed ^ 1
        return new_manager
    # END Transpose

    # Full Reduce operations
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.
    def full_reduce(self, axis, map_func, reduce_func=None):
        if not axis:
            index = self.columns
        else:
            index = self.index

        if reduce_func is None:
            reduce_func = map_func

        # The XOR here will ensure that we reduce over the correct axis that
        # exists on the internal partitions. We flip the axis
        result = self.data.full_reduce(map_func, reduce_func, axis ^ self._is_transposed)
        result.index = index
        return result

    def count(self, **kwargs):
        axis = kwargs.get("axis", 0)
        map_func = self._prepare_method(pandas.DataFrame.count, **kwargs)
        reduce_func = self._prepare_method(pandas.DataFrame.sum, **kwargs)
        return self.full_reduce(axis, map_func, reduce_func)

    def max(self, **kwargs):
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(pandas.DataFrame.max, **kwargs)
        return self.full_reduce(axis, func)

    def mean(self, **kwargs):
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        length = len(self.index) if not axis else len(self.columns)

        return self.sum(**kwargs) / length

    def min(self, **kwargs):
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(pandas.DataFrame.min, **kwargs)
        return self.full_reduce(axis, func)

    def prod(self, **kwargs):
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(pandas.DataFrame.prod, **kwargs)
        return self.full_reduce(axis, func)

    def sum(self, **kwargs):
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(pandas.DataFrame.sum, **kwargs)
        return self.full_reduce(axis, func)
    # END Full Reduce operations

    # Map partitions operations
    # These operations are operations that apply a function to every partition.
    def map_partitions(self, func, new_dtypes=None):
        cls = type(self)
        return cls(self.data.map_across_blocks(func), self.index, self.columns, new_dtypes)

    def abs(self):
        func = self._prepare_method(pandas.DataFrame.abs)
        new_dtypes = pandas.Series([np.dtype('float64') for _ in self.columns], index=self.columns)
        return self.map_partitions(func, new_dtypes=new_dtypes)

    def applymap(self, func):
        remote_func = self._prepare_method(pandas.DataFrame.applymap, func=func)
        return self.map_partitions(remote_func)

    def isin(self, **kwargs):
        func = self._prepare_method(pandas.DataFrame.isin, **kwargs)
        new_dtypes = pandas.Series([np.dtype('bool') for _ in self.columns], index=self.columns)
        return self.map_partitions(func, new_dtypes=new_dtypes)

    def isna(self):
        func = self._prepare_method(pandas.DataFrame.isna)
        new_dtypes = pandas.Series([np.dtype('bool') for _ in self.columns], index=self.columns)
        return self.map_partitions(func, new_dtypes=new_dtypes)

    def isnull(self):
        func = self._prepare_method(pandas.DataFrame.isnull)
        new_dtypes = pandas.Series([np.dtype('bool') for _ in self.columns], index=self.columns)
        return self.map_partitions(func, new_dtypes=new_dtypes)

    def negative(self, **kwargs):
        func = self._prepare_method(pandas.DataFrame.__neg__, **kwargs)
        return self.map_partitions(func)

    def notna(self):
        func = self._prepare_method(pandas.DataFrame.notna)
        new_dtypes = pandas.Series([np.dtype('bool') for _ in self.columns], index=self.columns)
        return self.map_partitions(func, new_dtypes=new_dtypes)

    def notnull(self):
        func = self._prepare_method(pandas.DataFrame.notnull)
        new_dtypes = pandas.Series([np.dtype('bool') for _ in self.columns], index=self.columns)
        return self.map_partitions(func, new_dtypes=new_dtypes)

    def round(self, **kwargs):
        func = self._prepare_method(pandas.DataFrame.round, **kwargs)
        return self.map_partitions(func, new_dtypes=self.dtypes.copy())
    # END Map partitions operations

    # Column/Row partitions reduce operations
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.
    def full_axis_reduce(self, func, axis):
        result = self.data.map_across_full_axis(axis, func).to_pandas(self._is_transposed)

        if not axis:
            result.index = self.columns
        else:
            result.index = self.index

        return result

    def _post_process_idx_ops(self, axis, intermediate_result):
        index = self.index if not axis else self.columns
        result = intermediate_result.apply(lambda x: index[x])
        return result

    def all(self, **kwargs):
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(pandas.DataFrame.all, **kwargs)
        return self.full_axis_reduce(func, axis)

    def any(self, **kwargs):
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(pandas.DataFrame.any, **kwargs)
        return self.full_axis_reduce(func, axis)

    def idxmax(self, **kwargs):

        # The reason for the special treatment with idxmax/min is because we
        # need to communicate the row number back here.
        def idxmax_builder(df, **kwargs):
            df.index = pandas.RangeIndex(len(df.index))
            return df.idxmax(**kwargs)

        axis = kwargs.get("axis", 0)
        func = self._prepare_method(idxmax_builder, **kwargs)
        max_result = self.full_axis_reduce(func, axis)
        # Because our internal partitions don't track the external index, we
        # have to do a conversion.
        return self._post_process_idx_ops(axis, max_result)

    def idxmin(self, **kwargs):

        # The reason for the special treatment with idxmax/min is because we
        # need to communicate the row number back here.
        def idxmin_builder(df, **kwargs):
            df.index = pandas.RangeIndex(len(df.index))
            return df.idxmin(**kwargs)

        axis = kwargs.get("axis", 0)
        func = self._prepare_method(idxmin_builder, **kwargs)
        min_result = self.full_axis_reduce(func, axis)
        # Because our internal partitions don't track the external index, we
        # have to do a conversion.
        return self._post_process_idx_ops(axis, min_result)

    def info(self, **kwargs):
        def info_builder(df, **kwargs):
            result = pandas.DataFrame()
            if memory_usage:
                result['memory'] = df.memory_usage(index=False, deep=memory_usage_deep)
            if null_counts:
                result['count'] = df.count(axis=0)
            return result

        memory_usage = kwargs.get('memory_usage', True)
        null_counts = kwargs.get('null_counts', True)

        if type(memory_usage) == str and memory_usage == 'deep':
            memory_usage_deep = True
        else:
            memory_usage_deep = False

        func = self._prepare_method(info_builder, **kwargs)
        return self.full_axis_reduce(func, 0)

    def first_valid_index(self):

        # It may be possible to incrementally check each partition, but this
        # computation is fairly cheap.
        def first_valid_index_builder(df):
            df.index = pandas.RangeIndex(len(df.index))
            return df.apply(lambda df: df.first_valid_index())

        func = self._prepare_method(first_valid_index_builder)
        # We get the minimum from each column, then take the min of that to get
        # first_valid_index.
        first_result = self.full_axis_reduce(func, 0)

        return self.index[first_result.min()]

    def last_valid_index(self):

        def last_valid_index_builder(df):
            df.index = pandas.RangeIndex(len(df.index))
            return df.apply(lambda df: df.last_valid_index())

        func = self._prepare_method(last_valid_index_builder)
        # We get the maximum from each column, then take the max of that to get
        # last_valid_index.
        first_result = self.full_axis_reduce(func, 0)

        return self.index[first_result.max()]

    def median(self, **kwargs):
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(pandas.DataFrame.median, **kwargs)
        return self.full_axis_reduce(func, axis)

    def memory_usage(self, **kwargs):
        def memory_usage_builder(df, **kwargs):
            return df.memory_usage(index=False, deep=deep)

        deep = kwargs.get('deep', False)
        func = self._prepare_method(memory_usage_builder, **kwargs)
        return self.full_axis_reduce(func, 0)

    def nunique(self, **kwargs):
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(pandas.DataFrame.nunique, **kwargs)
        return self.full_axis_reduce(func, axis)

    def skew(self, **kwargs):
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(pandas.DataFrame.skew, **kwargs)
        return self.full_axis_reduce(func, axis)

    def std(self, **kwargs):
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(pandas.DataFrame.std, **kwargs)
        return self.full_axis_reduce(func, axis)

    def to_datetime(self, **kwargs):
        columns = self.columns
        def to_datetime_builder(df, **kwargs):
            df.columns = columns
            return pandas.to_datetime(df, **kwargs)
        func = self._prepare_method(to_datetime_builder, **kwargs)
        return self.full_axis_reduce(func, 1)

    def var(self, **kwargs):
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(pandas.DataFrame.var, **kwargs)
        return self.full_axis_reduce(func, axis)

    def quantile_for_single_value(self, **kwargs):
        axis = kwargs.get("axis", 0)
        q = kwargs.get("q", 0.5)
        assert type(q) is float

        func = self._prepare_method(pandas.DataFrame.quantile, **kwargs)

        result = self.full_axis_reduce(func, axis)
        result.name = q
        return result
    # END Column/Row partitions reduce operations

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.
    def map_across_full_axis(self, axis, func):
        return self.data.map_across_full_axis(axis, func)

    def query(self, expr, **kwargs):
        cls = type(self)
        columns = self.columns

        def query_builder(df, **kwargs):
            # This is required because of an Arrow limitation
            # TODO revisit for Arrow error
            df = df.copy()
            df.index = pandas.RangeIndex(len(df))
            df.columns = columns
            df.query(expr, inplace=True, **kwargs)
            df.columns = pandas.RangeIndex(len(df.columns))
            return df

        func = self._prepare_method(query_builder, **kwargs)
        new_data = self.map_across_full_axis(1, func)
        # Query removes rows, so we need to update the index
        new_index = self.compute_index(0, new_data, True)

        return cls(new_data, new_index, self.columns, self.dtypes)

    def eval(self, expr, **kwargs):
        cls = type(self)
        columns = self.columns

        def eval_builder(df, **kwargs):
            df.columns = columns
            result = df.eval(expr, inplace=False, **kwargs)
            # If result is a series, expr was not an assignment expression.
            if not isinstance(result, pandas.Series):
                result.columns = pandas.RangeIndex(0, len(result.columns))
            return result

        func = self._prepare_method(eval_builder, **kwargs)
        new_data = self.map_across_full_axis(1, func)

        # eval can update the columns, so we must update columns
        columns_copy = pandas.DataFrame(columns=columns)
        columns_copy = columns_copy.eval(expr, inplace=False, **kwargs)
        if isinstance(columns_copy, pandas.Series):
            # To create a data manager, we need the 
            # columns to be in a list-like
            columns = list(columns_copy.name)
        else:
            columns = columns_copy.columns

        return cls(new_data, self.index, columns)

    def quantile_for_list_of_values(self, **kwargs):
        cls = type(self)
        axis = kwargs.get("axis", 0)
        q = kwargs.get("q", 0.5)
        assert isinstance(q, (pandas.Series, np.ndarray, pandas.Index, list))

        func = self._prepare_method(pandas.DataFrame.quantile, **kwargs)

        q_index = pandas.Float64Index(q)

        new_data = self.map_across_full_axis(axis, func)
        new_columns = self.columns if not axis else self.index
        return cls(new_data, q_index, new_columns)

    def _cumulative_builder(self, func, **kwargs):
        cls = type(self)
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(func, **kwargs)
        new_data = self.map_across_full_axis(axis, func)
        return cls(new_data, self.index, self.columns, self.dtypes)

    def cumsum(self, **kwargs):
        return self._cumulative_builder(pandas.DataFrame.cumsum, **kwargs)

    def cummax(self, **kwargs):
        return self._cumulative_builder(pandas.DataFrame.cummax, **kwargs)

    def cummin(self, **kwargs):
        return self._cumulative_builder(pandas.DataFrame.cummin, **kwargs)

    def cumprod(self, **kwargs):
        return self._cumulative_builder(pandas.DataFrame.cumprod, **kwargs)

    def dropna(self, **kwargs):
        axis = kwargs.get("axis", 0)
        subset = kwargs.get("subset")
        thresh = kwargs.get("thresh")
        how = kwargs.get("how", "any")
        # We need to subset the axis that we care about with `subset`. This
        # will be used to determine the number of values that are NA.
        if subset is not None:
            if not axis:
                compute_na = self.getitem_column_array(subset)
            else:
                compute_na = self.getitem_row_array(subset)
        else:
            compute_na = self

        if not isinstance(axis, list):
            axis = [axis]
        # We are building this dictionary first to determine which columns
        # and rows to drop. This way we do not drop some columns before we
        # know which rows need to be dropped.
        if thresh is not None:
            # Count the number of NA values and specify which are higher than
            # thresh.
            drop_values = {ax ^ 1: compute_na.isna().sum(axis=ax ^ 1) > thresh for ax in axis}
        else:
            drop_values = {ax ^ 1: getattr(compute_na.isna(), how)(axis=ax ^ 1) for ax in axis}

        if 0 not in drop_values:
            drop_values[0] = None

        if 1 not in drop_values:
            drop_values[1] = None

            rm_from_index = [obj for obj in compute_na.index[drop_values[1]]] if drop_values[1] is not None else None
            rm_from_columns = [obj for obj in compute_na.columns[drop_values[0]]] if drop_values[0] is not None else None
        else:
            rm_from_index = compute_na.index[drop_values[1]] if drop_values[1] is not None else None
            rm_from_columns = compute_na.columns[drop_values[0]] if drop_values[0] is not None else None

        return self.drop(index=rm_from_index, columns=rm_from_columns)

    def mode(self, **kwargs):
        cls = type(self)

        axis = kwargs.get("axis", 0)
        func = self._prepare_method(pandas.DataFrame.mode, **kwargs)
        new_data = self.map_across_full_axis(axis, func)

        counts = cls(new_data, self.index, self.columns).notnull().sum(axis=axis)
        max_count = counts.max()

        new_index = pandas.RangeIndex(max_count) if not axis else self.index
        new_columns = self.columns if not axis else pandas.RangeIndex(max_count)

        # We have to reindex the DataFrame so that all of the partitions are
        # matching in shape. The next steps ensure this happens.
        final_labels = new_index if not axis else new_columns
        # We build these intermediate objects to avoid depending directly on
        # the underlying implementation.
        final_data = cls(new_data, new_index, new_columns).map_across_full_axis(axis, lambda df: df.reindex(axis=axis, labels=final_labels))
        return cls(final_data, new_index, new_columns, self.dtypes)

    def fillna(self, **kwargs):
        cls = type(self)

        axis = kwargs.get("axis", 0)
        value = kwargs.pop("value")

        if isinstance(value, dict):
            if axis == 0:
                index = self.columns
            else:
                index = self.index
            value = {idx: value[key] for key in value for idx in index.get_indexer_for([key])}

            def fillna_dict_builder(df, func_dict={}):
                return df.fillna(value=func_dict, **kwargs)

            new_data = self.data.apply_func_to_select_indices(axis, fillna_dict_builder, value, keep_remaining=True)
            return cls(new_data, self.index, self.columns)
        else:
            func = self._prepare_method(pandas.DataFrame.fillna, **kwargs)
            new_data = self.map_across_full_axis(axis, func)
            return cls(new_data, self.index, self.columns)

    def describe(self, **kwargs):
        cls = type(self)
        axis = 0

        func = self._prepare_method(pandas.DataFrame.describe, **kwargs)
        new_data = self.map_across_full_axis(axis, func)
        new_index = self.compute_index(0, new_data, False)
        new_columns = self.compute_index(1, new_data, True)
        new_dtypes = pandas.Series([np.float64 for _ in new_columns], index=new_columns)

        return cls(new_data, new_index, new_columns, new_dtypes)

    def rank(self, **kwargs):
        cls = type(self)

        axis = kwargs.get("axis", 0)

        func = self._prepare_method(pandas.DataFrame.rank, **kwargs)
        new_data = self.map_across_full_axis(axis, func)

        # Since we assume no knowledge of internal state, we get the columns
        # from the internal partitions.
        if kwargs.get("numeric_only", False):
            new_columns = self.compute_index(1, new_data, True)
        else:
            new_columns = self.columns
        new_dtypes = pandas.Series([np.float64 for _ in new_columns], index=new_columns)
        return cls(new_data, self.index, new_columns, new_dtypes)

    def diff(self, **kwargs):
        cls = type(self)

        axis = kwargs.get("axis", 0)

        func = self._prepare_method(pandas.DataFrame.diff, **kwargs)
        new_data = self.map_across_full_axis(axis, func)

        return cls(new_data, self.index, self.columns)
    # END Map across rows/columns

    # Head/Tail/Front/Back
    def head(self, n):
        cls = type(self)
        # We grab the front if it is transposed and flag as transposed so that
        # we are not physically updating the data from this manager. This
        # allows the implementation to stay modular and reduces data copying.
        if self._is_transposed:
            # Transpose the blocks back to their original orientation first to
            # ensure that we extract the correct data on each node. The index
            # on a transposed manager is already set to the correct value, so
            # we need to only take the head of that instead of re-transposing.
            result = cls(self.data.transpose().take(1, n).transpose(), self.index[:n], self.columns, self.dtypes)
            result._is_transposed = True
        else:
            result = cls(self.data.take(0, n), self.index[:n], self.columns, self.dtypes)
        return result

    def tail(self, n):
        cls = type(self)
        # See head for an explanation of the transposed behavior
        if self._is_transposed:
            result = cls(self.data.transpose().take(1, -n).transpose(), self.index[-n:], self.columns, self.dtypes)
            result._is_transposed = True
        else:
            result = cls(self.data.take(0, -n), self.index[-n:], self.columns, self.dtypes)
        return result

    def front(self, n):
        cls = type(self)
        # See head for an explanation of the transposed behavior
        if self._is_transposed:
            result = cls(self.data.transpose().take(0, n).transpose(), self.index, self.columns[:n], self.dtypes[:n])
            result._is_transposed = True
        else:
            result = cls(self.data.take(1, n), self.index, self.columns[:n], self.dtypes[:n])
        return result

    def back(self, n):
        cls = type(self)
        # See head for an explanation of the transposed behavior
        if self._is_transposed:
            result = cls(self.data.transpose().take(0, -n).transpose(), self.index, self.columns[-n:], self.dtypes[-n:])
            result._is_transposed = True
        else:
            result = cls(self.data.take(1, -n), self.index, self.columns[-n:], self.dtypes[-n:])
        return result
    # End Head/Tail/Front/Back

    # Data Management Methods
    def free(self):
        """In the future, this will hopefully trigger a cleanup of this object.
        """
        # TODO create a way to clean up this object.
        return
    # END Data Management Methods

    # To/From Pandas
    def to_pandas(self):
        df = self.data.to_pandas(is_transposed=self._is_transposed)
        df.index = self.index
        df.columns = self.columns
        return df

    @classmethod
    def from_pandas(cls, df, block_partitions_cls):
        new_index = df.index
        new_columns = df.columns
        new_dtypes = df.dtypes

        # Set the columns to RangeIndex for memory efficiency
        df.index = pandas.RangeIndex(len(df.index))
        df.columns = pandas.RangeIndex(len(df.columns))
        new_data = block_partitions_cls.from_pandas(df)

        return cls(new_data, new_index, new_columns, dtypes=new_dtypes)

    # __getitem__ methods
    def getitem_single_key(self, key):
        numeric_index = self.columns.get_indexer_for([key])

        new_data = self.getitem_column_array([key])
        if len(numeric_index) > 1:
            return new_data
        else:
            # This is the case that we are returning a single Series.
            # We do this post processing because everything is treated a a list
            # from here on, and that will result in a DataFrame.
            return new_data.to_pandas()[key]

    def getitem_column_array(self, key):
        cls = type(self)
        # Convert to list for type checking
        numeric_indices = list(self.columns.get_indexer_for(key))

        # Internal indices is left blank and the internal
        # `apply_func_to_select_indices` will do the conversion and pass it in.
        def getitem(df, internal_indices=[]):
            return df.iloc[:, internal_indices]

        result = self.data.apply_func_to_select_indices(0, getitem, numeric_indices, keep_remaining=False)

        # We can't just set the columns to key here because there may be
        # multiple instances of a key.
        new_columns = self.columns[numeric_indices]
        new_dtypes = self.dtypes[numeric_indices]
        return cls(result, self.index, new_columns, new_dtypes)

    def getitem_row_array(self, key):
        cls = type(self)
        # Convert to list for type checking
        numeric_indices = list(self.index.get_indexer_for(key))

        def getitem(df, internal_indices=[]):
            return df.iloc[internal_indices]

        result = self.data.apply_func_to_select_indices(1, getitem, numeric_indices, keep_remaining=False)
        # We can't just set the index to key here because there may be multiple
        # instances of a key.
        new_index = self.index[numeric_indices]
        return cls(result, new_index, self.columns, self.dtypes)
    # END __getitem__ methods

    # __delitem__ and drop
    # These will change the shape of the resulting data.
    def delitem(self, key):
        return self.drop(columns=[key])

    def drop(self, index=None, columns=None):
        cls = type(self)

        if index is None:
            new_data = self.data
            new_index = self.index
        else:
            def delitem(df, internal_indices=[]):
                return df.drop(index=df.index[internal_indices])

            numeric_indices = list(self.index.get_indexer_for(index))
            new_data = self.data.apply_func_to_select_indices(1, delitem, numeric_indices, keep_remaining=True)
            # We can't use self.index.drop with duplicate keys because in Pandas
            # it throws an error.
            new_index = [self.index[i] for i in range(len(self.index)) if i not in numeric_indices]

        if columns is None:
            new_columns = self.columns
            new_dtypes = self.dtypes
        else:
            def delitem(df, internal_indices=[]):
                return df.drop(columns=df.columns[internal_indices])

            numeric_indices = list(self.columns.get_indexer_for(columns))
            new_data = new_data.apply_func_to_select_indices(0, delitem, numeric_indices, keep_remaining=True)
            # We can't use self.columns.drop with duplicate keys because in Pandas
            # it throws an error.
            new_columns = [self.columns[i] for i in range(len(self.columns)) if i not in numeric_indices]
            new_dtypes = self.dtypes.drop(columns)
        return cls(new_data, new_index, new_columns, new_dtypes)
    # END __delitem__ and drop

    # Insert
    # This method changes the shape of the resulting data. In Pandas, this
    # operation is always inplace, but this object is immutable, so we just
    # return a new one from here and let the front end handle the inplace
    # update.
    def insert(self, loc, column, value):
        cls = type(self)

        def insert(df, internal_indices=[]):
            internal_idx = internal_indices[0]
            df.insert(internal_idx, internal_idx, value, allow_duplicates=True)
            return df

        new_data = self.data.apply_func_to_select_indices_along_full_axis(0, insert, loc, keep_remaining=True)
        new_columns = self.columns.insert(loc, column)

	# Because a Pandas Series does not allow insert, we make a DataFrame
        # and insert the new dtype that way.
        temp_dtypes = pandas.DataFrame(self.dtypes).T
        temp_dtypes.insert(loc, column, _get_dtype_from_object(value))
        new_dtypes = temp_dtypes.iloc[0]

        return cls(new_data, self.index, new_columns, new_dtypes)
    # END Insert

    # astype
    # This method changes the types of select columns to the new dtype.
    def astype(self, col_dtypes, errors='raise', **kwargs):
        cls = type(self)

        # Group the indicies to update together and create new dtypes series
        dtype_indices = dict()
        columns = col_dtypes.keys()
        new_dtypes = self.dtypes.copy()

        numeric_indices = list(self.columns.get_indexer_for(columns))

        for i, column in enumerate(columns):
            dtype = col_dtypes[column]
            if dtype != self.dtypes[column]:
                if dtype in dtype_indices.keys():
                    dtype_indices[dtype].append(numeric_indices[i])
                else:
                    dtype_indices[dtype] = [numeric_indices[i]]
                new_dtype = np.dtype(dtype)
                if dtype != np.int32 and new_dtype == np.int32:
                    new_dtype = np.dtype('int64')
                elif dtype != np.float32 and new_dtype == np.float32:
                    new_dtype = np.dtype('float64')
                new_dtypes[column] = new_dtype

        new_data = self.data
        for dtype in dtype_indices.keys():
            resulting_dtype = None

            def astype(df, internal_indices=[]):
                block_dtypes = dict()
                for ind in internal_indices:
                    block_dtypes[df.columns[ind]]= dtype
                return df.astype(block_dtypes)

            new_data = self.data.apply_func_to_select_indices(0, astype, dtype_indices[dtype], keep_remaining=True)

        return cls(new_data, self.index, self.columns, new_dtypes)
    # END type conversions

    # UDF (apply and agg) methods
    # There is a wide range of behaviors that are supported, so a lot of the
    # logic can get a bit convoluted.
    def apply(self, func, axis, *args, **kwargs):
        if callable(func):
            return self._callable_func(func, axis, *args, **kwargs)
        elif isinstance(func, dict):
            return self._dict_func(func, axis, *args, **kwargs)
        elif is_list_like(func):
            return self._list_like_func(func, axis, *args, **kwargs)
        else:
            pass

    def _post_process_apply(self, result_data, axis):
        cls = type(self)
        try:
            index = self.compute_index(0, result_data, True)
        except IndexError:
            index = self.compute_index(0, result_data, False)
        try:
            columns = self.compute_index(1, result_data, True)
        except IndexError:
            columns = self.compute_index(1, result_data, False)

        # `apply` and `aggregate` can return a Series or a DataFrame object,
        # and since we need to handle each of those differently, we have to add
        # this logic here.
        if len(columns) == 0:
            series_result = result_data.to_pandas(False)
            if not axis and len(series_result) == len(self.columns) and len(index) != len(series_result):
                index = self.columns
            elif axis and len(series_result) == len(self.index) and len(index) != len(series_result):
                index = self.index

            series_result.index = index
            return series_result

        return cls(result_data, index, columns)

    def _dict_func(self, func, axis, *args, **kwargs):
        if "axis" not in kwargs:
            kwargs["axis"] = axis

        if axis == 0:
            index = self.columns
        else:
            index = self.index

        func = {idx: func[key] for key in func for idx in index.get_indexer_for([key])}

        def dict_apply_builder(df, func_dict={}):
            return df.apply(func_dict, *args, **kwargs)

        result_data = self.data.apply_func_to_select_indices_along_full_axis(axis, dict_apply_builder, func, keep_remaining=False)

        full_result = self._post_process_apply(result_data, axis)

        # The columns can get weird because we did not broadcast them to the
        # partitions and we do not have any guarantee that they are correct
        # until here. Fortunately, the keys of the function will tell us what
        # the columns are.
        if isinstance(full_result, pandas.Series):
            full_result.index = [self.columns[idx] for idx in func]
        return full_result

    def _list_like_func(self, func, axis, *args, **kwargs):
        cls = type(self)
        func_prepared = self._prepare_method(lambda df: df.apply(func, *args, **kwargs))
        new_data = self.map_across_full_axis(axis, func_prepared)

        # When the function is list-like, the function names become the index
        new_index = [f if isinstance(f, string_types) else f.__name__ for f in func]
        return cls(new_data, new_index, self.columns)

    def _callable_func(self, func, axis, *args, **kwargs):

        def callable_apply_builder(df, func, axis, index, *args, **kwargs):
            if not axis:
                df.index = index
                df.columns = pandas.RangeIndex(len(df.columns))
            else:
                df.columns = index
                df.index = pandas.RangeIndex(len(df.index))

            result = df.apply(func, axis=axis, *args, **kwargs)
            return result

        index = self.index if not axis else self.columns

        func_prepared = self._prepare_method(lambda df: callable_apply_builder(df, func, axis, index, *args, **kwargs))
        result_data = self.map_across_full_axis(axis, func_prepared)
        return self._post_process_apply(result_data, axis)
    # END UDF

    # Manual Partitioning methods (e.g. merge, groupby)
    # These methods require some sort of manual partitioning due to their
    # nature. They require certain data to exist on the same partition, and
    # after the shuffle, there should be only a local map required.
    def _manual_repartition(self, axis, repartition_func, **kwargs):
        """This method applies all manual partitioning functions.

        :param axis:
        :param repartition_func:

        Returns:
            A `BlockPartitions` object.
        """
        func = self._prepare_method(repartition_func, **kwargs)
        return self.data.manual_shuffle(axis, func)


class RayPandasDataManager(PandasDataManager):

    @classmethod
    def _from_old_block_partitions(cls, blocks, index, columns):
        blocks = np.array([[RayRemotePartition(obj) for obj in row] for row in blocks])
        return PandasDataManager(RayBlockPartitions(blocks), index, columns)


def pandas_index_extraction(df, axis):
    if not axis:
        return df.index
    else:
        try:
            return df.columns
        except AttributeError:
            return pandas.Index([])
