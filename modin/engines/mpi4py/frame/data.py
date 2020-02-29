from modin.engines.base.frame.data import BasePandasFrame
from .partition_manager import MPIFrameManager
from modin import __execution_engine__

if __execution_engine__ == "MPI":
    from modin.engines.mpi4py import _get_global_executor


class PandasOnMPIFrame(BasePandasFrame):

    _frame_mgr_cls = MPIFrameManager

    @property
    def _row_lengths(self):
        """Compute the row lengths if they are not cached.

        Returns:
            A list of row lengths.
        """
        if self._row_lengths_cache is None:
            self._row_lengths_cache = [f.result() for f in
                [obj.apply(lambda df: len(df)).future for obj in self._partitions.T[0]]
            ]
        return self._row_lengths_cache

    @property
    def _column_widths(self):
        """Compute the column widths if they are not cached.

        Returns:
            A list of column widths.
        """
        if self._column_widths_cache is None:
            self._column_widths_cache = [f.result() for f in
                [
                    obj.apply(lambda df: len(df.columns)).future
                    for obj in self._partitions[0]
                ]
            ]
        return self._column_widths_cache
