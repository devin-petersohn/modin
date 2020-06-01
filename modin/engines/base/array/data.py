class ModinArray(object):

    _frame_mgr_cls = None

    def __init__(self, partitions, row_lengths=None, col_widths=None):
        self._partitions = partitions
        self._row_lengths = row_lengths
        self._col_widths = col_widths

    @property
    def __constructor__(self):
        return type(self)

    def reduction_to_scalar(self, func):
        map_parts = self._frame_mgr_cls.map_partitions(self._partitions, func)
        reduce_parts = self._frame_mgr_cls.map_axis_partitions(0, map_parts, func)
        final_parts = self._frame_mgr_cls.map_axis_partitions(1, reduce_parts, func)
        return self.__constructor__(final_parts, [1], [1])

    def map(self, func):
        return self.__constructor__(self._frame_mgr_cls.map_partitions(self._partitions, func), self._row_lengths, self._col_widths)

    def to_numpy(self):
        return self._frame_mgr_cls.to_numpy(self._partitions)
