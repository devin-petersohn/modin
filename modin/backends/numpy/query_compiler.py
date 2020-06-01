class NumpyQueryCompiler(object):

    def __init__(self, modin_array):
        self._modin_array = modin_array

    @property
    def __constructor__(self):
        return type(self)

    def sum(self):
        return self.__constructor__(
            self._modin_array.reduction_to_scalar(lambda x: x.sum())
        )

    def add(self, scalar):
        return self.__constructor__(self._modin_array.map(lambda x: x + scalar))

    def to_numpy(self):
        return self._modin_array.to_numpy()
