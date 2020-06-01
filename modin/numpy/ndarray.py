class ndarray(object):

    def __init__(self, query_compiler):
        self._query_compiler = query_compiler

    @property
    def __constructor__(self):
        return type(self)

    def sum(self):
        return self._query_compiler.sum()

    def __repr__(self):
        return repr(self._query_compiler.to_numpy())

    def add(self, scalar):
        return self.__constructor__(query_compiler=self._query_compiler.add(scalar))

    def __add__(self, other):
        return self.add(other)

    @classmethod
    def from_modin_frame(cls, modin_frame):
        return cls(modin_frame._query_compiler.to_array_compiler())
