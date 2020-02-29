from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
import cloudpickle
import threading
import os
import uuid
import dask
from concurrent.futures import Future
from concurrent.futures._base import PENDING


class LockDict(object):
    def __init__(self):
        self._locks = {}

    def __iter__(self):
        return self._locks.__iter__()

    def __contains__(self, item):
        return item in self._locks

    def __getstate__(self):
        return {}

    def __setstate__(self, item):
        self.__init__()

    def __setitem__(self, item, value):
        self._locks[item] = value

    def __getitem__(self, item):
        return self._locks[item]


class PickleLock(object):
    _locks = LockDict()

    def __init__(self, key=None):
        self.key = key or str(uuid.uuid4())

    _lock = None

    @property
    def lock(self):
        if self._lock is None:
            if self.key in PickleLock._locks:
                self._lock = PickleLock._locks[self.key]
            else:
                self._lock = threading.Lock()
                PickleLock._locks[self.key] = self._lock
        return self._lock

    def acquire(self, *args, **kwargs):
        return self.lock.acquire(*args, **kwargs)

    def release(self, *args, **kwargs):
        return self.lock.release(*args, **kwargs)

    def locked(self):
        return self.lock.locked()

    def __getattr__(self, item):
        return getattr(self.lock, item)

    def __getstate__(self):
        return self.key

    def __setstate__(self, key):
        self.__init__(key)


def _future_init(self):
    self._condition = threading.Condition(dask.utils.SerializableLock())
    self._state = PENDING
    self._result = None
    self._exception = None
    self._waiters = []
    self._done_callbacks = []


Future.__init__ = _future_init
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads)


def _get_global_executor():
    import multiprocessing
    executor = MPIPoolExecutor(main=False)
    return executor


__all__ = ["_get_global_executor"]