from modin.pandas.dataframe import _DATAFRAME_EXTENSIONS_, DataFrame
from modin.pandas.series import _SERIES_EXTENSIONS_, Series
from modin.pandas import _PD_EXTENSIONS_
import modin.pandas


def _register_accessor(name, extensions_dict, obj):
    def decorator(accessor):
        extensions_dict[name] = accessor
        setattr(obj, name, accessor)
        return accessor

    return decorator


def register_dataframe_accessor(name):
    return _register_accessor(name, _DATAFRAME_EXTENSIONS_, DataFrame)


def register_series_accessor(name):
    return _register_accessor(name, _SERIES_EXTENSIONS_, Series)


def register_pd_accessor(name):
    return _register_accessor(name, _PD_EXTENSIONS_, modin.pandas)
