# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import os
import sys
import warnings
from packaging import version

from ._version import get_versions


def custom_formatwarning(msg, category, *args, **kwargs):
    # ignore everything except the message
    return "{}: {}\n".format(category.__name__, msg)


warnings.formatwarning = custom_formatwarning
# Filter numpy version warnings because they are not relevant
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="Large object of size")
warnings.filterwarnings(
    "ignore",
    message="The pandas.datetime class is deprecated and will be removed from pandas in a future version. "
    "Import from datetime module instead.",
)
warnings.filterwarnings(
    "ignore",
    message="pandas.core.index is deprecated and will be removed in a future version. "
    "The public classes are available in the top-level namespace.",
)


def get_execution_engine():
    # In the future, when there are multiple engines and different ways of
    # backing the DataFrame, there will have to be some changed logic here to
    # decide these things. In the meantime, we will use the currently supported
    # execution engine + backing (Pandas + Ray).
    if "MODIN_ENGINE" in os.environ:
        # .title allows variants like ray, RAY, Ray
        return os.environ["MODIN_ENGINE"].title()
    else:
        if "MODIN_DEBUG" in os.environ:
            return "Python"
        else:
            if sys.platform != "win32":
                try:
                    import ray

                except ImportError:
                    pass
                else:
                    if version.parse(ray.__version__) != version.parse("0.8.3"):
                        raise ImportError(
                            "Please `pip install modin[ray]` to install compatible Ray version."
                        )
                    return "Ray"
            try:
                import dask
                import distributed

            except ImportError:
                raise ImportError(
                    "Please `pip install {}modin[dask]` to install an engine".format(
                        "modin[ray]` or `" if sys.platform != "win32" else ""
                    )
                )
            else:
                if version.parse(dask.__version__) < version.parse(
                    "2.1.0"
                ) or version.parse(distributed.__version__) < version.parse("2.3.2"):
                    raise ImportError(
                        "Please `pip install modin[dask]` to install compatible Dask version."
                    )
                return "Dask"


def get_partition_format():
    # See note above about engine + backing.
    return os.environ.get("MODIN_BACKEND", "Pandas").title()


__version__ = "0.6.3"
__execution_engine__ = get_execution_engine()
__partition_format__ = get_partition_format()

# We don't want these used outside of this file.
del get_execution_engine
del get_partition_format

__version__ = get_versions()["version"]
del get_versions


from IPython.core.magic import line_cell_magic, magics_class, needs_local_scope
from IPython.core.magics.execution import ExecutionMagics
from IPython import get_ipython
import re
import ast
if sys.version_info > (3,8):
    from ast import Module
else:
    from ast import Module as OriginalModule
    Module = lambda nodelist, type_ignores: OriginalModule(nodelist)


_clusters = {}
ipython = get_ipython()


def register_cluster(name, ip, *args):
    if name in _clusters:
        teardown_cluster(name)
    ip_match = re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", ip)
    if ip_match is None:
        raise ValueError("Invalid Cluster IP")
    _clusters[name] = ip


def teardown_cluster(name):
    del _clusters[name]


@magics_class
class ModinMagics(ExecutionMagics):
    @line_cell_magic
    @needs_local_scope
    def deploy_modin_to(self, line, cell=None, local_ns=None):
        first_arg = line.split(" ")[0]
        line = " ".join(line.split(" ")[1:])
        if first_arg not in _clusters:
            raise ValueError("No such cluster: {}".format(first_arg))
        print("Distributing {} to {}".format(line, first_arg))
        from distributed import Client, get_client
        original_client = get_client()
        Client(_clusters[first_arg])

        if cell:
            expr = self.shell.transform_cell(cell)
        else:
            expr = self.shell.transform_cell(line)

        expr_ast = self.shell.compile.ast_parse(expr)
        # Apply AST transformations
        expr_ast = self.shell.transform_ast(expr_ast)

        expr_val = None
        if len(expr_ast.body) == 1 and isinstance(expr_ast.body[0], ast.Expr):
            mode = 'eval'
            source = '<distributed eval>'
            expr_ast = ast.Expression(expr_ast.body[0].value)
        else:
            mode = 'exec'
            source = '<distributed exec>'
            # multi-line %%time case
            if len(expr_ast.body) > 1 and isinstance(expr_ast.body[-1], ast.Expr):
                expr_val = expr_ast.body[-1]
                expr_ast = expr_ast.body[:-1]
                expr_ast = Module(expr_ast, [])
                expr_val = ast.Expression(expr_val.value)

        code = self.shell.compile(expr_ast, source, mode)

        # skew measurement as little as possible
        glob = self.shell.user_ns
        if mode == 'eval':
            try:
                out = eval(code, glob, local_ns)
            except:
                self.shell.showtraceback()
                return
        else:
            try:
                exec(code, glob, local_ns)
                out = None
                if expr_val is not None:
                    code_2 = self.shell.compile(expr_val, source, 'eval')
                    out = eval(code_2, glob, local_ns)
            except:
                self.shell.showtraceback()
                return
        Client(original_client.scheduler.addr)
        return out

try:
    ipython.register_magics(ModinMagics)
except:
    pass
