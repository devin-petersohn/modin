from threading import Thread
from typing import Tuple, Dict, Any, Optional

import xgboost as xgb
import os
import pandas

if os.environ.get("MODIN_ENGINE", "Ray").title() == "Ray":
    import ray
    from ray.services import get_node_ip_address
    import modin.pandas as pd

else:
    raise ValueError("Ray must be set as MODIN_ENGINE")


def _assert_modin_engine_ray():
    assert (
        os.environ.get("MODIN_ENGINE", "Ray").title() == "Ray"
    ), "MODIN_ENGINE environment variable must be set to Ray"


def _start_rabit_tracker(num_workers: int):
    """Start Rabit tracker. The workers connect to this tracker to share
    their results."""
    host = get_node_ip_address()

    env = {"DMLC_NUM_WORKER": num_workers}
    rabit_tracker = xgb.RabitTracker(hostIP=host, nslave=num_workers)

    # Get tracker Host + IP
    env.update(rabit_tracker.slave_envs())
    rabit_tracker.start(num_workers)

    # Wait until context completion
    thread = Thread(target=rabit_tracker.join)
    thread.daemon = True
    thread.start()

    return env


class RabitContext:
    """Context to connect a worker to a rabit tracker"""

    def __init__(self, actor_id, args):
        self.args = args
        self.args.append(("DMLC_TASK_ID=[xgboost.ray]:" + actor_id).encode())

    def __enter__(self):
        xgb.rabit.init(self.args)

    def __exit__(self, *args):
        xgb.rabit.finalize()


@ray.remote
class ModinXGBoostActor:
    def __init__(self):
        self._dtrain = []
        self._evals = []

    def set_X_y(self, *X, y):
        X = pandas.concat(list(X), axis=1)
        self._dtrain = xgb.DMatrix(X, y)

    def add_eval_X_y(self, *X, eval_method):
        if len(X) > 1:
            X = pandas.concat(list(X), axis=1)
        self._evals.append((X, eval_method))

    def train(self, rabit_args, params, *args, **kwargs):
        local_params = params.copy()
        local_dtrain = self._dtrain
        local_evals = self._evals

        evals_result = dict()

        with RabitContext(str(id(self)), rabit_args):
            bst = xgb.train(
                local_params,
                local_dtrain,
                *args,
                evals=local_evals,
                evals_result=evals_result,
                **kwargs
            )
            return {"bst": bst, "evals_result": evals_result}


class ModinDMatrix(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __iter__(self):
        yield self.X
        yield self.y



def train(
    params: Dict,
    data: ModinDMatrix,
    *args,
    evals=(),
    num_actors: Optional[int] = None,
    gpus_per_worker: Optional[int] = None,
    **kwargs
):
    X, y = data
    assert len(X) == len(y)
    left_frame = X._query_compiler._modin_frame
    right_frame = y._query_compiler._modin_frame
    assert (
        left_frame._partitions.shape[0] == right_frame._partitions.shape[0]
    ), "Unaligned train data"
    if len(evals):
        for eval in evals:
            eval_X, eval_y = eval[0]
            left_eval_frame = eval_X._query_compiler._modin_frame
            right_eval_frame = eval_y._query_compiler._modin_frame
            assert (
                left_eval_frame._partitions.shape[0]
                == right_eval_frame._partitions.shape[0]
            ), "Unaligned test data"
    if num_actors is None:
        num_actors = left_frame._partitions.shape[0]

    # Create remote actors
    if gpus_per_worker is not None:
        actors = [
            ModinXGBoostActor.options(num_gpus=gpus_per_worker).remote()
            for _ in range(num_actors)
        ]
    else:
        actors = [ModinXGBoostActor.remote() for _ in range(num_actors)]

    # Split data across workers
    for i, actor in enumerate(actors):
        actor.set_X_y.remote(
            *[part.oid for part in left_frame._partitions[i]],
            y=right_frame._partitions[i][0].oid
        )
        for _, ((eval_X, eval_y), eval_method) in enumerate(evals):
            actor.add_eval_X_y.remote(
                *[part.oid for part in eval_X._query_compiler._modin_frame._partitions[i]],
                y=eval_y._query_compiler._modin_frame.partitions[i][0].oid,
                eval_method=eval_method
            )

    # Start tracker
    env = _start_rabit_tracker(num_actors)
    rabit_args = [("%s=%s" % item).encode() for item in env.items()]

    # Train
    fut = [actor.train.remote(rabit_args, params, *args, **kwargs) for actor in actors]

    # All results should be the same because of Rabit tracking. So we just
    # return the first one.
    res: Dict[str, Any] = ray.get(fut[0])
    bst = res["bst"]
    evals_result = res["evals_result"]

    return bst, evals_result
