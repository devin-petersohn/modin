from modin.engines.base.frame.axis_partition import PandasFrameAxisPartition
from .partition import PandasOnMPIFramePartition
from modin import __execution_engine__

if __execution_engine__ == "MPI":
    from modin.engines.mpi4py import _get_global_executor
    from concurrent.futures import Future


class PandasOnMPIFrameAxisPartition(PandasFrameAxisPartition):
    def __init__(self, list_of_blocks):
        # Unwrap from BaseFramePartition object for ease of use
        for obj in list_of_blocks:
            obj.drain_call_queue()
        self.list_of_blocks = [obj.future for obj in list_of_blocks]

    partition_type = PandasOnMPIFramePartition
    if __execution_engine__ == "MPI":
        instance_type = Future

    @classmethod
    def deploy_axis_func(
        cls, axis, func, num_splits, kwargs, maintain_partitioning, *partitions
    ):
        client = _get_global_executor()
        axis_result = client.submit(
            PandasFrameAxisPartition.deploy_axis_func,
            axis,
            func,
            num_splits,
            kwargs,
            maintain_partitioning,
            *partitions,
            
        )
        if num_splits == 1:
            return axis_result
        # We have to do this to split it back up. It is already split, but we need to
        # get futures for each.
        return [
            client.submit(lambda l: l[i], axis_result)
            for i in range(num_splits)
        ]

    @classmethod
    def deploy_func_between_two_axis_partitions(
        cls, axis, func, num_splits, len_of_left, kwargs, *partitions
    ):
        client = _get_global_executor()
        axis_result = client.submit(
            PandasFrameAxisPartition.deploy_func_between_two_axis_partitions,
            axis,
            func,
            num_splits,
            len_of_left,
            kwargs,
            *partitions,
            
        )
        if num_splits == 1:
            return axis_result
        # We have to do this to split it back up. It is already split, but we need to
        # get futures for each.
        return [
            client.submit(lambda l: l[i], axis_result)
            for i in range(num_splits)
        ]


class PandasOnMPIFrameColumnPartition(PandasOnMPIFrameAxisPartition):
    """The column partition implementation for Multiprocess. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 0


class PandasOnMPIFrameRowPartition(PandasOnMPIFrameAxisPartition):
    """The row partition implementation for Multiprocess. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 1
