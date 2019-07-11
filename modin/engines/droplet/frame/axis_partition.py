from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fluent.functions.include.shared import FluentFuture

from modin.engines.base.frame.axis_partition import PandasFrameAxisPartition
from .partition import PandasOnDropletFramePartition


class PandasOnDropletFrameAxisPartition(PandasFrameAxisPartition):
    def __init__(self, list_of_blocks):
        # Unwrap from BaseFramePartition object for ease of use
        for obj in list_of_blocks:
            obj.drain_call_queue()
        self.list_of_blocks = [obj.future for obj in list_of_blocks]

    partition_type = PandasOnDropletFramePartition
    instance_type = FluentFuture


class PandasOnDropletFrameColumnPartition(PandasOnDropletFrameAxisPartition):
    """The column partition implementation for Ray. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 0


class PandasOnDropletFrameRowPartition(PandasOnDropletFrameAxisPartition):
    """The row partition implementation for Ray. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 1
