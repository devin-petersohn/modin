from modin import __execution_engine__

if __execution_engine__ == "MPI":
    from modin.engines.mpi4py import _get_global_executor


class MPITask:
    @classmethod
    def deploy(cls, func, num_return_vals, kwargs):
        client = _get_global_executor()
        remote_task_future = client.submit(func, **kwargs)

        def f(l, i):
            return l.result()[i]

        return [
            client.submit(f, remote_task_future, i)
            for i in range(num_return_vals)
        ]

    @classmethod
    def materialize(cls, future):
        if isinstance(future, list):
            return [f.result() for f in future]
        return future.result()
