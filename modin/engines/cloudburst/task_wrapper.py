from modin import __execution_engine__

if __execution_engine__ == "Cloudburst":
    from modin.engines.cloudburst.utils import get_or_init_client
    droplet = get_or_init_client()


class CloudburstTask:
    @classmethod
    def deploy(cls, func, num_return_vals, kwargs):
        f = droplet.register(lambda _, kwargs: func(**kwargs), func.__name__)
        future_obj = f(kwargs)
        unpack = droplet.register(lambda _, l, i: l[i], "unpack")
        return [unpack(future_obj, i) for i in range(num_return_vals)]

    @classmethod
    def materialize(cls, obj_id):
        return droplet.get_object(obj_id)
