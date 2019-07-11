import os
from fluent.functions.client import FluentConnection


class Connection:
    class __Connection:
        def __init__(self, address, ip):
            self.conn = FluentConnection(address, ip)

        def get(self):
            return self.conn

    instance = None

    def __init__(self, address, ip):
        if not Connection.instance:
            Connection.instance = Connection.__Connection(address, ip)

    def get(self):
        return Connection.instance.get()


conn = Connection(os.environ["FLUENT_ADDR"], os.environ["FLUENT_IP"])


def get_conn():
    return conn.get()


def apply_call_queue(call_queue, future):
    # Use *call_queue here and pre-register the functions
    # Add kwargs as a separate argument and zip the two
    df = future
    for func, kwargs in call_queue:
        df = func(df, **kwargs)
    return df


remote_apply = get_conn().register(apply_call_queue, "apply_call_queue")
