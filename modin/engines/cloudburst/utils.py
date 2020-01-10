import os
from droplet.client.client import DropletConnection

droplet = None


def get_or_init_client():
    global droplet
    if droplet is None:
        ip = os.eviron.get("MODIN_IP", None)
        conn = os.environ.get("MODIN_CONNECTION", None)
        droplet = DropletConnection(conn, ip)
    return droplet
