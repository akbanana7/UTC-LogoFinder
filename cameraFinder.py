# Basic script to find and return camera indexes for the system
from pygrabber.dshow_graph import FilterGraph
devices = FilterGraph().get_input_devices()
def findCamera():
    toRet = []
    for index, name in enumerate(devices):
        toRet.append(f"Index {index}: {name}")
    return toRet