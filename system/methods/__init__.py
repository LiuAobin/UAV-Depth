from .sc_depth_v1 import SCDepthV1
from .darknet import DarkNet
method_maps = {
    'sc-depth': SCDepthV1,
    'darknet': DarkNet,
}

__all__ = [
    'method_maps',
    'SCDepthV1',
    'DarkNet',
]