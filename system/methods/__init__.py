from .sc_depth_v1 import SCDepthV1
from .darknet import DarkNet
from .sql_depth import SQLDepth
method_maps = {
    'sc-depth': SCDepthV1,
    'darknet': DarkNet,
    'sql-depth': SQLDepth,
}

__all__ = [
    'method_maps',
    'SCDepthV1',
    'DarkNet',
    'SQLDepth'
]