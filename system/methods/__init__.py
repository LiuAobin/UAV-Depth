from .sc_depth_v1 import SCDepthV1

method_maps = {
    'sc-depth': SCDepthV1,
}

__all__ = [
    'method_maps',
    'SCDepthV1',
]