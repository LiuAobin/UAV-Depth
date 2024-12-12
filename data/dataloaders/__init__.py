from .kitti_raw_loader import KittiRawLoader
from .cityscapes_loader import CityscapesLoader
from .base_loader import BaseLoader

dataloader_map = {
    'kitti_raw': KittiRawLoader,
    'cityscapes': CityscapesLoader,
}
__all__ = ['KittiRawLoader',
           'CityscapesLoader',
           'base_loader',
           'dataloader_map']