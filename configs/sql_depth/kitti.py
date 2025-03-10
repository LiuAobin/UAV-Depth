

dataset_name = 'kitti'
# dataset_dir = 'E:/Depth/target_dataset/kitti'
dataset_dir = '/data/coding/kitti'
folder_type = 'sequence'
load_pseudo_depth = False

exp_name = 'sql-depth-kitti'
val_model = 'depth'
use_frame_index = False
height = 320
width = 1024
batch_size=32
epoch_size=1300
img_suffix='*.jpg'
depth_suffix='*.npz'
# model params
method = 'sql-depth'
num_layers=50
model_dim=32
num_features=512
query_nums=64
dim_out=64
patch_size=32
