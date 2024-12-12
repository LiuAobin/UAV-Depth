# 1 KITTI数据集
> 按照[Zhou]划分的训练集和[Eigen]划分的测试集，全部所需的场景为/config/kitti/train_scenes.txt所述
> 修改后的KITTI下载脚本为/scripts/kitti_raw_data_downloader.sh (已移除校园和人群场景和校准场景)

# 2 cityscapes数据集
> 按照[Zhou]和[Bian]划分的方式，仅需要从[官网](https://www.cityscapes-dataset.com/downloads/)下载以下三个文件
> 1. [leftImg8bit_sequence_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=14) (324GB)
> 2. [camera_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=8) (2MB)
> 3. [vehicle_sequence.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=20) (56MB)

# 3 DDAD数据集


# 4 Make3D数据集

# 5 Visual KITTI数据集

# 6 参考
    @INPROCEEDINGS{8100183,
        author={Zhou, Tinghui and Brown, Matthew and Snavely, Noah and Lowe, David G.},
        booktitle={2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
        title={Unsupervised Learning of Depth and Ego-Motion from Video}, 
        year={2017},
        volume={},
        number={},
        pages={6612-6619},
        keywords={Cameras;Training;Pose estimation;Three-dimensional displays;Geometry;Pipelines},
        doi={10.1109/CVPR.2017.700}
    }
    @inproceedings{10.5555/2969033.2969091,
        author = {Eigen, David and Puhrsch, Christian and Fergus, Rob}, 
        title = {Depth map prediction from a single image using a multi-scale deep network}, 
        year = {2014}, 
        publisher = {MIT Press},
        address = {Cambridge, MA, USA},
        abstract = {Predicting depth is an essential component in understanding the 3D geometry of a scene. While for stereo images local correspondence suffices for estimation, finding depth relations from a single image is less straightforward, requiring integration of both global and local information from various cues. Moreover, the task is inherently ambiguous, with a large source of uncertainty coming from the overall scale. In this paper, we present a new method that addresses this task by employing two deep network stacks: one that makes a coarse global prediction based on the entire image, and another that refines this prediction locally. We also apply a scale-invariant error to help measure depth relations rather than scale. By leveraging the raw datasets as large sources of training data, our method achieves state-of-the-art results on both NYU Depth and KITTI, and matches detailed depth boundaries without the need for superpixelation.},
        booktitle = {Proceedings of the 27th International Conference on Neural Information Processing Systems - Volume 2},
        pages = {2366–2374},
        numpages = {9},
        location = {Montreal, Canada},
        series = {NIPS'14}
    } 
    @article{bian2021ijcv, 
        title={Unsupervised Scale-consistent Depth Learning from Video}, 
        author={Bian, Jia-Wang and Zhan, Huangying and Wang, Naiyan and Li, Zhichao and Zhang, Le and Shen, Chunhua and Cheng, Ming-Ming and Reid, Ian}, 
        journal= {International Journal of Computer Vision (IJCV)}, 
        year={2021} 
    }
 