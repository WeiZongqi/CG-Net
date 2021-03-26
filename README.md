
# Learning Calibrated-Guidance for Object Detection in Aerial Images [arxiv](https://arxiv.org/abs/2103.11399)

We propose a simple yet effective Calibrated-Guidance (CG) scheme to enhance channel communications in a feature transformer fashion, which can adaptively determine the calibration weights for each channel based on the global feature affinity-pairs. Specifically, given a set of feature maps, CG first computes the feature similarity between each channel and the remaining channels as the intermediary calibration guidance. Then, re-representing each channel by aggregating all the channels weighted together via the guidance. Our CG can be plugged into any deep neural network, which is named as CG-Net. To demonstrate its effectiveness and efficiency, extensive experiments are carried out on both oriented and horizontal object detection tasks of aerial images. Results on two challenging benchmarks (\ie, DOTA and HRSC2016) demonstrate that our CG-Net can achieve state-of-the-art performance in accuracy with a fair computational overhead.

****

## Introduction
This codebase is created to build benchmarks for object detection in aerial images.
It is modified from [mmdetection](https://github.com/open-mmlab/mmdetection).
The master branch works with **PyTorch 1.1** or higher. If you would like to use PyTorch 0.4.1,
please checkout to the [pytorch-0.4.1](https://github.com/open-mmlab/mmdetection/tree/pytorch-0.4.1) branch.

## Results
(/show/all.png)

(/show/compare.png)

## Experiment

ImageNet Pretrained Model from Pytorch
- [ResNet50](https://drive.google.com/file/d/1mQ9S0FzFpPHnocktH0DGVysufGt4tH0M/view?usp=sharing)
- [ResNet101](https://drive.google.com/file/d/1qlVf58T0fY4dddKst5i7-CL3DXhBi3Mp/view?usp=sharing)
- [ResNet152](https://drive.google.com/file/d/1y08s30DdWUyaFU89vEpospMi8TjqrJIz/view?usp=sharing)

The effectiveness of our proposed methods with different backbone network on the test of DOTA.
|Backbone|+CG|Weight|mAP(%)|
|:---:|:---:|:---:|:---:|
|ResNet-50|||73.26|
|ResNet-50|+||74.21|
|ResNet-100|||73.06|
|ResNet-100|+||74.30|
|ResNet-152|||72.78|
|ResNet-152|+||73.53|

CG-Net Results in DOTA.
|Backbone|Aug Rotate|Task|Weight|mAP(%)|
|:---:|:---:|:---:|:---:|:---:|
|ResNet-100|+|Oriented||77.89|
|ResNet-100|+|Horizontal||78.26|

## Installation

Please refer to [INSTALL.md](INSTALL.md) for installation.

    
## Get Started

Please see [GETTING_STARTED.md](GETTING_STARTED.md) for the basic usage of mmdetection.

## Contributing

We appreciate all contributions to improve benchmarks for object detection in aerial images. 


## Citing

If you use our work, please consider citing:

```
@InProceedings{liang2021learning,
      title={Learning Calibrated-Guidance for Object Detection in Aerial Images}, 
      author={Dong, Liang and Zongqi, Wei and Dong, Zhang and Qixiang, Geng and Liyan, Zhang and Han, Sun and Huiyu, Zhou and Mingqiang, Wei and Pan, Gao},
      booktitle ={arXiv:2103.11399},
      year={2021}
}
```

## Thanks to the Third Party Libs

[Pytorch](https://pytorch.org/)

[mmdetection](https://github.com/open-mmlab/mmdetection)

[AerialDetection](https://github.com/dingjiansw101/AerialDetection)
