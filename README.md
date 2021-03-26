
# Learning Calibrated-Guidance for Object Detection in Aerial Images
Paper can be seen here [https://arxiv.org/abs/2103.11399](https://arxiv.org/abs/2103.11399)


****

|Backbone|+Ours|Weight|mAP(%)
|---|---|---|---|---
|ResNet-50|||73.26
|ResNet-50|+||74.21
|ResNet-100|||73.06
|ResNet-100|+||74.30
|ResNet-152|||72.78
|ResNet-152|+||73.53

## Introduction
This codebase is created to build benchmarks for object detection in aerial images.
It is modified from [mmdetection](https://github.com/open-mmlab/mmdetection).
The master branch works with **PyTorch 1.1** or higher. If you would like to use PyTorch 0.4.1,
please checkout to the [pytorch-0.4.1](https://github.com/open-mmlab/mmdetection/tree/pytorch-0.4.1) branch.

   
## License

This project is released under the [Apache 2.0 license](LICENSE).

- You can find the detailed configs in configs/DOTA.
- The trained models are available at [Google Drive](https://drive.google.com/drive/folders/1UhCU2H-kx4oSnN4eepGIxVz9i30eMslu?usp=sharing).
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
