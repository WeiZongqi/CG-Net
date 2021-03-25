
# Benchmarks for Object Detection in Aerial Images

## Introduction
This codebase is created to build benchmarks for object detection in aerial images.
It is modified from [mmdetection](https://github.com/open-mmlab/mmdetection).
The master branch works with **PyTorch 1.1** or higher. If you would like to use PyTorch 0.4.1,
please checkout to the [pytorch-0.4.1](https://github.com/open-mmlab/mmdetection/tree/pytorch-0.4.1) branch.

   
## License

This project is released under the [Apache 2.0 license](LICENSE).

- You can find the detailed configs in configs/DOTA.
- The trained models are available at [Google Drive](https://drive.google.com/drive/folders/1ChLKP16Z_QReTYWGivZ2OXXuITyKNw0r).
## Installation


  Please refer to [INSTALL.md](INSTALL.md) for installation.


    
## Get Started

Please see [GETTING_STARTED.md](GETTING_STARTED.md) for the basic usage of mmdetection.

## Contributing

We appreciate all contributions to improve benchmarks for object detection in aerial images. 


## Citing

If you use [DOTA](https://captain-whu.github.io/DOTA/) dataset, codebase or models in your research, please consider cite .

```
@inproceedings{xia2018dota,
  title={DOTA: A large-scale dataset for object detection in aerial images},
  author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3974--3983},
  year={2018}
}

@article{chen2019mmdetection,
  title={MMDetection: Open mmlab detection toolbox and benchmark},
  author={Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Xu, Jiarui and others},
  journal={arXiv preprint arXiv:1906.07155},
  year={2019}
}

@misc{liang2021cgnet,
title= {Learning Calibrated-Guidance for Object Detection in Aerial Images},
author = {Dong Liang and Zongqi Wei and Dong Zhang and Qixiang Geng and Liyan Zhang and Han Sun and Huiyu Zhou and Mingqiang Wei and Pan Gao},
booktitle= {arXiv:2103.11399},
year = {2021}
}
```

## Thanks to the Third Party Libs

[Pytorch](https://pytorch.org/)

[mmdetection](https://github.com/open-mmlab/mmdetection)

[AerialDetection](https://github.com/dingjiansw101/AerialDetection)