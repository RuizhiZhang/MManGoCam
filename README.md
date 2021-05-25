
**Notice**: The tool is still in developing. The framework is not fully well-organized. It's a good warm-up from 3 example .ipynb files, even there are somewhat equiped tools in 2 main_xxx.py. Have fun! 

## Introduction

The neural network feature map drawing tool is supposed to be developed to explain the attentions of deep neural network, especially at step of feature selection. The GradCam is a well-performed solution at present. The realization of GradCam used in this tool was enlightened by https://github.com/yizt/Grad-CAM.pytorch, although the backbone was extremely reconstructed. The tool shown on this page was developed particularly for object detection models for MMDetection. The later is an open source object detection toolbox based on PyTorch. It is a part of the OpenMMLab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

## Installation

Same with MMdetection. Please refer to [get_started.md](docs/get_started.md) for installation.

## Getting Started

First, you should enable the MMdetection. The customer data are supposed to deplyed, and the config file will be prepared. Please see [get_started.md](docs/get_started.md) for the basic usage of MMDetection. After the model training or model deployment, you should expect work_dir/ with your pth and config file. Both components are required for implementation of the GradCam drawing. Then, you could either start with the example_xxx.ipynb or main_xxx.py. 

## MMDetection Modification

The neccessary modifications were made in MMDetection framework. In order to saving training time, MMDet detached gradients generated which blocked backward propagation for the gradients hook. The changes declaimed in source code as dstinct comments. Also, since the network framework varies with respect to algorithm, the realization of the GradCam class get different, the FRCNN-r101, FRCNN-HRNET, and CascadedRCNN were designated seperately at present. There might be enhancement in terms of the design pattern in future plan.

## Citation

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
[Grad-CAM:Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391)  
