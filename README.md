# MiSLAS
**Improving Calibration for Long-Tailed Recognition**

**Authors**: Zhisheng Zhong, Jiequan Cui, Shu Liu, Jiaya Jia

[[`arXiv`](https://arxiv.org/pdf/2104.00466.pdf)] [[`slide`]]() [[`BibTeX`](#Citation)]

<div align="center">
  <img src="./assets/MiSLAS.PNG" style="zoom:90%;"/>
</div><br/>

**Introduction**: This repository provides an implementation for the CVPR 2021 paper: "[Improving Calibration for Long-Tailed Recognition](https://arxiv.org/pdf/2104.00466.pdf)" based on [LDAM-DRW](https://github.com/kaidic/LDAM-DRW) and [Decoupling models](https://github.com/facebookresearch/classifier-balancing). *Our study shows, because of the extreme imbalanced composition ratio of each class, networks trained on long-tailed datasets are more miscalibrated and over-confident*. MiSLAS is a simple, and efficient two-stage framework for long-tailed recognition, which improves recognition accuracy and relieves over-confidence simultaneously.


## Installation

**Requirements**

* Python 3.7
* torchvision 0.4.0
* Pytorch 1.2.0
* yacs 0.1.8

**Virtual Environment**
```
conda create -n MiSLAS python==3.7
source activate MiSLAS
```

**Install MiSLAS**
```
git clone https://github.com/Jia-Research-Lab/MiSLAS.git --recursive 
cd MiSLAS
pip install -r requirements.txt
```

**Dataset Preparation**
* [ImageNet_LT](http://image-net.org/index)
* [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018)
* [Places_LT](http://places2.csail.mit.edu/download.html)

Change the `data_path` in `config/*/*.yaml` accordingly.

## Training

**Stage-1**:

To train a model for Stage-1 with *mixup*, run:

```
python train_stage1.py --cfg ./config/DATASETNAME/DATASETNAME_ARCH_stage1_mixup.yaml
```

`DATASETNAME` can be selected from `cifar10`,  `cifar100`, `imagenet`, `ina2018`, and `places`.

`ARCH` can be `resnet32` for `cifar10/100`, `resnet50/101/152` for `imagenet`, `resnet50` for `ina2018`, and `resnet152` for `places`, respectively.

**Stage-2**:

To train a model for Stage-2 with *one GPU*, run:

```
python train_stage2.py --cfg ./config/DATASETNAME/DATASETNAME_ARCH_stage2_METHOD.yaml resume /path/to/checkpoint/stage1
```
`METHOD` can be selected from `cRT`,  `LWS`, and `MiSLAS`.

## Evaluation

To evaluate a pre-trained model, run:

```
python eval.py --cfg ./config/DATASETNAME/DATASETNAME_ARCH_stage2_METHOD.yaml resume /path/to/checkpoint/stage2
```

## Results and Models

**1) CIFAR-10-LT and CIFAR-100-LT**

* Stage-1:

| Dataset              | Top-1 Accuracy | ECE  | Model |
| -------------------- | -------------- | ---- | ----- |
| CIFAR-10-LT   IF=10  |                |      |       |
| CIFAR-10-LT   IF=50  |                |      |       |
| CIFAR-10-LT   IF=100 |                |      |       |
| CIFAR-100-LT IF=10   |                |      |       |
| CIFAR-100-LT IF=50   |                |      |       |
| CIFAR-100-LT IF=100  |                |      |       |

* Stage-2:

| Dataset              | Top-1 Accuracy | ECE  | Model |
| -------------------- | -------------- | ---- | ----- |
| CIFAR-10-LT   IF=10  |                |      |       |
| CIFAR-10-LT   IF=50  |                |      |       |
| CIFAR-10-LT   IF=100 |                |      |       |
| CIFAR-100-LT IF=10   |                |      |       |
| CIFAR-100-LT IF=50   |                |      |       |
| CIFAR-100-LT IF=100  |                |      |       |

*Note: To obtain better performance, we highly recommend changing the weight decay 2e-4 to 5e-4 on CIFAR-LT.*

**2) Large-scale Datasets**

* Stage-1:

| Dataset     | Arch       | Top-1 Accuracy | ECE  | Model |
| ----------- | ---------- | -------------- | ---- | ----- |
| ImageNet-LT | ResNet-50  |                |      |       |
| iNa'2018    | ResNet-50  |                |      |       |
| Places-LT   | ResNet-152 |                |      |       |

* Stage-2:

| Dataset     | Arch       | Top-1 Accuracy | ECE  | Model |
| ----------- | ---------- | -------------- | ---- | ----- |
| ImageNet-LT | ResNet-50  |                |      |       |
| iNa'2018    | ResNet-50  |                |      |       |
| Places-LT   | ResNet-152 |                |      |       |

## <a name="Citation"></a>Citation

Please consider citing MiSLAS in your publications if it helps your research. :)

```bib
@inproceedings{zhong2021mislas,
    title={Improving Calibration for Long-Tailed Recognition},
    author={Zhisheng Zhong, Jiequan Cui, Shu Liu, and Jiaya Jia},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021},
}
```

## Contact

If you have any questions about our work, feel free to contact us through email (Zhisheng Zhong: zszhong@pku.edu.cn) or Github issues.
