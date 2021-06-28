# [Demosaicing by Differentiable Deep Restoration](https://www.mdpi.com/2076-3417/11/4/1649).


## Introduction

This is the pytorch implementation of our paper.

## Dependency
```
PyTorch 1.8
```

## Setup
Compile functions for PSC layer:
```
cd exts
python setup.py install
```

## Dataset
Please download 
[MIT](https://groups.csail.mit.edu/graphics/demosaicnet/dataset.html),
[Kodak](http://www.cs.albany.edu/~xypan/research/snr/Kodak.html),
and 
[Mcm](https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm)
dataset.
The structure of data directory:
```
└── datas
    └── color
        ├── test
        │   ├── filelist.txt
        │   ├── hdrvdp
        │   ├── kodak
        │   ├── mcm
        │   └── moire
        ├── train
        │   ├── check.py
        │   ├── filelist.txt
        │   ├── hdrvdp
        │   └── moire
        └── val
            ├── hdrvdp
            └── moire
```
Then pack images into lmdb files.
```
python create_lmdb.py
```

## Configs
The config of different settings:
- DB.yaml (Demosaicing for Bayer CFA Pattern)
- DL.yaml (Demosaicing for 4x4 Learned CFA Pattern)
- DLN.yaml (Demosaicing for 4x4 Learned CFA Pattern with Noisy Data)



## Trained Models
You can directly download the model I trained:
- [DB](https://drive.google.com/file/d/1JCmaw82ubO8kRjf9sisgxUyB-qYvdDcJ/view?usp=sharing)
- [DL](https://drive.google.com/file/d/1AJ1AcY-KvJx3F8Jrpq-E-7IW_jGunrHq/view?usp=sharing)
- [DLN](https://drive.google.com/file/d/17-qa3EmrKjet-e3qTMlCAXkQn16EVfjU/view?usp=sharing)


## Train 
You can also train by yourself:
```
python train.py
```
*Pay attention to the settings in the config file (e.g. gpu id).*

## Test
With the trained model, 
you can test and save demosaiced results.
```
python test.py
```

## Citation
If you find this work useful in your research, please consider citing:
```
@article{D3R,
author = {Tang, Jie and Li, Jian and Tan, Ping},
title = {Demosaicing by Differentiable Deep Restoration},
journal = {Applied Sciences},
volume = {11},
year = {2021},
number = {4},
article-number = {1649},
}
```