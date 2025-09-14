# Think Locally, Act Globally: A Frequency Spatial Fusion Network for Infrared Small Target Detection [[Paper]](https://ieeexplore.ieee.org/document/10486932)

Weijie Xu, Zhenglong Ding, Ziheng Wang, Zhiqing Cui, Yifan Hu, and Feng Jiang, IEEE Transactions on Geoscience and Remote Sensing 2025.

# If the implementation of this repo is helpful to you, just star it！⭐⭐⭐

# Chanlleges and inspiration   
![Image text](https://github.com/qzxwj/FSCFNet/blob/main/figures/challenges.png)

# Structure
![Image text](https://github.com/qzxwj/FSCFNet/blob/main/figures/FSCFNet.png)

![Image text](https://github.com/qzxwj/FSCFNet/blob/main/figures/FSConv.png)

![Image text](https://github.com/qzxwj/FSCFNet/blob/main/figures/ACA.png)

![Image text](https://github.com/qzxwj/FSCFNet/blob/main/figures/MRCB.png)


# Introduction

We present a Frequency Spatial Fusion Network (FSCFNet) to the IRSTD task. Experiments on both public (e.g., IRSTD-1K, NUDT-SIRST, NUAA-SIRST) demonstrate the effectiveness of our method. Our main contributions are as follows:

1. The novel plug-and-play convolution module FSConv is designed, which integrates a DWT decomposer to capture both local details and global structural information in the additional frequency domain, preserving spatial characteristics at the same time.

2. The new cross-attention-based mechanism ACA is proposed to facilitate the feature fusion by focusing on the local central regions of IRST, effectively strengthening their salient spatial characteristics.

3. Inspired by Inception Architecture, we propose the customized MRCB to effectively capture long-range contextual dependencies using multi-scale dilated convolutions.

4. Extensive experiments across multiple datasets validate that FSCFNet maintains a lightweight design, while significantly improving both the accuracy and robustness of IRSTD.


## Usage

#### 1. Data

The **IRSTD-1K**, **NUDT-SIRST**, and **NUAA-SIRST** datasets are used to train FSCFNet.
* **IRSTD-1K** &nbsp; [[download dir]](https://github.com/RuiZhang97/ISNet) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9880295)
* **NUDT-SIRST** &nbsp; [[download]](https://github.com/YeRen123455/Infrared-Small-Target-Detection) &nbsp; [[paper]](https://ieeexplore.ieee.org/abstract/document/9864119)
* **NUAA-SIRST** &nbsp; [[download]]() &nbsp; [[paper]]()

* **Our project has the following structure:**
  ```
  ├──./datasets/
  │    ├── IRSTD-1K
  │    │    ├── images
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── labels
  │    │    │    ├── XDU0.txt
  │    │    │    ├── XDU1.txt
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_IRSTD-1K.txt
  │    │    │    ├── test_IRSTD-1K.txt
  │    ├── NUDT-SIRST
  │    │    ├── images
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── labels
  │    │    │    ├── 000001.txt
  │    │    │    ├── 000002.txt
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_NUDT-SIRST.txt
  │    │    │    ├── test_NUDT-SIRST.txt
  │    ├── NUAA-SIRST
  │    │    ├── images
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  │    │    │    ├── ...
  │    │    ├── labels
  │    │    │    ├── Misc_1.txt
  │    │    │    ├── Misc_2.txt
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_NUAA-SIRST.txt
  │    │    │    ├── test_NUAA-SIRST.txt
  ```


##### 2. Train.
```bash
python train.py
```

## Results and Trained Models

#### Qualitative Results
![Image text](https://github.com/qzxwj/FSCFNet/blob/main/figures/FSCFNetResult.png)

*The overall repository style is highly borrowed from [ultralytics](https://github.com/ultralytics/ultralytics). Thanks to ultralytics.

## Citation

If you find the code useful, please consider citing our paper using the following BibTeX entry.

```
@ARTICLE{10486932,
  author={Yuan, Shuai and Qin, Hanlin and Yan, Xiang and Akhtar, Naveed and Mian, Ajmal},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SCTransNet: Spatial-Channel Cross Transformer Network for Infrared Small Target Detection}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
  keywords={Semantics;Transformers;Decoding;Feature extraction;Task analysis;Object detection;Visualization;Convolutional neural network (CNN);cross-attention;deep learning;infrared small target detection (IRSTD);transformer},
  doi={10.1109/TGRS.2024.3383649}}
```


## Contact
**Welcome to raise issues or email to [wjxu@nuist.edu.cn](wjxu@nuist.edu.cn) for any question regarding our FSCFNet.**