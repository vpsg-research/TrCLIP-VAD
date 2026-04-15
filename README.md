# TrCLIP-VAD: Weak supervised video anomaly detection by improving CLIP training with text rewriting
## Overview
![alt text](https://github.com/ssjlyh/TrCLIP-VAD/blob/main/data/framework.png)
The overview of the proposed TrCLIP-VAD. It consists of several components, i.e. visual branch, text branch, C-branch (coarse-grained branch) and F-branch (fine-grained branch).
## Highlight

- We propose a novel TrCLIP-VAD framework, which enhances textual feature diversity through text rewriting strategy. As far as we know, TrCLIP-VAD is the first work to introduce an image-text dual feature enhancement approach to WSVAD.

- We design an LGM-Mamba module that innovatively integrates the abilities of local perception, global modeling and multi-scale analysis, enabling efficient and comprehensive temporal feature learning for VAD.

- TrCLIP-VAD achieves state-of-the-art (SOTA) performance on two widely-used datasets. Specifically, it achieves 86.38\% AP scores and 88.59\% AUC scores on the XD-Violence and UCF-Crime dataset, outperforming all compared methods in the experiments.

## Training
### Setup
We extract visual features and caption features for UCF-Crime and XD-Violence datasets, and release these features and pretrained models as follows:

| Benchmark | Visual[Baidu]                                                      | Caption[Baidu]                                                     | Model[Baidu]                                                        
|-----------|--------------------------------------------------------------------|--------------------------------------------------------------------|---------------------------------------------------------------------
| UCF-Crime   | [Code: kt75](https://pan.baidu.com/s/17FM7nZGr-Rm_XHp2jozS0w?pwd=kt75) | [Code: fjgy](https://pan.baidu.com/s/1v5nJP8CO2eNIB9DX-4zOWQ?pwd=fjgy) | [Code: vi75](https://pan.baidu.com/s/1uoWN0YooEZ7WckW7Si_asQ?pwd=vi75) 
| XD-Violence | [Code: 9yhu](https://pan.baidu.com/s/1YjcXLWPVOChml9vkqKsUtg?pwd=9yhu) | [Code: pr6e](https://pan.baidu.com/s/1in7-SjEIWf_mE692s2ETrQ?pwd=pr6e) | [Code: 75m7](https://pan.baidu.com/s/1FdLgvJVJ0RCXpN18Yx6K6g?pwd=75m7)

or generate the source_caption features using this [repo](https://github.com/coranholmes/SwinBERT) and then rewrite these captions using the [repo](https://github.com/LijieFan/LaCLIP).

The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the download datasets above in `list/xd_CLIP_rgb.csv` , `list/xd_CLIP_rgbtest.csv` , `list/ucf_CLIP_rgb.csv` and `list/ucf_CLIP_rgbtest.csv`. 
- Change the file paths to the download datasets above in `list/xd_text.csv` , `list/xd_text_test.csv` , `list/ucf_text.csv` and `list/ucf_text_test.csv`. 
- Feel free to change the hyperparameters in `xd_option.py` and `ucf_option.py`.
### Train and Test
After the setup, simply run the following command: 


Traing and infer for XD-Violence dataset
```
python xd_train.py
python xd_test.py
```
Traing and infer for UCF-Crime dataset
```
python ucf_train.py
python ucf_test.py
```
