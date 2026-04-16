# TrCLIP-VAD: Weak supervised video anomaly detection by improving CLIP training with text rewriting


## :loudspeaker:News
- [2026-04] Our paper has been accepted for publication in the journal **Neural Networks** (Elsevier)! 🎉
- [2026-04] The code are being organized and will be released shortly. Please star this repo for updates!

## :sparkles: Highlight

- 🧠 We propose a novel TrCLIP-VAD framework, which enhances textual feature diversity through text rewriting strategy. As far as we know, TrCLIP-VAD is the first work to introduce an image-text dual feature enhancement approach to WSVAD.

- ⚡ We design an LGM-Mamba module that innovatively integrates the abilities of local perception, global modeling and multi-scale analysis, enabling efficient and comprehensive temporal feature learning for VAD.

- 🚀 TrCLIP-VAD achieves state-of-the-art (SOTA) performance on two widely-used datasets. Specifically, it achieves 86.38\% AP scores and 88.59\% AUC scores on the XD-Violence and UCF-Crime dataset, outperforming all compared methods in the experiments.

## 📆Table of Contents

- [📖Introduction]


## 📖 Introduction

This is the official repository for the NN 2026 paper " [TrCLIP-VAD: Weak supervised video anomaly detection by improving CLIP training with text rewriting](https://www.sciencedirect.com/science/article/abs/pii/S0893608026004120) "

<p align="center">
  <img src="https://github.com/vpsg-research/TrCLIP-VAD/blob/main/data/motive.png" style="width:90%;">
</p>
<p style="font-size: 0.9em; color: #666; text-align: justify;">
  <em>Fig. 1: Comparisons with the existing approaches. Unlike other methods, after obtaining the caption features using the captioning network, we use LLaMA to rewrite the captions into two versions and randomly select one to fuse with visual features for anomaly detection.</em>
</p>

## 🧩 Overview

<p align="center">
  <img src="https://github.com/vpsg-research/TrCLIP-VAD/blob/main/data/framework.png" style="width:90%;">

  style="font-size: 0.9em; color: #666; text-align: justify;">
  <em>Fig. 1: The overview of the proposed TrCLIP-VAD. It consists of several components, i.e. visual branch, text branch, C-branch (coarse-grained branch) and F-branch (fine-grained branch).</em>
</p>

<p align="center">
  <img src="https://github.com/vpsg-research/TrCLIP-VAD/blob/main/data/framework.png" style="width:90%;">
</p>
<p style="font-size: 0.9em; color: #666; text-align: justify;">
  <em>Fig. 1: The overview of the proposed TrCLIP-VAD. It consists of several components, i.e. visual branch, text branch, C-branch (coarse-grained branch) and F-branch (fine-grained branch).</em>
</p>

## Training
### Setup
We extract visual features and caption features for UCF-Crime and XD-Violence datasets, and release these features and pretrained models as follows:

| Benchmark | Visual[Baidu]                                                      | Caption[Baidu]                                                     | Model[Baidu]                                                        
|-----------|--------------------------------------------------------------------|--------------------------------------------------------------------|---------------------------------------------------------------------
| UCF-Crime   | [Code: kt75](https://pan.baidu.com/s/17FM7nZGr-Rm_XHp2jozS0w?pwd=kt75) | [Code: fjgy](https://pan.baidu.com/s/1v5nJP8CO2eNIB9DX-4zOWQ?pwd=fjgy) | [Code: 8xqt](https://pan.baidu.com/s/1mXc87UVt8uPBA0JzwXhV_w?pwd=8xqt) 
| XD-Violence | [Code: 9yhu](https://pan.baidu.com/s/1YjcXLWPVOChml9vkqKsUtg?pwd=9yhu) | [Code: pr6e](https://pan.baidu.com/s/1in7-SjEIWf_mE692s2ETrQ?pwd=pr6e) | [Code: tztn](https://pan.baidu.com/s/1g212yNHYlEcJNCf8sEr_JQ?pwd=tztn)

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
## References
We referenced the repos below for the code.

[LaCLIP](https://github.com/LijieFan/LaCLIP)

[VadCLIP](https://github.com/nwpu-zxr/VadCLIP/tree/main)

## Citation
If you find this repo useful for your research, please consider citing our paper:

```bibtex
@article{SHEN2026108951,
	title = {TrCLIP-VAD : Weak supervised video anomaly detection by improving CLIP training with text rewriting},
	journal = {Neural Networks},
	volume = {201},
	pages = {108951},
	year = {2026},
	issn = {0893-6080},
	doi = {https://doi.org/10.1016/j.neunet.2026.108951},
	url = {https://www.sciencedirect.com/science/article/pii/S0893608026004120},
	author = {Shengjie Shen and Ziteng Guo and Yahui Li and Liejun Wang and Zhiqing Guo},
}
