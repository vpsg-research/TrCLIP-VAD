import numpy as np
import glob
import os
# import cv2
import pandas as pd
import warnings

clip_len = 16

# the dir of testing images
feature_list = 'list/xd_CLIP_rgbtest.csv'

# the ground truth txt
gt_txt = 'list/annotations_multiclasses.txt'
gt_lines = list(open(gt_txt))

#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

gt_segment = []
gt_label = []
lists = pd.read_csv(feature_list)

for idx in range(lists.shape[0]):
    name = lists.loc[idx]['path']
    if '__0.npy' not in name:
        continue
    segment = []
    label = []
    if '_label_A' in name:  #label_A表示视频是正常视频
        fea = np.load(name)  #加载特征
        lens = fea.shape[0] * clip_len  #计算视频长度
        name = name.split('/')[-1]  #获取视频名
        name = name[:-7]  #去掉后缀名
        segment.append([0, lens])  #存储正常片段的起始和结束时间
        label.append('A')  #存储正常片段的标签
    else:  #异常视频
        name = name.split('/')[-1]
        name = name[:-7]
        for gt_line in gt_lines:
            if name in gt_line:
                gt_content = gt_line.strip('\n').split()  #split()默认以空格为分隔符
                for j in range(1, len(gt_content), 3):  # 遍历每一个异常片段
                    print(gt_content, j)
                    segment.append([gt_content[j + 1], gt_content[j + 2]])  #存储异常片段的起始和结束时间
                    label.append(gt_content[j])  #存储异常片段的标签
                break
    gt_segment.append(segment)  #存储所有视频的异常片段
    gt_label.append(label)  #存储所有视频的异常片段的标签
    
np.save('list/gt_label.npy', gt_label)  #保存所有视频的异常片段的标签
np.save('list/gt_segment.npy', gt_segment)  #保存所有视频的异常片段