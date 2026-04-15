import numpy as np
# import glob
# import os
# import cv2
import pandas as pd
# import warnings

clip_len = 16

feature_list = '/test004/code/VadCLIP-main/list/ucf_CLIP_rgbtest.csv'

# the ground truth txt
gt_txt = '/test004/code/VadCLIP-main/list/Temporal_Anomaly_Annotation.txt'
gt_lines = list(open(gt_txt))

#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

gt_segment = []
gt_label = []
lists = pd.read_csv(feature_list)

for idx in range(lists.shape[0]):
    name = lists.loc[idx]['path']
    label_text = lists.loc[idx]['label']
    if '__0.npy' not in name:
        continue
    segment = []
    label = []
    if 'Normal' in label_text:  #判断是否为正常片段
        fea = np.load(name)
        lens = fea.shape[0] * clip_len  #计算该片段的时长
        name = name.split('/')[-1]
        name = name[:-7]
        segment.append([0, lens])  #存储正常片段的起始和结束时间
        label.append('A')  #添加正常类型
    else:
        name = name.split('/')[-1]
        name = name[:-7]
        for gt_line in gt_lines:
            if name in gt_line:
                gt_content = gt_line.strip('\n').split('  ')
                segment.append([gt_content[2], gt_content[3]])  #存储异常片段的起始和结束时间
                label.append(gt_content[1])  #添加异常类型
                if gt_content[4] != '-1':  #检查是否存在第二个异常片段
                    segment.append([gt_content[4], gt_content[5]])  #若存在，存储异常片段的起始和结束时间
                    label.append(gt_content[1])  #添加异常类型
                break
    gt_segment.append(segment)
    gt_label.append(label)
    
np.save('list/gt_label_ucf.npy', gt_label)  #保存异常类型标签
np.save('list/gt_segment_ucf.npy', gt_segment)   #保存异常片段