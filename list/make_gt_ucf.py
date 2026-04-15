import numpy as np
import pandas as pd
# import cv2

clip_len = 16

# the dir of testing images
feature_list = 'list/ucf_CLIP_rgbtest.csv'
# the ground truth txt

gt_txt = 'list/Temporal_Anomaly_Annotation.txt'     ## the path of test annotations
gt_lines = list(open(gt_txt))
gt = []
lists = pd.read_csv(feature_list)
count = 0

for idx in range(lists.shape[0]):
    name = lists.loc[idx]['path']
    if '__0.npy' not in name:  #只处理没有异常的视频，所有的name都没有__0.npy，那就是所有的文件都不处理吗？
        continue
    #feature = name.split('label_')[-1]
    fea = np.load(name)
    lens = (fea.shape[0] + 1) * clip_len  #数量加1并乘以每个片段的帧数来计算视频的总帧数
    name = name.split('/')[-1]  #用/分割文件名，取最后一个元素作为文件名
    name = name[:-7]  #去掉后7个字符，即'__0.npy'
    # the number of testing images in this sub-dir

    gt_vec = np.zeros(lens).astype(np.float32)
    if 'Normal' not in name:
        for gt_line in gt_lines:
            if name in gt_line:
                count += 1
                gt_content = gt_line.strip('\n').split('  ')[1:-1]  #去掉换行符，并以'  '分割，取第2到倒数第二个元素作为gt
                abnormal_fragment = [[int(gt_content[i]),int(gt_content[j])] for i in range(1,len(gt_content),2) \
                                     for j in range(2,len(gt_content),2) if j==i+1]
                if len(abnormal_fragment) != 0:  #如果长度不为0，说明有异常片段
                    abnormal_fragment = np.array(abnormal_fragment)
                    for frag in abnormal_fragment:  #遍历异常片段
                        if frag[0] != -1 and frag[1] != -1:  #检查 frag 数组的第一个元素（起始帧索引）和第二个元素（结束帧索引）是否不等于 -1。-1 通常表示没有异常或无效的帧索引。这个条件确保只处理有效的异常片段。
                            gt_vec[frag[0]:frag[1]]=1.0  #将异常片段的gt置为1.0
                break
    gt.extend(gt_vec[:-clip_len])

print(count)
np.save('list/gt_ucf.npy', gt)  #保存ucf测试集的ground truth

