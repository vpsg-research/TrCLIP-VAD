#生成ucf_CLIP_rgb.csv文件，用于保存训练集的特征文件路径及其标签

import os
import csv

root_path = '/home/xbgydx/Desktop/UCFClipFeatures/'
txt = 'list/Anomaly_Train.txt'
files = list(open(txt))
normal = []
count = 0

with open('list/ucf_CLIP_rgb.csv', 'w+') as f:  ## the name of feature list
    writer = csv.writer(f)  #创建wrieter写入器
    writer.writerow(['path', 'label'])  #写入表头
    for file in files:
        filename = root_path + file[:-5] + '__0.npy'  #特征文件名
        label = file.split('/')[0]  #标签
        if os.path.exists(filename):  #判断文件是否存在
            if 'Normal' in label:
                #continue
                filename = filename[:-5]  #去掉后缀名
                for i in range(0, 10, 1):  #生成10个特征文件
                    normal.append(filename + str(i) + '.npy')
            else:
                filename = filename[:-5]  #去掉后缀名
                for i in range(0, 10, 1):  #生成10个特征文件
                    writer.writerow([filename + str(i) + '.npy', label])  #写入文件路径及标签
        else:
            count += 1
            print(filename)
            
    for file in normal:  #重复写入正常标签
        writer.writerow([file, 'Normal'])

print(count)

