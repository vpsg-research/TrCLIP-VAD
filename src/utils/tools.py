import torch
import numpy as np

def get_batch_label(texts, prompt_text, label_map: dict):
    label_vectors = torch.zeros(0)
    if len(label_map) != 7:
        if len(label_map) == 2:
            for text in texts:
                label_vector = torch.zeros(2)
                if text == 'Normal':
                    label_vector[0] = 1
                else:
                    label_vector[1] = 1
                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
        else:
            for text in texts:
                label_vector = torch.zeros(len(prompt_text))
                if text in label_map:
                    label_text = label_map[text]
                    label_vector[prompt_text.index(label_text)] = 1

                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
    else:
        for text in texts:
            label_vector = torch.zeros(len(prompt_text))
            labels = text.split('-')
            for label in labels:
                if label in label_map:
                    label_text = label_map[label]
                    label_vector[prompt_text.index(label_text)] = 1
            
            label_vector = label_vector.unsqueeze(0)
            label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    return label_vectors

def get_prompt_text(label_map: dict): # 获取提示文本
    prompt_text = []
    for v in label_map.values():
        prompt_text.append(v)

    return prompt_text

def get_batch_mask(lengths, maxlen):
    batch_size = lengths.shape[0]  # 1
    mask = torch.empty(batch_size, maxlen)  # 1 * 256
    mask.fill_(0)
    for i in range(batch_size):
        if lengths[i] < maxlen:
            mask[i, lengths[i]:maxlen] = 1
    
    return mask.bool()

def random_extract(feat, t_max):
   r = np.random.randint(feat.shape[0] - t_max)
   return feat[r : r+t_max, :]

def uniform_extract(feat, t_max, avg: bool = True):
    new_feat = np.zeros((t_max, feat.shape[1])).astype(np.float32)  # 初始化新的特征数据 96 * 512
    r = np.linspace(0, len(feat), t_max+1, dtype=np.int32)    # 等间隔采样
    if avg == True:
        for i in range(t_max):
            if r[i]!=r[i+1]:
                new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
            else:
                new_feat[i,:] = feat[r[i],:]
    else:
        r = np.linspace(0, feat.shape[0]-1, t_max, dtype=np.uint16)
        new_feat = feat[r, :]
            
    return new_feat

def pad(feat, min_len): # 输入的特征数据 feat 进行 padding，使其长度达到 min_len
    clip_length = feat.shape[0]
    if clip_length <= min_len:
       return np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat

def process_feat(feat, length, is_random=False): # 对输入的特征数据进行处理，使其长度符合期望的长度
    clip_length = feat.shape[0]
    if feat.shape[0] > length:  # 帧数大于期望长度
        if is_random:
            return random_extract(feat, length), length
        else:
            return uniform_extract(feat, length), length
    else:  # 帧数小于期望长度
        return pad(feat, length), clip_length

def process_feat_text(text, length, is_random=False): # 对输入的特征数据进行处理，使其长度符合期望的长度
    text_length = text.shape[0]

    if text.shape[0] > length:  # 帧数大于期望长度
        if is_random:
            return random_extract(text, length), length
        else:
            return uniform_extract(text, length), length
    else:  # 帧数小于期望长度
        return pad(text, length), text_length
def process_split(feat, length): # 输入的特征数据 feat 按照指定的长度 length 进行分割
    clip_length = feat.shape[0]
    if clip_length < length:
        return pad(feat, length), clip_length
    else:
        split_num = int(clip_length / length) + 1  # 计算需要分割的次数, 向上取整
        for i in range(split_num):
            if i == 0:
                split_feat = feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])
            elif i < split_num - 1:
                split_feat = np.concatenate([split_feat, feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])], axis=0)
            else:
                split_feat = np.concatenate([split_feat, pad(feat[i*length:i*length+length, :], length).reshape(1, length, feat.shape[1])], axis=0)

        return split_feat, clip_length


def process_split_text(feat, length): # 输入的特征数据 feat 按照指定的长度 length=256 进行分割
    clip_length = feat.shape[0] # 输入数据的长度 115
    if clip_length < length: # 如果输入数据的长度小于clip_length=256
        return pad(feat, length), clip_length # 填充

    else: # 输入数据的长度大于等于clip_length=256
        split_num = int(clip_length / length) + 1  # 分割次数, 向上取整
        for i in range(split_num):
            if i == 0:
                split_feat = feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1]) # 取出从索引 0 到 length 的数据并重新调整形状为三维数组
            elif i < split_num - 1: # 拼接新分割的数据
                split_feat = np.concatenate([split_feat, feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])], axis=0)
            else:
                split_feat = np.concatenate([split_feat, pad(feat[i*length:i*length+length, :], length).reshape(1, length, feat.shape[1])], axis=0)

        return split_feat, clip_length