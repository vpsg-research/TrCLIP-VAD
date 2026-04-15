import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools

class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, text_path: str, test_mode: bool, label_map: dict, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.dftext = pd.read_csv(text_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
            self.dftext = self.dftext.loc[self.dftext['label'] == 'Normal']
            self.dftext = self.dftext.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()
            self.dftext = self.dftext.loc[self.dftext['label'] != 'Normal']
            self.dftext = self.dftext.reset_index()

    def __len__(self):
        # return self.df.shape[0], self.dftext.shape[0]
        return self.df.shape[0]
        # return self.dftext.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        text_feature = np.load(self.dftext.loc[index]['path'])
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
            text_feature, text_length = tools.process_feat_text(text_feature, self.clip_dim)

        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
            text_feature, text_length = tools.process_split_text(text_feature, self.clip_dim)
        clip_feature = torch.tensor(clip_feature)
        text_feature = torch.tensor(text_feature)
        clip_label = self.df.loc[index]['label']
        text_label = self.dftext.loc[index]['label']
        return clip_feature, clip_label, clip_length, text_feature, text_label, text_length



class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, text_path: str, test_mode: bool, label_map: dict):
        self.df = pd.read_csv(file_path)
        self.dftext = pd.read_csv(text_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        text_feature = np.load(self.dftext.loc[index]['path'])
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
            text_feature, text_length = tools.process_feat_text(text_feature, self.clip_dim)

        else: # test
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
            text_feature, text_length = tools.process_split_text(text_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        text_feature = torch.tensor(text_feature)
        clip_label = self.df.loc[index]['label']
        text_label = self.dftext.loc[index]['label']
        return clip_feature, clip_label, clip_length, text_feature, text_label, text_length

