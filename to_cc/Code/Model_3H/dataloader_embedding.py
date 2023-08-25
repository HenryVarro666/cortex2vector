from __future__ import print_function

import torch.utils.data as data
import os
import random
import os.path
import numpy as np
import torch

import pdb


def load_data(data_path):
    # pdb.set_trace()
    data = []
    DataID_list = [x.split('.')[0] for x in os.listdir(data_path) if not x.startswith('.')]
    for DataID in DataID_list:
        multi_hot_feature_path = data_path + '/' + DataID + '.txt'
        data.append((DataID, multi_hot_feature_path))
    random.shuffle(data)
    return data


class GraphEmbedding(data.Dataset):
    def __init__(self, data_path, all_data, train=True, test=False):
        self.data_path = data_path
        self.train = train  # training set or val set
        self.test = test
        # pdb.set_trace()
        if self.train:
            self.data = all_data[:120000]
            # self.data = all_data[:151]
        elif not test:
            self.data = all_data[120000:140000]
            # self.data = all_data[151:]
        else:
            self.data = all_data[140000:]

        random.shuffle(self.data)

    def __getitem__(self, index):
        DataID, multi_hot_feature_path = self.data[index]
        multi_hot_feature_orig = np.loadtxt(multi_hot_feature_path)
        multi_hot_feature = multi_hot_feature_orig[0:2, :]
        return DataID, multi_hot_feature

    def debug_getitem__(self, index=0):
        pdb.set_trace()
        DataID, multi_hot_feature_path = self.data[index]
        multi_hot_feature_orig = np.loadtxt(multi_hot_feature_path)
        multi_hot_feature = multi_hot_feature_orig[0:2, :]
        return DataID, multi_hot_feature

    def __len__(self):
        return len(self.data)


def get_loader(data_path, data_list, training, test, batch_size=128, num_workers=4):
    dataset = GraphEmbedding(data_path, data_list, training, test)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              drop_last=not test)
    return data_loader


if __name__ == '__main__':
    data_path = '../Graph_embedding_data_500/node_input_data'
    all_data = load_data(data_path)

    dataset = GraphEmbedding(data_path, all_data)
    for i in range(len(dataset)):
        DataID, multi_hot_feature = dataset.debug_getitem__(i)