import json
import torch
from torch.utils.data.dataloader import default_collate
import pdb 
import random
import numpy as np
import math
import os
random.seed(1234)
np.random.seed(1234)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, datas, images, split):
        # set parameter
        self.split = split
        self.dataset = args.dataset
        self.data_path = args.data_path
        self.datas = datas
        self.images = images
        print(" {} set has {} datas".format(split, len(self.datas)))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # training and validation
        data = self.datas[index]
        batch = {}
        # get raw pairwise input data (img1, img2)
        img1 = torch.from_numpy(self.images[data['img1']]).transpose(0,1)
        img2 = torch.from_numpy(self.images[data['img2']]).transpose(0,1)
        batch['img1'] = img1
        batch['img2'] = img2
        batch['class1'] = torch.LongTensor([data['class_id1']])
        batch['class2'] = torch.LongTensor([data['class_id2']])
        batch['label'] = torch.LongTensor([data['label']])
        return batch
