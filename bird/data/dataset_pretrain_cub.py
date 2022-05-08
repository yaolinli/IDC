import json
import torch
from torch.utils.data.dataloader import default_collate
import pdb
import random
import numpy as np
import math

random.seed(1234)
np.random.seed(1234)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, task, datas, images, vocabs, rev_vocabs, split):
        # set parameter
        self.task = task
        self.images = images
        self.max_len = args.max_len
        self.vocabs = vocabs
        self.rev_vocabs = rev_vocabs
        self.split = split
        self.dataset = args.dataset

        self.BOS = vocabs['<BOS>']
        self.EOS = vocabs['<EOS>']
        self.PAD = vocabs['<PAD>']
        self.UNK = vocabs['<UNK>']
        self.MASK = vocabs['<MASK>']
        self.CLS = vocabs['<CLS>']

        self.datas = datas
        print("[{} task] {} set has {} datas".format(task, split,
                                                     len(self.datas)))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # training and validation
        data = self.datas[index]
        description = data['description']
        batch = {}
        # get raw pairwise input data (img,  text)
        img = torch.from_numpy(self.images[data['img']])
        dim, n, n = img.size(0), img.size(1), img.size(2)
        img = img.view(dim, -1).transpose(0, 1)
        pad_img = torch.LongTensor(n * n * [self.PAD])

        # make sure img shape = [49,2048]
        ImgId = data['img']
        cap, cap_len, cap_label = self.padding(description)
        batch['cap'] = cap
        batch['cap_label'] = cap_label
        # process input data (img, text) according to diff tasks
        # mlm task processed in func self.padding()
        if self.task == 'itm_a':
            rep_prob = random.random()
            if rep_prob < 0.5:
                old_idx = index
                idx = random.choice(range(len(self.datas)))
                if old_idx == idx:
                    align_label = 1
                else:
                    align_label = 0
                    if self.datas[idx]['img'] != self.datas[old_idx]['img']:
                        tri_prob = random.random()
                        ImgId = self.datas[idx]['img']
                        # 1/2 prob replace img
                        if tri_prob < 1/2:
                            img = torch.from_numpy(
                                self.images[self.datas[idx]['img']])
                            img = img.view(dim, -1).transpose(0, 1)
                        # 1/2 prob replace text
                        else:
                            description = self.datas[idx]['description']
                            cap, cap_len, cap_label = self.padding(description)
                    else:
                        # choose the same img
                        align_label = 1
            else:
                align_label = 1
            batch['align_label'] = torch.LongTensor([align_label])
            batch['cap'] = cap
            batch['cap_label'] = cap_label

        elif 'mvm' in self.task:
            img_label = img  # shape [49*2, 2048]
            rep_prob = random.random()
            img_feat_num = n * n
            img_mask = self.get_img_mask(0.15, img_feat_num)
            # mask img
            img = self.mask_img_feat(img, img_mask)
            # img_mask shape [49] if masked feat: 1 else 0
            batch['img_mask'] = img_mask
            batch['img_label'] = img_label

        elif self.task == 'fda':
            pos_cap = [cap]
            negs = data['negs']
            neg_caps = []
            for neg_cap in negs:
                cap, cap_len, cap_label = self.padding(neg_cap)
                neg_caps.append(cap)
            caps = pos_cap + neg_caps
            batch['caps'] = torch.stack(caps) # each image pair has five neg sentences
            batch['cap_label'] = cap_label

        batch['img'] = img
        return batch

    def get_img_mask(self, mask_prob, feat_num):
        img_mask = [random.random() < mask_prob for _ in range(feat_num)]
        if not any(img_mask):
            # at least mask 1 region among n*n
            img_mask[random.choice(range(feat_num))] = True
        img_mask = torch.tensor(img_mask)
        return img_mask

    def mask_img_feat(self, img_feat, img_masks):
        # img_feat shape [49, 2048]
        # img_mask shape [49] e.g. [1,0,0,0,..,1,...]  while 1 means be masked
        img_masks_ext = img_masks.unsqueeze(-1).expand_as(
            img_feat)  # [49] -> [49, 2048]
        img_feat_masked = img_feat.data.masked_fill(
            img_masks_ext, 0)  # replace masked img feature with 0 vector
        return img_feat_masked

    def padding(self, sent):
        if len(sent) > self.max_len - 3:
            sent = sent[:self.max_len - 3]
        text = list(map(lambda t: self.vocabs.get(t, self.UNK), sent))
        # text = [self.CLS, self.BOS] + text + [self.EOS, self.PAD]
        if self.task == 'mlm':
            text, output_label = self.mask_sent(text)
            prob = random.random()
            if prob < 0.15:  # 15% mask <EOS>
                text = [self.BOS] + text + [self.MASK]
                output_label = [-1] + output_label + [self.EOS]
            else:
                text = [self.BOS] + text + [self.EOS]
                output_label = [-1] + output_label + [-1]
        else:
            text = [self.BOS] + text + [self.EOS]
            output_label = [-1] * len(text)
        length = len(text)
        text = text + [self.PAD] * (self.max_len - length)
        output_label = output_label + [-1] * (self.max_len - length)
        T = torch.LongTensor(text)
        output_label = torch.LongTensor(output_label)
        return T, length, output_label

    def random_mask(self, x, i, prob):
        # 80% randomly change token to mask token
        if prob < 0.8:
            x[i] = self.MASK
        # 10% randomly change token to random token
        elif prob < 0.9:
            x[i] = random.choice(list(range(len(self.vocabs))))
        # -> rest 10% randomly keep current token
        return x

    def mask_sent(self, x):
        output_label = []
        for i, token in enumerate(x):
            prob = random.random()
            # mask normal token with 15% probability
            if prob < 0.15:
                prob /= 0.15
                x = self.random_mask(x, i, prob)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
        if all(o == -1 for o in output_label):
            # at least mask 1
            output_label[0] = x[0]
            x[0] = self.MASK
        return x, output_label

    def CLS(self):
        return self.CLS


class MetaLoader(object):
    """ wraps multiple data loaders """
    def __init__(self, loaders, accum_steps=1):
        assert isinstance(loaders, dict)
        self.name2loader = {}
        self.name2iter = {}
        self.sampling_pools = []
        for task, loader in loaders.items():
            if isinstance(loader, tuple):
                l, r = loader
            elif isinstance(loader, torch.utils.data.DataLoader):
                r = 1
            else:
                raise ValueError()
            self.name2loader[task] = l
            self.name2iter[task] = iter(l)
            self.sampling_pools.extend([task] * r)
        self.accum_steps = accum_steps
        self.step = 0

    def __iter__(self):
        """ this iterator will run indefinitely """
        task = self.sampling_pools[0]
        while True:
            if self.step % self.accum_steps == 0:
                task = random.choice(self.sampling_pools)
                self.step += 1
                iter_ = self.name2iter[task]
                try:
                    batch = next(iter_)
                except StopIteration:
                    iter_ = iter(self.name2loader[task])
                    batch = next(iter_)
                    self.name2iter[task] = iter_
            yield task, batch