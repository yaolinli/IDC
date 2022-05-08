import json
import torch
from torch.utils.data.dataloader import default_collate
import pdb 
import random
import numpy as np
from para import parse_args
args = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path,  vocabs, rev_vocabs, images, split, set_type=None):
        # set parameter
        self.max_len = args.max_len
        self.vocabs = vocabs
        self.rev_vocabs = rev_vocabs
        self.split = split
        self.dataset = args.dataset
        self.set_type = set_type

        self.BOS = vocabs['<BOS>']
        self.EOS = vocabs['<EOS>']
        self.PAD = vocabs['<PAD>']
        self.UNK = vocabs['<UNK>']
        self.MASK = vocabs['<MASK>']
        self.CLS = vocabs['<CLS>']

        # load data
        if self.split == 'train' or self.set_type == 'P':
            self.load_data_multi_sents(data_path)
        else:
            self.load_data(data_path)
        
        self.images = {}
        for type in images.keys():
            self.images.update(images[type])

    def load_data_multi_sents(self, data_path):
        self.datas = []
        count_s = 0
        count_ns = 0
        with open(data_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                jterm = json.loads(line.strip())
                # change multi description to one description per data
                img2name = jterm['img2']
                type = img2name.split('_')[1]
                if type == 'semantic':
                    count_s += len(jterm['sentences'])
                else:
                    count_ns += len(jterm['sentences'])
                for des in jterm['sentences']:
                    new_jterm = {}
                    new_jterm['img1'] = jterm['img1']
                    new_jterm['img2'] = jterm['img2']
                    new_jterm['sentences'] = des.split(' ')
                    self.datas.append(new_jterm)
                    if self.set_type == 'P':  # for efficient, validate one datat per caption
                        break
        print(self.split, 'Total datas ', len(self.datas), '; semantic ', count_s, '; nonsemantic ', count_ns)
    
    def load_data(self, data_path):
        self.datas = []
        with open(data_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                jterm = json.loads(line.strip())
                jterm['sentences'] = [cap.split(' ') for cap in jterm['sentences']]
                self.datas.append(jterm)
        print('Total datas ', len(self.datas))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        description = data['sentences']
        batch = {}
        # train 
        if self.split == 'train' or self.set_type == 'P':
            # get raw triplet input data (img1, img2, text)
            img1 = torch.from_numpy(self.images[data['img1']]).float()
            img2 = torch.from_numpy(self.images[data['img2']]).float()
            dim, n, n = img1.size(0), img1.size(1), img1.size(2)
            img1, img2 = img1.view(dim, -1).transpose(0,1), img2.view(dim, -1).transpose(0,1)
            # make sure img1 & img2 shape = [49,2048] 
            ImgId = data['img1']+'_'+data['img2']
            # semantic or non semantic change
            type = data['img2'].split('_')[1]
            if type == 'semantic':
                diff_label = 1
            else: # type = nonsemantic
                diff_label = 0
            cap, cap_len, cap_label = self.padding(description)
            batch['img1'] = img1
            batch['img2'] = img2
            batch['cap'] = cap
            batch['cap_label'] = cap_label
            batch['diff_label'] = torch.LongTensor([diff_label])
            return batch
        # valid and test (used for Inference algorithm)
        else:
            # get raw triplet input data (img1, img2, text)
            img1 = torch.from_numpy(self.images[data['img1']]).float()
            img2 = torch.from_numpy(self.images[data['img2']]).float()
            dim, n, n = img1.size(0), img1.size(1), img1.size(2)
            img1, img2 = img1.view(dim, -1).transpose(0,1), img2.view(dim, -1).transpose(0,1)
            # make sure img1 & img2 shape = [49,2048] 
            ImgId = data['img1']+'_'+data['img2']
            gt_caps =  [' '.join(tokens) for tokens in description]
            return img1, img2, gt_caps, ImgId


    def padding(self, sent):
        if len(sent) > self.max_len-3:
            sent = sent[:self.max_len-3]
        text = list(map(lambda t: self.vocabs.get(t, self.UNK), sent))
        text, output_label = self.mask_sent(text)
        
        prob = random.random()
        if prob < 0.15: # 15% mask <EOS>
            text = [self.BOS] + text + [self.MASK]
            output_label = [-1] + output_label + [self.EOS]
        else:
            text = [self.BOS] + text + [self.EOS]
            output_label = [-1] + output_label + [-1]
        length = len(text)
        text = text + [self.PAD]*(self.max_len - length)
        output_label = output_label + [-1]*(self.max_len - length)
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

def self_collate(batch):
    transposed = list(zip(*batch))
    img1_batch = default_collate(transposed[0])
    img2_batch = default_collate(transposed[1])
    GT_batch = transposed[2]  # type: list
    ImgId_batch = transposed[3] # type: list
    return (img1_batch, img2_batch, GT_batch, ImgId_batch)

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        kwargs['collate_fn'] = self_collate
        super().__init__(dataset, **kwargs)