from collections import defaultdict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
import random

class Model(nn.Module):
    def __init__(self, ff_dim, img_embs, n_hidden, n_head, n_block, vocab_size, dropout, max_len, tasks, CLS):
        super(Model, self).__init__()

        # initilize parameter
        self.ff_dim = ff_dim # FeedForward 2048
        self.img_embs = img_embs # 2048
        self.n_hidden = n_hidden # 512
        self.n_head = n_head
        self.n_block = n_block
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dropout = dropout
        self.CLS = CLS
        self.tasks = tasks
        self.nce_temp = 1.0 # temperature in NCE loss
        self.neg_num = 6

        self.word_embedding = nn.Embedding(self.vocab_size, n_hidden)
        self.img_project = nn.Sequential(nn.Linear(img_embs, n_hidden), nn.Sigmoid())
        self.position_encoding = PositionalEmb(n_hidden, dropout, 7)
        self.single_img_encoder = Transformer(n_embs=n_hidden, dim_ff=ff_dim, n_head=n_head, n_block=1, dropout=dropout)
        self.double_img_encoder = Transformer(n_embs=n_hidden, dim_ff=ff_dim, n_head=n_head, n_block=1, dropout=dropout)
        self.encoder = Transformer(n_embs=n_hidden, dim_ff=ff_dim, n_head=n_head, n_block=self.n_block, dropout=dropout)
        self.output_layer = nn.Linear(self.n_hidden, self.vocab_size)
        # self.pooler = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.Tanh(), nn.Linear(n_hidden, 2)) 
        self.pooler = nn.Sequential(nn.Linear(self.n_hidden*3, 2))
        if 'itm_b' in tasks:
            self.aligner = nn.Sequential(nn.Linear(self.n_hidden*3, 2))
        self.feat_regress = RegionFeatureRegression(n_hidden, img_embs, self.img_project[0].weight)

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        Img = Variable(batch['img']).cuda()
        Img = self.img_project(Img)
        img_embs = self.position_encoding(Img, mode='img', pos=2)
        # add a special <CLS> token
        CLS = torch.tensor([[self.CLS]])
        CLS = CLS.repeat(Img.size(0), 1)
        CLS = self.word_embedding(CLS.cuda())
        CLS1 = self.position_encoding(CLS, mode='cls', pos=1)
        CLS2 = self.position_encoding(CLS, mode='cls', pos=3)
        CLS3 = self.position_encoding(CLS, mode='cls', pos=5)
        image_embs1 = torch.cat((CLS1, img_embs), dim=1)
        image_embs1 = self.single_img_encoder(image_embs1)
        image_embs = self.double_img_encoder(torch.cat((image_embs1, CLS2), dim=1))
        
        img_toklen = Img.size(1)
        start_idx = {
            'CLS1':0,
            'img':1,
            'CLS2':1+img_toklen,
            'CLS3':2+img_toklen,
            'text':3+img_toklen,
        }
        if task == 'fda':
            # each img pair has 1 pos_caps & 5 neg_caps
            Caps = Variable(batch['caps']).cuda() # [bz, 6, max_len]
            self.neg_num = Caps.size(1)-1
            # print("neg_num", self.neg_num)
            bz, n, mlen = Caps.size(0), Caps.size(1), Caps.size(2)
            Caps = Caps.view(-1, mlen)
            # convert token to word embedding & add position embedding
            text_embs = self.word_embedding(Caps)
            text_embs = self.position_encoding(text_embs, mode='text', pos=6)
            text_embs = torch.cat((CLS3.repeat(self.neg_num+1, 1, 1), text_embs), dim=1)
            
            bz, n, dim = image_embs.size(0), image_embs.size(1), image_embs.size(2)
            image_embs = image_embs.repeat(1, self.neg_num+1, 1).view(-1, n, dim)
            # input to multi_modal transformer
            input_embs = torch.cat((image_embs, text_embs), dim=1)
            return self.fda_forward(task, input_embs, start_idx, compute_loss)
        else:
            Cap = Variable(batch['cap']).cuda()
            Cap_label = Variable(batch['cap_label']).cuda()
            # convert token to word embedding & add position embedding
            text_embs = self.word_embedding(Cap)
            text_embs = self.position_encoding(text_embs, mode='text', pos=6)
            # concate [cls1,img1,cls2,img2,cls3,text] as input to Unified Transformer
            input_embs = torch.cat((image_embs, CLS3, text_embs), dim=1)
            text_toklen = Cap.size(1)

            if task == 'mlm':
                return self.mlm_forward(input_embs, Cap_label, img_toklen, text_toklen, start_idx, compute_loss)
            elif task.startswith('mvm'):
                img_mask = Variable(batch['img_mask']).cuda()
                img_label = Variable(batch['img_label']).cuda()
                return self.mvm_forward(task, input_embs, img_mask, img_label, img_toklen, start_idx, compute_loss)
            elif task.startswith('itm'):
                align_label = Variable(batch['align_label']).cuda()
                return self.itm_forward(task, input_embs, align_label, start_idx, compute_loss)

    def mlm_forward(self, input_embs, cap_label, img_toklen, text_toklen, start_idx, compute_loss=True):
        '''
        input_embs shape([batch size, 199, 512])
        cap_label shape([64, 100]) e.g.  [-1,-1,..,129,...]
        '''
        output = self.encoder(input_embs)
        text = output[:, start_idx['text']:, :]
        # only compute masked tokens for better efficiency
        masked_output = text[cap_label != -1].contiguous().view(-1, text.size(-1))
        # map hidden dim to vocab size
        prediction_scores = self.output_layer(masked_output)
        
        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                            cap_label[cap_label != -1].contiguous().view(-1),
                                            reduction='none')                            
            return masked_lm_loss, prediction_scores
        else:
            return prediction_scores

    def mvm_forward(self, task, input_embs, img_mask, img_label, img_toklen, start_idx, compute_loss=True):
        output = self.encoder(input_embs)
        img = output[:, start_idx['img']:(start_idx['img']+img_toklen), :]
        # only compute masked tokens for better efficiency
        mask = img_mask.unsqueeze(-1).expand_as(img) # [64, 98] -> [64, 98, 512]
        img_masked = img[mask].contiguous().view(-1, img.size(-1))
        prediction_feat = self.feat_regress(img_masked)

        label_mask = img_mask.unsqueeze(-1).expand_as(img_label)
        target_feat = img_label[label_mask].contiguous().view(-1, img_label.size(-1))
        if compute_loss:
            if task == 'mvm': # use l2 loss
                mrfr_loss = F.mse_loss(prediction_feat, target_feat, reduction='none')
            else: # use contrastive loss
                mask = (~img_mask).unsqueeze(-1).expand_as(img) 
                img_masked = img[mask].contiguous().view(-1, img.size(-1))
                neg_pred_feat = self.feat_regress(img_masked)
                mrfr_loss = self.mfm_nce(prediction_feat, target_feat, neg_pred_feat)
            return mrfr_loss, prediction_feat
        else:
            return prediction_feat
        
    def mfm_nce(self, masked_output, pos_output, neg_output):
        masked_output = F.normalize(masked_output, dim=1)
        pos_output = F.normalize(pos_output, dim=1)
        neg_output = F.normalize(neg_output, dim=1)
        # dot product of ground truth feature
        masked_score = masked_output.matmul(pos_output.t()) # [N, N]
        # dot product of neative samples
        neg_score = masked_output.matmul(neg_output.t()) # [N, M]
        logits = torch.cat([masked_score, neg_score], dim=1).float() # [N, N+M]
        targets = torch.arange(0, masked_output.size(0)).long().cuda()
        loss = F.cross_entropy(logits/self.nce_temp, targets,
                                reduction='none')
        return loss
    
    def itm_forward(self, task, input_embs, align_label, start_idx, compute_loss=True):
        output = self.encoder(input_embs)
        CLS = torch.cat((output[:, start_idx['CLS1'], :], output[:, start_idx['CLS2'], :], output[:, start_idx['CLS3'], :]), dim=1)
        # feed the cls feature to 2 layer FC layers
        if task=='itm_a':
            itm_scores = self.pooler(CLS)
        else:
            itm_scores = self.aligner(CLS)
        if compute_loss:
            align_label = align_label.view(-1)
            itm_loss = F.cross_entropy(itm_scores, align_label, reduction='none')
            return itm_loss, itm_scores
        else:
            return itm_scores
    
    def fda_forward(self, task, input_embs, start_id, compute_loss=True):
        output = self.encoder(input_embs)
        img_CLS = (output[:, start_id['CLS1'], :] + output[:, start_id['CLS2'], :])/2
        text_CLS = output[:, start_id['CLS3'], :]
        # feed the cls feature to 2 layer FC layers
        if compute_loss:
            sim_loss, avg_pos_sim, avg_neg_sim, acc = self.fda_nce(img_CLS, text_CLS)
            return sim_loss, [avg_pos_sim, avg_neg_sim, acc]
        else:
            return sim_scores

    def fda_nce(self, img, text, temperature=1.0):
        batch_size = img.size(0)
        img = F.normalize(img, dim=1)
        text = F.normalize(text, dim=1)
        sim = torch.diag(img.matmul(text.t()))
        avg_pos_sim = 0.0
        avg_neg_sim = 0.0
        logits = []
        for i in range(0, batch_size, self.neg_num+1):
            pos_sim = sim[i]
            neg_sim = sim[(i+1):(i+self.neg_num+1)]
            logits.append(sim[i:i+self.neg_num+1])
            avg_pos_sim += pos_sim
            avg_neg_sim += neg_sim.mean()
        avg_pos_sim /= (batch_size/(self.neg_num+1))
        avg_neg_sim /= (batch_size/(self.neg_num+1))

        logits = torch.stack(logits, dim=0) # [batch_size/neg_samples, 1+neg_samples]
        logits = logits / temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda() # pos sample is label 0
        nce_loss = F.cross_entropy(logits, labels, reduction='mean')
        
        acc = (logits.max(dim=-1)[1] == labels).sum().item() / labels.size(0)
        return nce_loss, avg_pos_sim, avg_neg_sim, acc


class RegionFeatureRegression(nn.Module):
    " for MVM"
    def __init__(self, hidden_size, feat_dim, img_linear_weight):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                GELU(),
                                LayerNorm(hidden_size, eps=1e-12))

        self.weight = img_linear_weight
        self.bias = nn.Parameter(torch.zeros(feat_dim))

    def forward(self, input_):
        hidden = self.net(input_)
        output = F.linear(hidden, self.weight.t(), self.bias)
        return output

class Transformer(nn.Module):
    def __init__(self, n_embs, dim_ff, n_head, dropout, n_block):
        super(Transformer, self).__init__()
        self.norm = LayerNorm(n_embs)
        self.layers = nn.ModuleList([AttnBlock(n_embs, dim_ff, n_head, dropout) for _ in range(n_block)])
        self.N = n_block
        self.cache = None
    def _init_cache(self):
        self.cache = {}
        for i in range(self.N):
            self.cache['layer_%d'%i] = {
                'self_keys': None,
                'self_values': None,
                'self_masks': None,
            }
    def get_cache(self):
        return self.cache
    def forward(self, x, mask=None, step=None):
        x = self.norm(x)
        if step == 0:
            self._init_cache()
        for i in range(self.N):
            layer_cache = self.cache['layer_%d'%i] if step is not None else None
            x = self.layers[i](x, mask, layer_cache=layer_cache)
        return x

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def shape(self, x):
        bs = x.size(0)
        return x.view(bs, -1, self.h, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None, layer_cache=None):
        if layer_cache is not None:
            k = self.shape(self.k_linear(k)) # ([150, 8, n, 64])
            v = self.shape(self.v_linear(v))
            if layer_cache['self_keys'] is not None:
                # load img tokens k cache
                k = torch.cat((layer_cache['self_keys'], k), dim=2) # ([150, 8, 99+n, 64])
            else:
                layer_cache['self_keys'] = k

            if layer_cache['self_values'] is not None:
                # load img tokens v cache
                v = torch.cat((layer_cache['self_values'], v), dim=2)
            else:
                layer_cache['self_values'] = v
        else:
            k = self.shape(self.k_linear(k))
            v = self.shape(self.v_linear(v))
        bs = q.size(0)
        q = self.shape(self.q_linear(q)) # ([150, 8, n, 64])
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))

class GELU(nn.Module):
    def forward(self, input_):
        output = gelu(input_)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, dim_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        # return self.w_2(self.dropout(gelu(self.w_1(x))))

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class AttnBlock(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout):
        super(AttnBlock, self).__init__()
        self.attn = MultiHeadAttention(dim, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(dim, dim_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(dim, dropout) for _ in range(2)])

    def forward(self, x, mask=None, layer_cache=None):
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, mask, layer_cache))
        return self.sublayer[1](x, self.feed_forward)

def subsequent_mask(batch, size):
    attn_shape = (batch, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    mask = torch.from_numpy(subsequent_mask) == 0
    return mask

class PositionalEmb(nn.Module):
    "Implement the PE function for img or text or <CLS>."
    def __init__(self, d_model, dropout, max_len=7, max_seq_len=200):
        super(PositionalEmb, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.type_pe = torch.nn.Embedding(max_len, d_model)
        self.layernorm = LayerNorm(d_model, eps=1e-12)
        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, mode=None, pos=None):
        batchSize = x.size(0)
        patchLen = x.size(1)
        if mode in ['img1', 'img2', 'img']:
            img1 = torch.LongTensor(batchSize, patchLen).fill_(pos)
            img_position = Variable(img1).cuda()
            type_pe = self.type_pe(img_position)
            pos_pe = Variable(self.pe[:,:patchLen], requires_grad=False).cuda()
            x = x + type_pe + pos_pe
        elif mode == 'text':
            # make embeddings relatively larger
            x = x * math.sqrt(self.d_model)
            text = torch.LongTensor(batchSize, patchLen).fill_(pos)
            text_position = Variable(text).cuda()
            type_pe = self.type_pe(text_position)
            pos_pe = Variable(self.pe[:,:patchLen], requires_grad=False).cuda()
            x = x + type_pe + pos_pe
        else: # [cls] or [diff] or [single] 
            CLS = torch.LongTensor(batchSize, 1).fill_(pos)
            cls_type =  Variable(CLS).cuda()
            type_pe = self.type_pe(cls_type)
            x = x + type_pe
        #layer norm & dropout
        x = self.layernorm(x)
        x = self.dropout(x)
        return x