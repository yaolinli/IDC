from collections import defaultdict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

MASK = 5
BOS = 1

class Model(nn.Module):
    def __init__(self, ff_dim, img_embs, n_hidden, n_head, n_block, se_block, de_block, vocab_size, dropout, max_len, CLS):
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

        self.word_embedding = nn.Embedding(self.vocab_size, n_hidden)
        self.img_project = nn.Sequential(nn.Linear(img_embs, n_hidden), nn.Sigmoid())
        self.position_encoding = PositionalEmb(n_hidden, dropout, 7)
        self.single_img_encoder = Transformer(n_embs=n_hidden, dim_ff=ff_dim, n_head=n_head, n_block=se_block, dropout=dropout)
        self.double_img_encoder = Transformer(n_embs=n_hidden, dim_ff=ff_dim, n_head=n_head, n_block=de_block, dropout=dropout)
        self.encoder = Transformer(n_embs=n_hidden, dim_ff=ff_dim, n_head=n_head, n_block=self.n_block, dropout=dropout)
        self.output_layer = nn.Linear(self.n_hidden, self.vocab_size)
        
    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        Img1 = Variable(batch['img1']).cuda()
        Img2 = Variable(batch['img2']).cuda()
        Cap = Variable(batch['cap']).cuda()
        Cap_label = Variable(batch['cap_label']).cuda()
        # convert token to word embedding & add position embedding
        text_embs = self.word_embedding(Cap)
        text_embs = self.position_encoding(text_embs, mode='text', pos=6)
        # reduce img feat dim & add img position embedding
        img_embs1 = self.position_encoding(self.img_project(Img1), mode='img1', pos=2)
        img_embs2 = self.position_encoding(self.img_project(Img2), mode='img2', pos=4)
        CLS = torch.tensor([[self.CLS]])
        CLS = CLS.repeat(Img1.size(0), 1)
        CLS = self.word_embedding(CLS.cuda())
        CLS1 = self.position_encoding(CLS, mode='cls', pos=1)
        CLS2 = self.position_encoding(CLS, mode='cls', pos=3)
        image_embs1 = torch.cat((CLS1, img_embs1), dim=1)
        image_embs2 = torch.cat((CLS2, img_embs2), dim=1)
        image_embs1 = self.single_img_encoder(image_embs1)
        image_embs2 = self.single_img_encoder(image_embs2)
        image_embs = self.double_img_encoder(torch.cat((image_embs1, image_embs2), dim=1))
        # concate [cls,i mg1,img2,text] as input to Transformer
        input_embs = torch.cat((image_embs, text_embs), dim=1)
        # input_embs = torch.cat((CLS, img_embs1, img_embs2, text_embs), dim=1)
        img_toklen = Img1.size(1)+1

        # input to Transformer
        att_mask = Variable(subsequent_mask(Cap.size(0), img_toklen, Cap.size(1))).cuda()
        output = self.encoder(input_embs, att_mask)
        text = output[:, (img_toklen*2):, :]
        # only compute masked tokens for better efficiency
        masked_output = text[Cap_label != -1].contiguous().view(-1, text.size(-1))
        # map hidden dim to vocab size
        prediction_scores = self.output_layer(masked_output)
        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                            Cap_label[Cap_label != -1].contiguous().view(-1),
                                            reduction='none')                            
            return masked_lm_loss, prediction_scores
        else:
            return prediction_scores
    
    @torch.no_grad()
    def greedy(self, Img1, Img2): 
        # Inference algorithm
        # print("this is greedy() function")
        Img1 = Variable(Img1).cuda()
        Img2 = Variable(Img2).cuda()
        batch_size = Img1.size(0)
        img_embs1 = self.position_encoding(self.img_project(Img1), mode='img1', pos=2)
        img_embs2 = self.position_encoding(self.img_project(Img2), mode='img2', pos=4)
        CLS = torch.tensor([[self.CLS]])
        CLS = CLS.repeat(Img1.size(0), 1)
        CLS = self.word_embedding(CLS.cuda())
        CLS1 = self.position_encoding(CLS, mode='cls', pos=1)
        CLS2 = self.position_encoding(CLS, mode='cls', pos=3)
        image_embs1 = torch.cat((CLS1, img_embs1), dim=1)
        image_embs2 = torch.cat((CLS2, img_embs2), dim=1)
        image_embs1 = self.single_img_encoder(image_embs1)
        image_embs2 = self.single_img_encoder(image_embs2)
        image_embs = self.double_img_encoder(torch.cat((image_embs1, image_embs2), dim=1))
        # to get img token k,v cache -> improve computation efficiency
        output = self.encoder(image_embs, step=0)
        # decoder process
        mask_token = Variable(torch.LongTensor([MASK]*batch_size).unsqueeze(1)).cuda() # MASK token: (batch size, 1)
        gen_tokens = Variable(torch.ones(batch_size, 1).long()).cuda() # input cap initialize: <BOS>
        for i in range(1, self.max_len):
            # each time input previous generated token + <MASK>    
            Des = torch.cat([gen_tokens, mask_token], dim=-1)
            text_embs = self.word_embedding(Des)
            text_embs = self.position_encoding(text_embs, mode='text', pos=6)
            step_attn_mask = Variable(decode_mask(batch_size, Img1.size(1)+1, i+1), requires_grad=False).cuda()
            out = self.encoder(text_embs, mask=step_attn_mask, step=i).squeeze()
            out = self.output_layer(out)
            prob = out[:,-1] # output prob
            _, next_w = torch.max(prob, dim=-1, keepdim=True)
            next_w = next_w.data
            gen_tokens = torch.cat([gen_tokens, next_w], dim=-1)
        return gen_tokens

def subsequent_mask(batch, img_size, cap_size):
    tot_size = img_size*2 + cap_size
    attn_mask = np.zeros((batch, tot_size, tot_size))
    # each cap token can attend <cls> + all img tokes
    # img tokens and attend each other
    attn_mask[:, :, :2*img_size] = 1
    # cap attn: each token can only attend the forward tokens
    cap_attn_shape = (batch, cap_size, cap_size)
    subsequent_mask = np.triu(np.ones(cap_attn_shape), k=1) == 0
    attn_mask = (attn_mask == 1)
    attn_mask[:, (2*img_size):, (2*img_size):] = subsequent_mask
    # change type
    attn_mask = attn_mask.astype('uint8')
    mask = torch.from_numpy(attn_mask) 
    return mask

def decode_mask(batch, img_size, cap_size):
    tot_size = img_size*2 + cap_size
    attn_mask = np.zeros((batch, cap_size, tot_size))
    # each cap token can attend <cls> + all img tokes
    attn_mask[:, :, :2*img_size] = 1
    attn_mask = (attn_mask == 1)
    # cap attn: each token can only attend the forward tokens
    cap_attn_shape = (batch, cap_size, cap_size)
    subsequent_mask = np.triu(np.ones(cap_attn_shape), k=1) == 0
    attn_mask[:, :, (2*img_size):] = subsequent_mask
    # change type
    attn_mask = attn_mask.astype('uint8')
    mask = torch.from_numpy(attn_mask) 
    return mask


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
                k = torch.cat((layer_cache['self_keys'], k), dim=2) # ([150, 8, 99+n, 64])
            else:
                layer_cache['self_keys'] = k

            if layer_cache['self_values'] is not None:
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