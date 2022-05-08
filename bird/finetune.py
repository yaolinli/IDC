'''
    Transformer based pretraining Model
'''

import os
import sys
import json
import time
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision
import torch.optim as Optim
from torch.autograd import Variable
import torch.nn.functional as F
from utils import *
from eval import Evaluator

import pdb
import shutil
import random
from tqdm import tqdm

from para import parse_args


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
# import dataset: diff dataset use diff function
if args.dataset == 'bird':
    import data.dataset_finetune_bird as dataset
    import modules_finetune_bird as modules
elif args.dataset == 'cub':
    import data.dataset_finetune_cub as dataset
    import modules_finetune_cub as modules

# use tensorboard
from torch.utils.tensorboard import SummaryWriter 
# random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print("seed", args.seed)


# load images
images = load_images(args.data_path)
vocabs, rev_vocabs = load_vocabs(args.vocab_path)
vocab_size = len(vocabs)

if not os.path.exists('./experiments'):
    os.mkdir('./experiments')
out_path = os.path.join('experiments', args.exp_name)
if not os.path.exists(out_path):
    os.mkdir(out_path)
checkpoint_path = os.path.join(out_path, 'checkpoint') 
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path) 

if args.mode == 'train':
    val_sent_path = os.path.join(out_path, 'val_sent') 
    if not os.path.exists(val_sent_path):
        os.mkdir(val_sent_path)  
    train_logger = Logger(out_path, is_train=True)
    val_logger = Logger(out_path, is_train=False)
    # save configs to json
    save_para(args, out_path + '/config.json')
    # set tensorboard log file
    if not os.path.exists(out_path+'/log/'):
        os.mkdir(out_path+'/log/')  
    writer = SummaryWriter(out_path+'/log/')

def save_model(path, model):
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, path)

def get_dataset(data_path, images, split, set_type=None):
    return dataset.Dataset( args,
                            data_path,
                            vocabs = vocabs,
                            rev_vocabs = rev_vocabs,
                            images = images,
                            split = split,
                            set_type = set_type
                            )

def get_dataloader(input_dataset, batch_size, is_train=True, self_define=False):
    # use torch default dataloader function
    if not self_define:
        loader = torch.utils.data.DataLoader(dataset=input_dataset, batch_size=batch_size, shuffle=is_train)
    # use self defined dataloader func to get gts/imgID list batch
    else:
        loader = dataset.DataLoader(dataset=input_dataset, batch_size=batch_size, shuffle=is_train)
    return loader


def train():
    # build data loaders
    train_set = get_dataset(os.path.join(args.data_path, 'train.json'), images, 'train')
    valid_set_P = get_dataset(os.path.join(args.data_path, 'val.json'), images, 'val', 'P') 
    valid_set = get_dataset(os.path.join(args.data_path, 'val.json'), images, 'val') 
    
    train_loader = get_dataloader(train_set, args.batch_size, is_train=True)
    valid_loader_P = get_dataloader(valid_set_P, args.batch_size, is_train=False)
    valid_loader = get_dataloader(valid_set, args.batch_size, is_train=False, self_define=True)

    test_set = get_dataset(os.path.join(args.data_path, 'test.json'), images, 'test')
    test_loader = get_dataloader(test_set, args.batch_size, is_train=False, self_define=True)

    # Prepare model
    model = modules.Model(ff_dim = args.dim_ff,
                          img_embs = args.img_embs,
                          n_hidden = args.n_embs,
                          n_head = args.n_head,
                          n_block = args.n_block,
                          se_block = args.se_block,
                          de_block = args.de_block,
                          vocab_size = vocab_size,
                          dropout = args.dropout,
                          max_len = args.max_len,
                          CLS = vocabs['<CLS>'])

    # load checkpoint
    if args.restore != '':
        print("load parameters from {}".format(args.restore))
        checkpoint = torch.load(args.restore)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("load checkpoint finished")

    model.cuda()
    # Prepare optimizer
    if args.se_block > 1:
        for k,v in model.named_parameters():
            if ('single_img_encoder.layers.' in k) and (str(args.se_block-1) not in k):
                v.requires_grad=False  #固定single encoder和double encoder除了最后一层外的参数
    if args.de_block > 1:
        for k,v in model.named_parameters():
            if ('double_img_encoder.layers.' in k) and (str(args.de_block-1) not in k):
                v.requires_grad=False  #固定double encoder和double encoder除了最后一层外的参数
    optim = Optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr, weight_decay=args.weight_decay)
    best_score = 0
    no_beat = 0
    global_step = -1
    while True:
        report_loss, start_time, n_samples = 0, time.time(), 0
        for step, batch in tqdm(enumerate(train_loader)):
            global_step += 1
            model.train()
            model.zero_grad()
            loss, _ = model(batch, compute_loss=True)
            if args.l2_wd > 0:
                l2_loss = L2_SP(args, model, pretrained_dict)
                loss = loss.sum() + l2_loss
            else:
                loss = loss.sum() 
            loss.backward()

            # learning rate scheduling
            lr_this_step = get_lr_sched(global_step, args, pretrain=False)
            for param_group in optim.param_groups:
                param_group['lr'] = lr_this_step

            # learning rate keep unchanged lr = 5e-5
            writer.add_scalar('lr', optim.state_dict()['param_groups'][0]['lr'], global_step)
            optim.step()

            report_loss += loss.data.item()
            n_samples += len(batch['img1'].data)

            # evaluating and logging
            if global_step > 0 and global_step % args.report == 0:
                # report train loss & writing to log & add to tensorboard 
                print('global_step: %d, epoch: %d, report_loss: %.3f, time: %.2f'
                        % (global_step, global_step//len(train_loader), report_loss / n_samples, time.time() - start_time))
                train_logger.print_train_stats('mlm', global_step//len(train_loader), global_step, report_loss / n_samples, time.time() - start_time)
                writer.add_scalar(args.dataset + '_train_loss', report_loss/n_samples, global_step)
                # calculate validate loss + word acc
                stats = validate(valid_loader_P, model, global_step)
                # Inference algorithm & calculate main metric
                scores = Inference(valid_loader, test_loader, model, global_step)
                score = scores[args.main_metric] 
                model.train()
                if global_step > 2000 and score > best_score:
                    no_beat = 0
                    best_score = score
                    print('Score Beat ', score, '\n')
                    save_model(os.path.join(checkpoint_path, 'best_checkpoint.pt'), model)
                else:
                    no_beat += 1
                    # save_model(os.path.join(checkpoint_path, 'checkpoint_{}.pt'.format(global_step)), model)
                    print('Term ', no_beat, 'Best Term', best_score, '\n')
                # add main metric score to log
                stats.update(scores)
                stats['main_metric_best'] = best_score
                val_logger.print_eval_stats(global_step, stats, no_beat)
                for k,v in stats.items():
                    writer.add_scalar(args.dataset + '_validate/' + k, v, global_step)

                # early stop
                if no_beat == args.early_stop:
                    test()
                    sys.exit()
                report_loss, start_time, n_samples = 0, time.time(), 0
            if global_step > args.total_train_steps:
                test()
                sys.exit()
        # print('Learning Rate ', optim.state_dict()['param_groups'][0]['lr'])
    return 0

def validate(loader, model, global_step):
    print('Starting Evaluation...')
    model.eval()
    val_loss = 0
    n_correct = 0
    n_word = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            scores = model(batch, compute_loss=False)
            labels = Variable(batch['cap_label'], requires_grad=False).cuda()
            labels = labels[labels != -1]
            loss = F.cross_entropy(scores, labels, reduction='sum')
            val_loss += loss.item()
            n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
            n_word += labels.numel()
    val_loss /= n_word
    acc = n_correct / n_word
    val_log = {'mlm loss': val_loss,
                'mlm acc': acc}
    return val_log

def Inference(loader, test_loader, model, global_step):
    print('Starting Inference...')
    start_time = time.time()
    model.eval()
    candidates = {}
    references = {}
    cand_lists = []
    ref_lists = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            img1, img2, gts, ImgID = batch
            ids = model.greedy(img1, img2).data.tolist()
            for i in range(len(img1.data)):
                id = ids[i]
                sentences = transform(id)
                # show generate sent case
                if bi == 0 and i < 3:
                    print(' '.join(sentences))

                candidates[ImgID[i]] = [' '.join(sentences)]
                references[ImgID[i]] = gts[i]
                cand_lists.append(' '.join(sentences))
                ref_lists.append(gts[i])

    evaluator = Evaluator(references, candidates)
    score = evaluator.evaluate()

    # "ROUGE_L", "CIDEr"
    print(args.main_metric, score[args.main_metric])
    print("evaluting time:", time.time() - start_time)
    # save val generate sentences
    with open(os.path.join(val_sent_path, 'iter_{}.json'.format(global_step)), 'w', encoding='utf-8') as fout:
        for ImgID in candidates.keys():
            sample = {'ImgId': ImgID,
                      'candidates': candidates[ImgID]
                      # 'references': references[ImgID]
                    } 
            jterm = json.dumps(sample, ensure_ascii=False)
            fout.write(jterm + '\n')
    
    # evaluate for test
    candidates = {}
    references = {}
    cand_lists = []
    ref_lists = []
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            img1, img2, gts, ImgID = batch
            ids = model.greedy(img1, img2).data.tolist()
            for i in range(len(img1.data)):
                id = ids[i]
                sentences = transform(id)
                candidates[ImgID[i]] = [' '.join(sentences)]
                references[ImgID[i]] = gts[i]
                cand_lists.append(' '.join(sentences))
                ref_lists.append(gts[i])
    test_score = {}
    evaluator = Evaluator(references, candidates)
    test_score = evaluator.evaluate()

    for key in test_score.keys():
        score['test_'+key] = test_score[key]
    return score



def test():
    test_set = get_dataset(os.path.join(args.data_path, 'test.json'), images, 'test')
    test_loader = get_dataloader(test_set, args.batch_size, is_train=False, self_define=True)
    # Prepare model
    model = modules.Model(ff_dim = args.dim_ff,
                          img_embs = args.img_embs,
                          n_hidden = args.n_embs,
                          n_head = args.n_head,
                          n_block = args.n_block,
                          se_block = args.se_block,
                          de_block = args.de_block,
                          vocab_size = vocab_size,
                          dropout = args.dropout,
                          max_len = args.max_len,
                          CLS = vocabs['<CLS>'])

    if args.best_ckpt == '':
        print(" ==> test stage load checkpoint from ", os.path.join(out_path, 'checkpoint/best_checkpoint.pt'))
        model_dict = torch.load(os.path.join(out_path, 'checkpoint/best_checkpoint.pt'))
    else:
        print(" ==> test stage load checkpoint from ", args.best_ckpt)
        model_dict = torch.load(args.best_ckpt)
    model.load_state_dict({k.replace('module.', ''):v for k,v in model_dict.items()})
    model.cuda()
    model.eval()

    candidates = {}
    references = {}
    cand_lists = []
    ref_lists = []
    with torch.no_grad():
        with open(os.path.join(out_path, args.out_file), 'w', encoding='utf-8') as fout:
            for i, batch in enumerate(test_loader):
                img1, img2, gts, ImgID = batch
                ids = model.greedy(img1, img2).data.tolist()

                for i in range(len(img1.data)):
                    id = ids[i]
                    sentences = transform(id)
                    candidates[ImgID[i]] = [' '.join(sentences)]
                    references[ImgID[i]] = gts[i]
                    cand_lists.append(' '.join(sentences))
                    ref_lists.append(gts[i])
                    sample = {'ImgId': ImgID[i],
                            'candidates': [' '.join(s for s in sentences)]
                            #'references': Caps
                            }
                    
                    jterm = json.dumps(sample, ensure_ascii=False)
                    fout.write(jterm + '\n')
    evaluator = Evaluator(references, candidates)
    score = evaluator.evaluate()
    print(args.main_metric, score[args.main_metric])


def val():
    val_set = get_dataset(os.path.join(args.data_path, 'val.json'), images, 'val')
    val_loader = get_dataloader(val_set, args.batch_size, is_train=False, self_define=True)

    # Prepare model
    model = modules.Model(ff_dim = args.dim_ff,
                          img_embs = args.img_embs,
                          n_hidden = args.n_embs,
                          n_head = args.n_head,
                          n_block = args.n_block,
                          se_block = args.se_block,
                          de_block = args.de_block,
                          vocab_size = vocab_size,
                          dropout = args.dropout,
                          max_len = args.max_len,
                          CLS = vocabs['<CLS>'])
    if args.best_ckpt == '':
        print(" ==> test stage load checkpoint from ", os.path.join(out_path, 'checkpoint/best_checkpoint.pt'))
        model_dict = torch.load(os.path.join(out_path, 'checkpoint/best_checkpoint.pt'))
    else:
        print(" ==> test stage load checkpoint from ", args.best_ckpt)
        model_dict = torch.load(args.best_ckpt)
    model.load_state_dict({k.replace('module.', ''):v for k,v in model_dict.items()})
    model.cuda()
    model.eval()
    candidates = {}
    references = {}
    with torch.no_grad():
        with open(os.path.join(out_path, "val_result.json"), 'w', encoding='utf-8') as fout:
            for bi, batch in enumerate(val_loader):
                img1, img2, gts, ImgID = batch
                ids = model.greedy(img1, img2).data.tolist()

                for i in range(len(img1.data)):
                    id = ids[i]
                    sentences = transform(id)

                    candidates[ImgID[i]] = [' '.join(sentences)]
                    references[ImgID[i]] = gts[i]

                    sample = {'ImgId': ImgID[i],
                            'candidates': [' '.join(s for s in sentences)]
                            #'references': Caps
                            }
                    
                    jterm = json.dumps(sample, ensure_ascii=False)
                    fout.write(jterm + '\n')
    evaluator = Evaluator(references, candidates)
    score = evaluator.evaluate()
    print(args.main_metric, args.main_metric)

def transform(ids):
    sentences = []
    for wid in ids:
        if wid == vocabs['<BOS>']:
            continue
        if wid == vocabs['<EOS>']:
            break
        sentences.append(rev_vocabs[wid])
    return sentences


if __name__ == '__main__':
    if args.mode == 'train':
        print('------Train Mode------')
        train()
        test()
    elif args.mode == 'test':
        print('------Test  Mode------')
        test()
    elif args.mode == 'val':
        print('------Valid  Mode------')
        val()
    else:
        print('Please enter the mode')