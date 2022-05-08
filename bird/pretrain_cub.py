'''
    Transformer based pretraining Model
'''
import os
import sys
import json
import time
import argparse
import numpy as np
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

# depend on dataset 
from para import parse_args


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
pretrain_tasks = args.pretrain_tasks # ['itm_a', 'itm_b', 'mlm', 'mvm']
# import dataset: diff dataset use diff function
if args.dataset == 'bird':
    import data.dataset_pretrain_bird as dataset
    import modules_pretrain_bird as modules
elif args.dataset == 'ImageEdit':
    import data.dataset_pretrain_edit as dataset
elif args.dataset == 'nlvr':
    import data.dataset_pretrain_nlvr as dataset
elif args.dataset == 'cub':
    import data.dataset_pretrain_cub as dataset
    import modules_pretrain_cub as modules

# use tensorboard
from torch.utils.tensorboard import SummaryWriter 
# random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print("seed", args.seed)

# load images
if args.dataset == 'nlvr':
    images = get_images_name(args.data_path)
else:
    images = load_images(args.data_path)
vocabs, rev_vocabs = load_vocabs(args.vocab_path)
vocab_size = len(vocabs)
# load split json files
datas = load_data(args.data_path, args.dataset, images)


if 'fda' in args.pretrain_tasks:
    neg_datas = load_neg_data_cub(args.neg_num, args.data_path, args.dataset)
    
if not os.path.exists('./experiments'):
    os.mkdir('./experiments')
out_path = os.path.join('experiments', args.exp_name)
if not os.path.exists(out_path):
    os.mkdir(out_path)
checkpoint_path = os.path.join(out_path, 'checkpoint') 
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path) 


train_logger = Logger(out_path, is_train=True)
val_logger = Logger(out_path, is_train=False)
# save configs to json
save_para(args, out_path + '/config.json')
# set tensorboard log file
if not os.path.exists(out_path+'/log/'):
    os.mkdir(out_path+'/log/')  
writer = SummaryWriter(out_path+'/log/')


def save_model(path, model, step, count):
    model_state_dict = model.state_dict()
    torch.save({
            'global_step': step,
            'model_state_dict': model_state_dict,
            'task_batch_num': count,
            }, path)

def get_dataset(task, images, split):
    if task == 'fda':
        input_datas = neg_datas[split]
    else:
        input_datas = datas[split]
    return dataset.Dataset( args,
                            task,
                            input_datas,
                            vocabs = vocabs,
                            rev_vocabs = rev_vocabs,
                            images = images,
                            split = split)

def get_dataloader(task, input_dataset, batch_size, is_train=True):
    if task == 'fda':
        batch_size = int(batch_size / (args.neg_num+1))
    return torch.utils.data.DataLoader(dataset=input_dataset, batch_size=batch_size, shuffle=is_train, num_workers=4, pin_memory=True)
    # return torch.utils.data.DataLoader(dataset=input_dataset, batch_size=batch_size, shuffle=is_train)

def train():
    # build data loaders
    train_set = {}
    train_batch = {}
    valid_set = {}
    valid_loader = {}
    batch_per_epo = defaultdict(int)
    for task in pretrain_tasks:
        train_set[task] = get_dataset(task, images, 'train')
        valid_set[task] = get_dataset(task, images, 'val')
        r = args.r[task]
        train_batch[task] = (get_dataloader(task, train_set[task], args.batch_size, is_train=True), r)
        valid_loader[task] = get_dataloader(task, valid_set[task], args.batch_size, is_train=False)
        batch_per_epo[task] = len(train_set[task]) // args.batch_size + 1
    train_loader = dataset.MetaLoader(train_batch)

    # Prepare model
    model = modules.Model(ff_dim = args.dim_ff,
                          img_embs = args.img_embs,
                          n_hidden = args.n_embs,
                          n_head = args.n_head,
                          n_block = args.n_block,
                          vocab_size = vocab_size,
                          dropout = args.dropout,
                          max_len = args.max_len,
                          tasks = pretrain_tasks,
                          CLS = vocabs['<CLS>'])

    global_step = -1
    count = defaultdict(int)
    # load checkpoint
    if args.restore != '':
        print("load parameters from {}".format(args.restore))
        checkpoint = torch.load(args.restore)
        model.load_state_dict(checkpoint['model_state_dict'])
        global_step = checkpoint['global_step']
        if 'task_batch_num' in list(checkpoint.keys()):
            count = checkpoint['task_batch_num']
            
    model.cuda()
    model.train()
    # Prepare optimizer
    optim = Optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr, weight_decay=args.weight_decay)
    #scheduler = Optim.lr_scheduler.StepLR(optim, step_size=4, gamma=0.9)

    best_score = defaultdict(int)
    report_loss = defaultdict(int)
    n_samples = defaultdict(int)
    use_time = defaultdict(float)
    
    for step, (task, batch) in enumerate(train_loader):
        global_step += 1
        start_time =  time.time()
        model.train()
        model.zero_grad()

        
        raw_loss, _ = model(batch, task, compute_loss=True)

        if args.l2_wd > 0:
            l2_loss = L2_SP(args, model, checkpoint['model_state_dict'])
            loss = raw_loss.mean() + l2_loss
        else:
            loss = raw_loss.mean()

        loss = raw_loss.mean()
        loss.backward()
        

        # learning rate scheduling
        
        lr_this_step = get_lr_sched(global_step, args)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        writer.add_scalar('lr', lr_this_step, global_step)
        

        optim.step()

        report_loss[task] += loss.item()
        n_samples[task] += batch['img'].size(0)
        count[task] += 1
        use_time[task] += time.time() - start_time

        # log batch loss to tensorboard
        writer.add_scalar(args.dataset + '_train_batch_loss_sum/' + task, raw_loss.sum().item()/batch['cap'].size(0), count[task])
        writer.add_scalar(args.dataset + '_train_batch_loss_mean/' + task, loss.item(), count[task])
        
        # log epoch loss for each task
        if count[task] % batch_per_epo[task] == 0:
            # report loss
            print('task: %s, epoch: %d, global_step: %d, report_loss: %.3f, time: %.2f'
                    % (task, count[task]//batch_per_epo[task], global_step, report_loss[task]/n_samples[task], use_time[task]))
            train_logger.print_train_stats(task, count[task]//batch_per_epo[task], global_step, report_loss[task]/n_samples[task], use_time[task])
            writer.add_scalar(args.dataset + '_train_epoch_loss/' + task, report_loss[task]/n_samples[task], count[task]//batch_per_epo[task])
            report_loss[task], n_samples[task], use_time[task] = 0, 0, 0.0
        

        # evaluating
        if global_step % args.valid_steps == 0:
            stats = validate(valid_loader, model, global_step, count)
            val_logger.print_eval_stats(global_step, stats)

        # save model
        if global_step > args.warmup_steps and global_step % 1000 == 0:
            save_model(os.path.join(checkpoint_path, 'checkpoint.pt'), model, global_step, count)
        # if global_step > args.warmup_steps and global_step % 10000 == 0:
        #     save_model(os.path.join(checkpoint_path, 'checkpoint_{}.pt'.format(global_step)), model, global_step, count)
        # print('Learning Rate ', optim.state_dict()['param_groups'][0]['lr'])
        if global_step >= args.total_train_steps:
            os.rename(os.path.join(checkpoint_path, 'checkpoint.pt'), os.path.join(checkpoint_path, 'checkpoint_{}.pt'.format(global_step)))
            break
    return 0


def validate(val_loader, model, step, count):
    print("Step {}: start validation ...".format(step))
    model.eval()
    start_time = time.time()
    results = {}
    with torch.no_grad():
        for task, loader in val_loader.items():
            if task.startswith('mlm'):
                val_log = validate_mlm(model, loader)
            elif task.startswith('mvm'):
                val_log = validate_mvm(model, loader, task)
            elif task.startswith('itm'):
                val_log = validate_itm(model, loader, task, count[task])
            elif task == 'fda':
                val_log = validate_fda(model, loader, task)
            else:
                raise ValueError('Undefined task {}'.format(task))
            results.update(val_log)
            # add to Tensorboard
            for k,v in val_log.items():
                if k.find('loss') == -1:
                    writer.add_scalar(args.dataset + '_validate/' + k, v, count[task])
                else:
                    writer.add_scalar(args.dataset + '_validate_loss/' + k, v, count[task])
    print("validate used total time: ", time.time()-start_time)
    model.train()
    return results

@torch.no_grad()
def validate_mlm(model, val_loader):
    print("start running MLM validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='mlm', compute_loss=False)
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

@torch.no_grad()
def validate_itm(model, val_loader, task, step):
    print("start running ITM validation...")
    val_loss = 0
    tot_score = 0
    n_ex = 0
    
    all_scores = []
    all_targets = []
    for i, batch in enumerate(val_loader):
        scores = model(batch, task=task, compute_loss=False)
        targets =  Variable(batch['align_label'], requires_grad=False).cuda().view(-1)
        loss = F.cross_entropy(scores, targets, reduction='sum')
        val_loss += loss.item()
        tot_score += (scores.max(dim=-1)[1] == targets).sum().item()
        n_ex += targets.size(0)
        all_scores.append(F.softmax(scores, dim=-1).cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    val_loss /= n_ex
    val_acc = tot_score / n_ex
    all_scores = np.concatenate(all_scores, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    if task == 'itm_a':
        val_log = {'itm_a loss': val_loss,
                    'itm_a acc': val_acc}
        # np.save(out_path + "/itm_a/scores_{}.npy".format(step), all_scores)
        # np.save(out_path + "/itm_a/targets_{}.npy".format(step), all_targets)
    else:
        val_log = {'itm_b loss': val_loss,
                    'itm_b acc': val_acc}
        # np.save(out_path + "/itm_b/scores_{}.npy".format(step), all_scores)
        # np.save(out_path + "/itm_b/targets_{}.npy".format(step), all_targets)
    return val_log

@torch.no_grad()
def validate_mvm(model, val_loader, task):
    print("start running MVM validation...")
    IMG_DIM = 2048 # img feat dim
    val_loss = 0
    n_feat = 0
    for i, batch in enumerate(val_loader):
        loss, _ = model(batch, task=task, compute_loss=True)
        if task == "mvm":
            val_loss += loss.sum().item() / IMG_DIM
        else:
            val_loss += loss.sum().item()
        n_feat += Variable(batch['img_mask'], requires_grad=False).cuda().sum().item()
    val_loss /= n_feat
    val_log = {'mvm loss': val_loss}
    return val_log

@torch.no_grad()
def validate_fda(model, val_loader, task):
    print("start running FDA validation...")
    val_loss = 0
    num = 0
    avg_neg_sim = 0
    avg_pos_sim = 0
    avg_acc = 0
    for i, batch in enumerate(val_loader):
        loss, [pos_sim, neg_sim, acc] = model(batch, task=task, compute_loss=True)
        val_loss += loss.item()
        avg_neg_sim += neg_sim
        avg_pos_sim += pos_sim
        avg_acc += acc
    val_loss /= len(val_loader)
    avg_neg_sim /= len(val_loader)
    avg_pos_sim /= len(val_loader)
    avg_acc /= len(val_loader)
    val_log = {'fda loss': val_loss,
               'avg_neg_sim': avg_neg_sim,
               'avg_pos_sim': avg_pos_sim,
               'avg_acc': avg_acc}
    return val_log


if __name__ == '__main__':
    print("==> Pretraining stage")
    print("==> dataset: ", args.dataset)
    print("==> Pretrain task: ", pretrain_tasks)
    train()