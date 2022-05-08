import argparse
import torch
import os
import json
import sys
'''
use config json file
e.g. args = parse_with_config(parser)
'''
def parse_with_config(parser):
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:] if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args


def parse_args():   
    parser = argparse.ArgumentParser(description='train.py')
    # hyperparameter
    parser.add_argument('--n_embs', type=int, default=512, help='Embedding size')
    parser.add_argument('--img_embs', type=int, default=2048, help='Image embedding size')
    parser.add_argument('--dim_ff', type=int, default=512, help='Feed forward hidden size')
    parser.add_argument('--n_head', type=int, default=8, help='head number')
    parser.add_argument('--n_block', type=int, default=4, help='block number')
    parser.add_argument('--se_block', type=int, default=1, help='single image encoder block number')
    parser.add_argument('--de_block', type=int, default=1, help='double image encoder block number')
    parser.add_argument('--neg_num', type=int, default=6) 
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout Rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--l2_wd', type=float, default=0.0, help='l2 norm weight decay')
    parser.add_argument('--max_len', type=int, default=100, help='Max length of captions')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--beam_size', type=int, default=1, help='Beam Size')
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--report', type=int, default=200)
    parser.add_argument('--warmup_steps', type=int, default=4000) # small data 4000; big data 8000
    parser.add_argument('--valid_steps', type=int, default=220)
    parser.add_argument('--total_train_steps', type=int, default=10000) # around 45 epoch
    
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--vocab_path', type=str, default='')
    parser.add_argument('--dataset', type=str, required=True, default='bird')
    parser.add_argument('--main_metric', type=str, default='CIDEr') #or 'ROUGE_L' or 'METEOR'
    parser.add_argument('--restore', type=str, default='', help="Pretraining model path")
    parser.add_argument('--best_ckpt', type=str, default='', help="checkpoint path of test stage")
    parser.add_argument('--gpu_id', type=str, default='3', help='gpu id')
    parser.add_argument('--exp_name', type=str, required=True, default='spot')
    parser.add_argument('--out_file', type=str, default='result.json', help = 'result for evaluation')
    parser.add_argument('--para_file', type=str, default='para.py')
    parser.add_argument('--config', type=str, required=True, default='./config/bird.json')
    parser.add_argument('--mode', type=str, default='', help='train or test mode') 
    parser.add_argument('--contrast_loss', type=int, default=1) 
    parser.add_argument('--set_type', type=str, default='', help='load multiple or one sent per data in valid') # 'P' means one sent per data
    parser.add_argument('--neg_name', type=str, default='neg_tfidf6', help='neg samples file for fda task') 

    # merge config json file to args
    args = args = parse_with_config(parser) 

    return args


# args = parse_args()
# print(args)
# with open("test.json", "w", encoding="utf-8") as f:
#     args = json.dumps(vars(args), indent=4)
#     f.write(args)
