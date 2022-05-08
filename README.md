# Image Difference Captioning with Pre-training and Contrastive Learning

This repository is the official implementation of [Image Difference Captioning with Pre-training and Contrastive Learning](https://arxiv.org/abs/2202.04298) in AAAI2022.


The Image Difference Captioning(IDC) task aims to describe the visual differences between two similar images with natural language. In this work, we propose a new  framework following the pre-training and fine-tuning paradigm for IDC. Specifically, we design three self-supervised tasks with contrastive learning strategies to align visual differences and text descriptions at a fine-grained level. Moreover, we propose a data expansion strategy to utilize extra cross-task supervision information, such as data for fine-grained image classification, to alleviate the limitation of available supervised IDC data.


![model](https://user-images.githubusercontent.com/24662157/165236680-ade4d5f2-3e49-41d5-a5de-91882aad9389.png)



## Installation

```
conda create --name IDC python=3.6
conda activate IDC
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```



## Data Download

We provide the pre-processed image features (by pre-trained ResNet101) , the annotations and the constructed negative data samples of CLEVR-Change and Birds-to-Words dataset  in  [baidu cloud]() .

You should put the files under the corresponding`./clver`  or  `./bird`  folder as follows:

```
clver
├── dataset_clver

bird
├── dataset
    ├── bird
    ├── cub
    └── nabirds
```



## CLEVR-Change dataset 

`cd  ./clver`

#### Pre-training

```
python3.6 pretrain.py --dataset clver --gpu_id 3 \
--exp_name pretrain_clver_neg_tfidf6_t1.0 \
--config ./config/pretrain_clver.json \
--total_train_steps 250000 \
--tmp 1.0 
```

[Note] All settable parameters are explained in `para.py`

**(Optional) View logs via tensorboard**

```
tensorboard --logdir=./experiments/pretrain_clver_neg_tfidf6_t1.0/log --host=0.0.0.0  --port=8080
```

#### Fine-tuning

```
python3.6 finetune.py --mode train --dataset clver --gpu_id 0 \
--exp_name finetune_clver_neg_tfidf6_t1.0 \
--config ./config/finetune_clver.json \
--restore ./experiments/pretrain_clver_neg_tfidf6_t1.0/checkpoint/checkpoint_250000.pt 
```

#### Inference & Evaluation

```
python3.6 finetune.py --mode test --dataset clver --gpu_id 0 \
--exp_name finetune_clver_neg_tfidf6_t1.0 \
--config ./config/finetune_clver.json


cd ../eval
python3.6 eval_models.py --dataset clevr \
--testfile  ../clver/experiments/finetune_clver_neg_tfidf6_t1.0/results.json \
--gtfile ../clver/dataset_clver/test.json
```

We also provide the pre-trained and fine-tuned checkpoints at [baidu yun (password: 0b07)](https://pan.baidu.com/s/1F3hxERJQZT_1MUICqDxJsQ). The  reported results on CLEVR-Change dataset are as follows:

| Dataset      | BLEU4 | METEOR | ROUGE-L | CIDEr |
| ------------ | ----- | ------ | ------- | ----- |
| CLEVR-Change | 51.2  | 36.2   | 71.7    | 128.9 |



## Birds-to-Words dataset 

`cd  ./bird`

#### Pre-training

We adopt cross-task data expansion strategy  on  Birds-to-Words dataset to provide additional in-domain knowledge.  Specifically, we utilize extra data from general image captioning (GIC), that is the CUB dataset, and Fine-grained visual classification (FGVC), that is the NABirds dataset. 

(img)

```
# Stage 1: training with CUB dataset
python3.6 pretrain_cub.py --dataset cub --exp_name pretrain_cub  --gpu_id 0 --config ./config/pretrain_cub.json 


# Stage 2: training with Birds-to-Words and NABirds dataset alternately
python3.6 pretrain.py --dataset bird --exp_name pretrain_cub_nabirds_bird  --gpu_id 3 --config ./config/pretrain_bird_nabirds.json --restore ./experiments/pretrain_cub/checkpoint/checkpoint_60000.pt
```

#### Fine-tuning

```
python3.6 finetune.py --dataset bird --exp_name finetune_bird \
--mode train --gpu_id 3 --config ./config/finetune_bird.json \
--restore experiments/pretrain_cub_nabirds_bird/checkpoint/checkpoint_60000.pt --batch_size 32
```

#### Inference & Evaluation

```
python3.6 finetune.py --mode test --dataset bird --gpu_id 0 \
--exp_name finetune_bird \
--config ./config/finetune_bird.json 

cd ../eval
python3.6 eval_models.py --dataset bird \
--testfile ../bird/experiments/finetune_bird/result.json  \
--gtfile ../bird/dataset/bird/test_self.json
```

We also provide the pre-trained and fine-tuned checkpoints at [baidu yun (password: )](). The  reported results on Birds-to-Words  dataset are as follows:

| Dataset        | BLEU4 | METEOR | CIDEr-D | ROUGE-L |
| -------------- | ----- | ------ | ------- | ------- |
| Birds-to-Words | 31.0  | 23.4   | 25.3    | 49.1    |



## Citation

```
@article{Yao2022ImageDC,
  title={Image Difference Captioning with Pre-training and Contrastive Learning},
  author={Linli Yao and Weiying Wang and Qin Jin},
  journal={ArXiv},
  year={2022},
  volume={abs/2202.04298}
}
```


