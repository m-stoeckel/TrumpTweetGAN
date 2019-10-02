# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : run_relgan.py
# @Time         : Created at 2019-05-28
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import sys
from subprocess import call

import os

# Job id and gpu_id
if len(sys.argv) > 2:
    job_id = int(sys.argv[1])
    gpu_id = str(sys.argv[2])
    print('job_id: {}, gpu_id: {}'.format(job_id, gpu_id))
elif len(sys.argv) > 1:
    job_id = int(sys.argv[1])
    gpu_id = 0
    print('job_id: {}, missing gpu_id (use default {})'.format(job_id, gpu_id))
else:
    job_id = 0
    gpu_id = 0
    print('Missing argument: job_id and gpu_id. Use default job_id: {}, gpu_id: {}'.format(job_id, gpu_id))

# Executables
executable = 'python3'

# =====Program=====
if_test = int(False)
run_model = 'relgan'
CUDA = int(True)
if_real_data = int(True)
data_shuffle = int(False)
gen_init = 'normal'
dis_init = 'uniform'
oracle_pretrain = int(True)
gen_pretrain = int(True)
dis_pretrain = int(False)

# =====Oracle  or Real=====
dataset = 'trump_tweets_2009-2019'
train_data = 'dataset/' + dataset + '.txt'
test_data = 'dataset/testdata/' + dataset + '_test.txt'
model_type = 'vanilla'
loss_type = 'rsgan'
vocab_size = 20393
temp_adpt = 'exp'
temperature = 100

# =====Basic Train=====
samples_num = 10000
MLE_train_epoch = 150
ADV_train_epoch = 3000
batch_size = 64
max_seq_len = 20
gen_lr = 0.01
gen_adv_lr = 1e-4
dis_lr = 1e-4
pre_log_step = 10
adv_log_step = 20

# =====Generator=====
ADV_g_step = 1
gen_embed_dim = 32
gen_hidden_dim = 32
mem_slots = 1
num_heads = 2
head_size = 256

# =====Discriminator=====
ADV_d_step = 5
dis_embed_dim = 64
dis_hidden_dim = 64
num_rep = 64

# =====Run=====
rootdir = '../'
scriptname = 'main.py'
cwd = os.path.dirname(os.path.abspath(__file__))

args = [
    # Program
    '--if_test', if_test,
    '--run_model', run_model,
    '--dataset', dataset,
    '--train_data', train_data,
    '--test_data', test_data,
    '--if_real_data', if_real_data,
    '--model_type', model_type,
    '--loss_type', loss_type,
    '--cuda', CUDA,
    # '--device', gpu_id,   # comment for auto GPU
    '--shuffle', data_shuffle,
    '--gen_init', gen_init,
    '--dis_init', dis_init,

    # Basic Train
    '--samples_num', samples_num,
    '--vocab_size', vocab_size,
    '--mle_epoch', MLE_train_epoch,
    '--adv_epoch', ADV_train_epoch,
    '--batch_size', batch_size,
    '--max_seq_len', max_seq_len,
    '--gen_lr', gen_lr,
    '--gen_adv_lr', gen_adv_lr,
    '--dis_lr', dis_lr,
    '--pre_log_step', pre_log_step,
    '--adv_log_step', adv_log_step,
    '--temp_adpt', temp_adpt,
    '--temperature', temperature,
    '--ora_pretrain', oracle_pretrain,
    '--gen_pretrain', gen_pretrain,
    '--dis_pretrain', dis_pretrain,

    # Generator
    '--adv_g_step', ADV_g_step,
    '--gen_embed_dim', gen_embed_dim,
    '--gen_hidden_dim', gen_hidden_dim,
    '--mem_slots', mem_slots,
    '--num_heads', num_heads,
    '--head_size', head_size,

    # Discriminator
    '--adv_d_step', ADV_d_step,
    '--dis_embed_dim', dis_embed_dim,
    '--dis_hidden_dim', dis_hidden_dim,
    '--num_rep', num_rep,

    # Log
    '--tips', 'vanilla RelGAN',
]

args = list(map(str, args))
my_env = os.environ.copy()
call([executable, scriptname] + args, env=my_env, cwd=rootdir)
