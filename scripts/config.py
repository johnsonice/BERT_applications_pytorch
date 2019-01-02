#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 13:55:44 2019

@author: chengyu
"""

data_dir='../data/classification_datasets/sst2'                ## where to load training and eval dataset
train_data_file_name = 'train_orig.txt'
test_data_file_name = 'test.txt'
bert_model='bert-base-uncased'                  ## "Bert pre-trained model selected in the list: bert-base-uncased, "
                                                ##"bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese."
task_name = 'sst2'                              ## The name of the task to train.
model_output_dir = '../models/sst2'             ## model output folder path
result_out_dir = '../results/sst2'              ## results output folder path
max_seq_length = 128                            ## max total input sequence length after tokenization
do_train = True                                 ## whether to run training
do_eval = True                                  ## whether to run evaluation
do_lower_case = True                            ## Set this flag if you are using an uncased model.
train_batch_size = 32                           ## Total batch size for training.
eval_batch_size = 8                             ## Total batchsize for eval.
learning_rate = 5e-5                            ## The initial learning rate for Adam.
num_train_epochs = 5.0                          ## total number of training epochs to perform
warmup_proportion = 0.1                         ## Proportion of training to perform linear learning rate warmup for
no_cuda = False                                 ## Whether not to sue CUDA when available
local_rank = -1                                 ## local_rank for distributed training on gpus
seed = 42                                       ## random seed for initialization
gradient_accumulation_steps=1                   ## Number of updates steps to accumulate before performing a backward/update pass.
fp16 = False                                    ## Whether to use 16-bit float precision instead of 32-bit
loss_scale = 0                                  ## Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.

train_batch_size = int(train_batch_size / gradient_accumulation_steps) 
