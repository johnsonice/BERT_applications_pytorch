#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 10:56:52 2019

@author: chengyu
"""

## run an example 

import sys
sys.path.append('./libs')
import os 
import torch
from classification_model import set_device,set_seed,file_checking,BertTokenizer
from classification_model import load_training_data_model,set_optimizer,train_model,evel_model
from classification_model import BertForSequenceClassification

from utils import EarlyStopping,DataProcessor,InputExample
import config as args
#%%
#### since data can come in different format
#### you can defined your own data loader 
#### for example 
class SST2_Processor(DataProcessor):
    ''' process ss1 data '''
    
    def _read_txt(self,data_path):
        lines = open(data_path, 'r').readlines()
        return lines
    
    def get_train_examples(self, data_dir,file_name=args.train_data_file_name):
        """See base class."""
        #logger.info("LOOKING AT {}".format(os.path.join(data_dir, file_name)))
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, file_name)), "train")

    def get_dev_examples(self, data_dir,file_name=args.dev_data_file_name):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, file_name)), "dev")
    
    def get_test_examples(self, data_dir,file_name=args.test_data_file_name):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, file_name)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]  ## if multi class, change this 

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line = line.split('\t')
            text_a = line[1]
            text_b = None
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

if __name__ == "__main__":
    ## if you have multiple tasks, you can put all of them in a dictorinay
    processors = {
        "sst2": SST2_Processor,
    }
    num_labels_task = {
        "sst2": 2,  ## if it is multi classification, you need to change this to n_class
    }   
    
    ## check task is already implemented, and setup data processor 
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    device,n_gpu = set_device(args)
    set_seed(args.seed,n_gpu)
    file_checking(args)        
    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()
    ## read bert tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    ## load data and model 
    train_examples,num_train_steps,model=load_training_data_model(args,device,processor,num_labels,n_gpu)
    ## set optimizers 
    optimizer,t_total = set_optimizer(model,num_train_steps)  # t_total = number of steps
    
    ## train model 
    if args.do_train:
        es = EarlyStopping(min_delta=args.min_delta,patience=args.patience)
        model,tr_loss,nb_tr_steps,global_step = train_model(args,processor,train_examples,label_list,
                                                            tokenizer,model,device,n_gpu,
                                                            optimizer,num_train_steps,t_total,
                                                            early_stoping_obj = es,
                                                            eval_dev = True)
        
        # save trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.model_output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
    else:
        tr_loss = -1
        nb_tr_steps = -1
        global_step = -1 
        
    # Load a trained model that you have fine-tuned
    if args.do_eval:
        output_model_file = os.path.join(args.model_output_dir, "pytorch_model.bin")
        model_state_dict = torch.load(output_model_file)
        model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict)
        model.to(device)
        
        # evaluate model and export results, if  you don't want to export_results = False
        results =  evel_model(args,processor,label_list,tokenizer,model,device,optimizer,
                              tr_loss,nb_tr_steps,global_step,export_results=True)

    print('Done')
##       results looks something like this 
#        result = {'eval_loss': eval_loss,
#                  'eval_accuracy': eval_accuracy,
#                  'global_step': global_step,
#                  'loss': tr_loss}


