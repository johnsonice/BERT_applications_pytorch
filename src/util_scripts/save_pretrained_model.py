#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 23:24:19 2020

@author: chengyu
"""

#https://huggingface.co/hfl

from transformers import AutoTokenizer, AutoModel
#%%
save_path = '../../data/pre_trained_model/chinese-bert-wwm'

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm")
model = AutoModel.from_pretrained("hfl/chinese-bert-wwm")
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

#%%
AutoTokenizer.from_pretrained(save_path)