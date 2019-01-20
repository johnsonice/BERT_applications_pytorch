#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 14:03:04 2019

@author: chengyu
"""

class EarlyStopping():
    """
    Early Stopping to terminate training early under certain conditions
    """

    def __init__(self, 
                 min_delta=0.001,
                 patience=3):
        """
        EarlyStopping callback to exit the training loop if training or
        validation loss does not improve by a certain amount for a certain
        number of epochs
        Arguments
        ---------
        min_delta : float
            minimum change in monitored value to qualify as improvement.
            This number should be positive.
        patience : integer
            number of epochs to wait for improvment before terminating.
            the counter be reset after each improvment
        """
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e15
        self.current_loss = 1e15
        self.stopped_epoch = 0
        self.stop_training = False

    def eval_loss(self, epoch, loss):
        self.current_loss = loss
        if (self.current_loss - self.best_loss) < -self.min_delta:
            self.best_loss = loss
            self.wait = 1
            print("ok, epoch{}".format(epoch))
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                print('\nTerminated Training for Early Stopping at Epoch %04i' % 
                      (self.stopped_epoch))
            self.wait += 1

        
            
#%%
            
def loss_generator(n_epoch):
    
    return 100 - n_epoch*0.8**(n_epoch)


es = EarlyStopping(min_delta=0.0,patience=3)

#%%
for epoch in range(100):
    loss = loss_generator(epoch)
    es.eval_loss(epoch,loss)
    print(es.stop_training,es.current_loss,es.best_loss,es.wait)
    if es.stop_training:
        break

#%%

