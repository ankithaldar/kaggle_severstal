#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Load Model Check point'''


# imports
from pytorch_lightning.callbacks import ModelCheckpoint

#    script imports
# imports


# constants
# constants


# functions
def get_classification_checkpoint(save_dir):
  '''
  Saves best model according to validation AUC.
  '''
  return ModelCheckpoint(
    dirpath=save_dir,
    save_top_k=1,
    monitor='val_auc',
    mode='max',
    filename='classifier-{epoch:02d}-{val_auc:.4f}'
  )
# functions
