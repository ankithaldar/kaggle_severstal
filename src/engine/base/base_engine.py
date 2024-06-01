#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Base Engine for model training'''


# imports
import tensorflow as tf
#    script imports
# imports


# constants
# constants


# classes
class BaseEngine:
  '''Base Engine for model training'''

  def __init__(self, hparams):
    self.hparams = hparams

    self._init_score_function()
    self._init_augmentation()

    self._init_train_datalader()
    self._init_valid_dataloader()
    self._init_test_dataloader()

    self._init_model()
    self._init_loss_function()
    self._init_metrics()

    self.setup()


  def _init_score_function(self):
    # look into this from kaggle-birdsong-recognition repo
    pass


  def _init_augmentation(self):
    self.tfms = None


  def _init_train_dataloader(self):
    self.train_ds = None

  def _init_valid_dataloader(self):
    self.valid_ds = None


  def _init_test_dataloader(self):
    self.test_ds = None


  def _init_model(self):
    raise NotImplementedError


  def _init_loss_function(self):
    raise NotImplementedError


  def _init_metrics(self):
    raise NotImplementedError


  def setup(self):
    self._init_distribution()


  def _init_distribution(self):
    # Distributed training with TensorFlow
    # Distributed training with Keras
    # check tensorflow mixed precision

    # set all seeds
    tf.keras.utils.set_random_seed(self.hparams.seed)


# classes


# functions
def function_name():
  pass
# functions


# main
def main():
  pass


# if main script
if __name__ == '__main__':
  main()
