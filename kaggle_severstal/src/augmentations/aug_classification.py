#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Augmentation pipeline for classification stage.'''


# imports
import albumentations as A
from albumentations.pytorch import ToTensorV2

#    script imports
# imports


# constants
# constants


# functions
def get_classification_augmentations(train: bool = True):
  '''
  Augmentation pipeline for classification stage.

  Cropping must happen BEFORE other augmentations, because the crop affects
  whether defects are visible.
  '''
  if train:
    return A.Compose([
      A.RandomCrop(height=224, width=1568),
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.5),
      A.RandomBrightnessContrast(p=0.3),
      ToTensorV2(),
    ])
  else:
    # For validation we do NOT crop — the dataloader will supply crops or full images.
    return A.Compose([
      ToTensorV2()
    ])

# functions
