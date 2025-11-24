#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Classification Dataset'''


# imports
import os

import cv2
import numpy as np
import pandas as pd
#    script imports
from src.aug_classification import get_classification_augmentations
from src.custom_defect_blackout import DefectBlackout
from src.utils.rle import rle_decode  # we will implement later
from torch.utils.data import Dataset

# imports


# constants
DATA_ROOT = '/kaggle/input/severstal-steel-defect-detection/'
# constants


# classes
class ClassificationDataset(Dataset):
  '''
  Returns (image_tensor, label)

  label = 1 if any defect is present (unless blackout removes all)
  '''

  def __init__(
    self,
    csv_path=os.path.join(DATA_ROOT, 'train.csv'),
    images_dir=os.path.join(DATA_ROOT, 'train_images'),
    train=True,
    pseudo_labels=None,
  ):
    self.train = train
    self.images_dir = images_dir

    df = pd.read_csv(csv_path)

    # group by image
    grouped = df.groupby('ImageId')['EncodedPixels'].apply(list)
    grouped_cls = df.groupby('ImageId')['ClassId'].apply(list)

    self.items = []
    for img_id in grouped.index:
      masks_rle = grouped[img_id]
      classes = grouped_cls[img_id]

      self.items.append({
        'image_id': img_id,
        'masks_rle': masks_rle,
        'classes': classes,
      })

    # include pseudo-labeled data if provided
    if pseudo_labels:
      self.items.extend(pseudo_labels)

    self.aug = get_classification_augmentations(train=train)
    self.blackout = DefectBlackout(p=0.5)

  def load_image(self, image_id):
    path = os.path.join(self.images_dir, image_id)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

  def decode_masks(self, masks_rle):
    masks = []
    for rle in masks_rle:
      if isinstance(rle, str):
        masks.append(rle_decode(rle, shape=(256, 1600)))
      else:
        masks.append(np.zeros((256, 1600), dtype=np.uint8))
    return masks

  def __len__(self):
    return len(self.items)

  def __getitem__(self, idx):
    item = self.items[idx]
    image_id = item['image_id']
    masks_rle = item['masks_rle']

    image = self.load_image(image_id)
    masks = self.decode_masks(masks_rle)

    # Apply Defect Blackout BEFORE augmentations
    if self.train:
      image, masks, label = self.blackout(image, masks)
    else:
      label = int(any(m.sum() > 0 for m in masks))

    # Albumentations requires all masks stacked
    stacked = np.stack(masks, axis=-1)

    augmented = self.aug(image=image, masks=[stacked])
    image_tensor = augmented['image']
    # label is scalar int
    return image_tensor, label

# classes
