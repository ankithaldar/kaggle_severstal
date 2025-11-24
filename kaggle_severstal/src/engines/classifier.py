#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''All model classifier'''


# imports
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule
from src.models.classification_models import (EfficientNetB1Classifier,
                                              ResNet34Classifier)

#    script imports

# imports


# constants
# constants


# classes
class LitClassifier(LightningModule):

  def __init__(
    self,
    model_name='effnet_b1',
    lr=0.01,
    weight_decay=1e-4,
    momentum=0.9
  ):
    super().__init__()
    self.save_hyperparameters()

    if model_name == 'effnet_b1':
      self.model = EfficientNetB1Classifier()
    elif model_name == 'resnet34':
      self.model = ResNet34Classifier()
    else:
      raise ValueError(f'Unsupported model: {model_name}')

    self.loss_fn = nn.BCEWithLogitsLoss()

    # metrics
    self.train_acc = torchmetrics.Accuracy(task='binary')
    self.val_acc = torchmetrics.Accuracy(task='binary')
    self.val_auc = torchmetrics.AUROC(task='binary')

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = self.loss_fn(logits, y.float())

    preds = torch.sigmoid(logits)
    self.train_acc(preds, y)

    self.log('train_loss', loss, prog_bar=True)
    self.log('train_acc', self.train_acc, prog_bar=True, on_epoch=True)

    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = self.loss_fn(logits, y.float())

    preds = torch.sigmoid(logits)
    self.val_acc(preds, y)
    self.val_auc(preds, y)

    self.log('val_loss', loss, prog_bar=True)
    self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
    self.log('val_auc', self.val_auc, on_epoch=True, prog_bar=True)

    return loss

  # SGD optimizer
  def configure_optimizers(self):
    optimizer = torch.optim.SGD(
      self.parameters(),
      lr=self.hparams.lr,
      momentum=self.hparams.momentum,
      weight_decay=self.hparams.weight_decay,
      nesterov=True
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,
      mode='min',
      patience=2,
      factor=0.5
    )

    return {
      'optimizer': optimizer,
      'lr_scheduler': {
        'scheduler': scheduler,
        'monitor': 'val_loss'
      }
    }

  # for inference
  def predict_proba(self, x):
    logits = self(x)
    return torch.sigmoid(logits)
# classes
