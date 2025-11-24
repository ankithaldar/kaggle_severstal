#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''training script for classification models'''


# imports
import argparse
import os

from callbacks.callbacks import get_classification_checkpoint
from callbacks.discord_callbacks import DiscordCallback
from datamodules import ClassificationDataModule
from engines.classifier import LitClassifier
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything

# imports


# constants
# constants

# functions
def train_classifier(
  model_name='effnet_b1',
  batch_size=8,
  max_epochs=10,
  save_dir='checkpoints/',
  pseudo_labels=None
):

  seed_everything(42)

  dm = ClassificationDataModule(
    batch_size=batch_size,
    pseudo_labels=pseudo_labels,
    num_workers=4
  )

  model = LitClassifier(
    model_name=model_name,
    lr=0.01,
    weight_decay=1e-4,
    momentum=0.9
  )

  ckpt_callback = get_classification_checkpoint(save_dir)
  discord_callback = DiscordCallback(
    experiment_name=f'Severstal-classifier-{model_name}-{max_epochs}'
  )

  wandb_logger = WandbLogger(
    name=f'Severstal-classifier-{model_name}-{max_epochs}',
    project='kaggle-Severstal'
  )

  trainer = Trainer(
    accelerator='gpu',
    devices=1,
    precision='16-mixed',
    max_epochs=max_epochs,
    accumulate_grad_batches=4,
    logger=[wandb_logger],
    callbacks=[ckpt_callback, discord_callback],
    log_every_n_steps=20
  )

  trainer.fit(model, dm)
# functions


# main
def main():
  pass


# if main script
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--model_name',
    type=str,
    default='effnet_b1',
    choices=['effnet_b1', 'resnet34']
  )
  parser.add_argument('--batch_size', type=int, default=8)
  parser.add_argument('--max_epochs', type=int, default=10)
  parser.add_argument('--save_dir', type=str, default='checkpoints/')
  args = parser.parse_args()

  os.makedirs(args.save_dir, exist_ok=True)
  train_classifier(**vars(args))
