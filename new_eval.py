#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-06 09:17:36
# @Author  : Chenghao Mou (chengham@isi.edu)
# @Link    : https://github.com/ChenghaoMou/ai2

# pylint: disable=unused-wildcard-import
# pylint: disable=no-member

import os
import sys

import torch
import numpy as np
import yaml
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.trainer_io import load_hparams_from_tags_csv
from pytorch_lightning.utilities.arg_parse import add_default_args
from test_tube import HyperOptArgumentParser
from torch.utils.data import DataLoader, RandomSampler
from huggingface import HuggingFaceClassifier
import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(hparams):

    # TODO: Change this model loader to your own.

    model = HuggingFaceClassifier.load_from_metrics(
        hparams=hparams,
        weights_path=hparams.weights_path,
        tags_csv=hparams.tags_csv,
        on_gpu=torch.cuda.is_available(),
        # on_gpu=False,
        map_location=None
    )

    model = model.to(device)
    model.eval()

    trainer = Trainer(gpus=1)
    results = []
    print (torch.cuda.is_available(), device)
    for batch in DataLoader(
            model.val_dataloader.dataset, sampler=RandomSampler(model.val_dataloader.dataset, replacement=False),
            shuffle=False, batch_size=8, collate_fn=model.collate_fn):
        # print(batch)
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        batch["token_type_ids"] = batch["token_type_ids"].to(device)
        batch["y"] = batch["y"].to(device)
        with torch.no_grad():
            results.append(model.validation_step(batch, -1))

    acc = model.validation_end(results)['val_acc']

    logger.info('Accuracy: {:.4f}'.format(acc*100.0))


if __name__ == '__main__':
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=True)
    add_default_args(parent_parser, root_dir)

    # TODO: Change this to your own model
    parser = HuggingFaceClassifier.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    main(hyperparams)
