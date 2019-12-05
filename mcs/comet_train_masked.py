# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os, sys
import random
import datetime

import numpy as np
import torch

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
import torch.nn.functional as F
from .conceptnet_utils import load_comet_dataset
from .data_utils import tokenize_and_encode


def pre_process_datasets(encoded_datasets, input_len, max_e1, max_r, max_e2, mask_parts, mask_token):
    tensor_datasets = []
    assert (mask_parts in ["e1", "r", "e2"])
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.full((n_batch, input_len), fill_value=0, dtype=np.int64)
        lm_labels = np.full((n_batch, input_len), fill_value=-1, dtype=np.int64)
        for i, (e1, r, e2, label), in enumerate(dataset):
            # truncate if input is too long
            if len(e1) > max_e1:
                e1 = e1[:max_e1]
            if len(r) > max_r:
                r = r[:max_r]
            if len(e2) > max_e2:
                e2 = e2[:max_e2]

            if mask_parts == "e1":
                input_ids[i, :len(e1)] = mask_token
                lm_labels[i, :len(e1)] = e1
            else:
                input_ids[i, :len(e1)] = e1
            start_r = max_e1
            end_r = max_e1 + len(r)
            if mask_parts == "r":
                input_ids[i, start_r:end_r] = mask_token
                lm_labels[i, start_r:end_r] = r
            else:
                input_ids[i, start_r:end_r] = r
            start_e2 = max_e1 + max_r
            end_e2 = max_e1 + max_r + len(e2)
            if mask_parts == "e2":
                input_ids[i, start_e2:end_e2] = mask_token
                lm_labels[i, start_e2:end_e2] = e2
            else:
                input_ids[i, start_e2:end_e2] = e2

            if i == 0:
                print("one encoded sample: e1", e1, "r", r, "e2", e2)
                print("input_ids:", input_ids[i])
                print("lm_labels", lm_labels[i])

        input_mask = (input_ids != 0)  # attention mask
        all_inputs = (input_ids, lm_labels, input_mask)
        tensor_datasets.append((torch.tensor(input_ids), torch.tensor(lm_labels),
                                torch.tensor(input_mask).to(torch.float32)))
    return tensor_datasets


if __name__ == '__main__':
    main()
