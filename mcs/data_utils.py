from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm, trange


def tokenize_and_encode(obj, tokenizer, add_special_tokens=False):
    """ Tokenize and encode a nested object """
    if isinstance(obj, str):
        return tokenizer.encode(obj, add_special_tokens=add_special_tokens)
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, float):
        return None
    elif isinstance(obj, tuple):
        return list(tokenize_and_encode(o, tokenizer) for o in obj)
    return list(tokenize_and_encode(o, tokenizer) for o in tqdm(obj))


if __name__ == "__main__":
    pass
