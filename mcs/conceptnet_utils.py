from __future__ import absolute_import, division, print_function

from tqdm import tqdm
import numpy as np
import random
import torch
import os

import argparse
import glob
import logging
import pickle

from tqdm import tqdm, trange


relations = [
    'AtLocation', 'CapableOf', 'Causes', 'CausesDesire',
    'CreatedBy', 'DefinedAs', 'DesireOf', 'Desires', 'HasA',
    'HasFirstSubevent', 'HasLastSubevent', 'HasPainCharacter',
    'HasPainIntensity', 'HasPrerequisite', 'HasProperty',
    'HasSubevent', 'InheritsFrom', 'InstanceOf', 'IsA',
    'LocatedNear', 'LocationOfAction', 'MadeOf', 'MotivatedByGoal',
    'NotCapableOf', 'NotDesires', 'NotHasA', 'NotHasProperty',
    'NotIsA', 'NotMadeOf', 'PartOf', 'ReceivesAction', 'RelatedTo',
    'SymbolOf', 'UsedFor', 'FormOf', 'DerivedFrom', 'HasContext',
    'Synonym', 'Etymologically', 'SimilarTo', 'Antonym', 'MannerOf',
    'dbpedia', 'DistinctFrom', 'Entails', 'EtymologicallyDerivedFrom',
]

split_into_words = {
    'Antonym': "antonym",
    'AtLocation': "at location",
    'CapableOf': "capable of",
    'Causes': "causes",
    'CausesDesire': "causes desire",
    'CreatedBy': "created by",
    'dbpedia': "dbpedia",
    'DefinedAs': "defined as",
    'DesireOf': "desire of",
    'Desires': "desires",
    'DerivedFrom': "derived from",
    'DistinctFrom': "distinct from",
    'Entails': "entails",
    'EtymologicallyRelatedTo': "etymologically related to",
    'EtymologicallyDerivedFrom': "etymologically derived from",
    'HasA': "has a",
    'HasContext': "has context",
    'HasFirstSubevent': "has first subevent",
    'HasLastSubevent': "has last subevent",
    'HasPainCharacter': "has pain character",
    'HasPainIntensity': "has pain intensity",
    'HasPrerequisite': "has prequisite",
    # actually it is "has prerequisite, but models were trained on it ..."
    'HasProperty': "has property",
    'HasSubevent': "has subevent",
    'FormOf': "form of",
    'InheritsFrom': "inherits from",
    'InstanceOf': 'instance of',
    'IsA': "is a",
    'LocatedNear': "located near",
    'LocationOfAction': "location of action",
    'MadeOf': "made of",
    'MannerOf': "manner of",
    'MotivatedByGoal': "motivated by goal",
    'NotCapableOf': "not capable of",
    'NotDesires': "not desires",
    'NotHasA': "not has a",
    'NotHasProperty': "not has property",
    'NotIsA': "not is a",
    'NotMadeOf': "not made of",
    'PartOf': "part of",
    'ReceivesAction': "receives action",
    'RelatedTo': "related to",
    'SimilarTo': "similar to",
    'SymbolOf': "symbol of",
    'Synonym': "synonym of",
    'UsedFor': "used for",
}


def load_comet_dataset(dataset_path=None, 
                       eos_token=None, 
                       sep_token=None, 
                       rel_lang=True, 
                       toy=False, 
                       discard_negative=True, 
                       sep=False, 
                       add_sep=False, 
                       prefix=None,
                       no_scores=False):
    if not eos_token:
        end_token = ""
    with open(dataset_path, encoding='utf_8') as f:
        f = f.read().splitlines()
        if toy:
            random.shuffle(f)
            f = f[:1000]
        output = []
        for line in tqdm(f):
            x = line.split("\t")
            if len(x) == 4:
                rel, e1, e2, label = x
            else:
                e1, rel, e2 = x
            if len(x) == 4:
                if discard_negative and label == "0": 
                    continue
            if not discard_negative and len(x) ==4:
                # convert to int, to avoid being encoded
                try:
                    label = int(label)
                except:
                    # in ConceptNet training data the label is float
                    label = -1
            if add_sep:
                e1 += (" " + sep_token)
            if prefix:
                e1 = prefix + " " + e1
            if eos_token is not None:
                e2 += (" " + eos_token)
            if rel_lang:
                rel = split_into_words[rel]
                if not rel:
                    continue
            else:
                rel = rel.lower()
            if add_sep:
                rel += (" " + sep_token)
            if len(x) == 4 and not no_scores:
                output.append((e1, rel, e2, label))
            elif len(x) == 4 and no_scores:
                output.append((e1, rel, e2))
            else:
                output.append((e1, rel, e2))
        # print some samples for sure
        # print(output[-3:])
    return output


if __name__ == "__main__":
    pass
