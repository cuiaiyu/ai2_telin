from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import csv
import json
from configs import argparser
import numpy as np
import random
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from get_multimodal_data import get_concept_net, get_concept_net_100k
from get_multimodal_data import get_physical_cs_concept_net_100k
from get_multimodal_data import get_person_cs_concept_net_100k
from get_multimodal_data import get_random_cs_concept_net_100k


class KG_Dict(object):
    def __init__(self, args):
        self._build_entity_dict(args)

    def _build_entity_dict(self, args):
        if type(args.kg_datasets) == str:
            args.kg_dataset = args.kg_datasets
        elif len(args.kg_datasets) == 1:
            args.kg_dataset = args.kg_datasets[0]
        else:
            raise NotImplementedError("Currently only deal with one KG file at a time.")

        d = {}

        if args.kg_dataset == "concept_net_100k":
            cn = get_concept_net_100k(args, as_tuple=True)
        elif args.kg_dataset == "concept_net":
            cn = get_concept_net(args, as_tuple=True)
        elif args.kg_dataset == "physical_cs_concept_net_100k":
            cn = get_physical_cs_concept_net_100k(args, as_tuple=True)
        elif args.kg_dataset == "person_cs_concept_net_100k":
            cn = get_person_cs_concept_net_100k(args, as_tuple=True)
        elif args.kg_dataset == "random_cs_concept_net_100k":
            cn = get_random_cs_concept_net_100k(args, as_tuple=True)
        else:
            raise NotImplementedError("Not implemented for dataset {}".format(args.kg_dataset))

        for t in cn:
            e1, rel, e2 = t
            e1, rel, e2 = e1.lower(), rel.lower(), e2.lower()
            if args.kg_multiword_matching:
                e1_words = word_tokenize(e1)
                e2_words = word_tokenize(e2)
                for e1_word in e1_words:
                    if e1_word not in d:
                        d[e1_word] = []
                    d[e1_word].append(t)
                for e2_word in e2_words:
                    if e2_word not in d:
                        d[e2_word] = []
                    d[e2_word].append(t)
            else:
                if e1 not in d:
                    d[e1] = []
                if e2 not in d:
                    d[e2] = []
                d[e1].append(t)
                d[e2].append(t)

        self.d = d

    def entity_sample(self, entity, k=3, if_tuple=False, dummy_dict=None):
        if dummy_dict is not None:
            self.d = dummy_dict
        assert (type(entity) == str or type(entity) == list)
        if entity not in self.d:
            raise ValueError("No such entity in KG dict")

        tuples = self.d[entity]
        # print (len(tuples))
        indices = range(len(tuples))
        sampled_indices = np.random.choice(indices, min(len(tuples), k), replace=False)
        sampled_tuples = [tuples[i] for i in sampled_indices]
        ub = random.randint(1, len(sampled_tuples)+1)
        sampled_tuples = sampled_tuples[:ub]
        # print (sampled_tuples); raise

        if type(sampled_tuples[0]) == str:
            assert if_tuple == False
        else:
            if not if_tuple:
                sampled_tuples = [' '.join(t) for t in sampled_tuples]

        return sampled_tuples


if __name__ == "__main__":
    args = argparser()
    kg_dict = KG_Dict(args)
    kg_dict.entity_sample("hockey")
