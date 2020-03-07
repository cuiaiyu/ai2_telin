from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import csv
import json
from configs import argparser
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import random
import re
import nltk
from conceptnet_utils import load_comet_dataset


DATA_SOURCES_LIST = [
    "densecap_videos",
    "crosstalk",
    "conceptual_captions",
    "coin",
    "how2",
    "whats_cookin",
    "coco",
    "20BN_something_to_something",
    "atomic",
    "atomic_new",
    "concept_net_100k",
    "concept_net",
    "physical_cs_concept_net_100k",
    "person_cs_concept_net_100k",
    "random_cs_concept_net_100k",
    "wikihow",
]


ATOMIC_ATTR_TABLE = {
    'oEffect': " has an effect ",
    'oReact':  " has a reaction ",
    'oWant':   " wants ",
    'xAttr':   ", personX has the attribute ",
    'xEffect': ", personX will ",
    'xIntent': ", personX has an intent of ",
    'xNeed':   ", personX needs ",
    'xReact':  ", personX has a reaction of ",
    'xWant':   ", PersonX wants "
}


def remove(text):
    remove_chars = '[0-9’!"#$%&\()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    return re.sub(remove_chars, '', text)


def detect(text):
    text_list = nltk.word_tokenize(text)
    english_punctuations = [
        ',', '.', ':', ';', '?', '(', ')',
        '[', ']', '&', '!', '*', '@', '#', '$', '%'
    ]

    text_list = [word for word in text_list if word not in english_punctuations]
    stops = set(stopwords.words("english"))
    text_list = [word for word in text_list if word not in stops]
    return nltk.pos_tag(text_list)


def get_atomic(args):
    all_captions = []
    attr_list = [
        'oEffect',
        'oReact',
        'oWant',
        'xAttr',
        'xEffect',
        'xIntent',
        'xNeed',
        'xReact',
        'xWant'
    ]

    data_path = os.path.join(args.data_root, 'atomic_data')
    for file_name in [os.path.join(data_path, 'v4_atomic_all.csv')]:
        data = pd.read_csv(file_name, sep=',')
        processing_data = data.values
        for i in range((processing_data.shape[0])):
            if len(all_captions) >= args.num_sentences:
                break
            main_sentence = processing_data[i][0].replace('___', 'something')
            for k in range(len(attr_list)):
                if len(all_captions) >= args.num_sentences:
                    break
                next = []
                next = processing_data[i][k+1].split('"')
                for j in range(len(next)): #str structure
                    result = remove(next[j])
                    if result == '' or result == 'none' or result == ' ':
                        pass
                    else:
                        if args.naturalize_atomic:
                            final = main_sentence + \
                                ATOMIC_ATTR_TABLE[attr_list[k]] + result
                        else:
                            final = main_sentence + ' ' + result
                        all_captions.append(final)
    print ("ATOMIC: Number of captions: {}".format(len(all_captions)))
    return all_captions


def get_atomic_new(args):
    all_captions = []
    data_path = os.path.join(args.data_root, 'atomic_data')
    for file_name in [os.path.join(data_path, 'v4_atomic_all.csv')]:
        all_captions = atomic_new_generator(file_name)
    if args.num_sentences > 0:
        all_captions = all_captions[:args.num_sentences]
    print ("ATOMIC: Number of captions: {}".format(len(all_captions)))
    return all_captions


def get_concept_net_100k(args, data_path=None, as_tuple=False):
    all_captions = []
    if data_path is None:
        data_path = os.path.join(args.data_root,
            # 'concept_net/original/train100k.txt')
            'train100k.txt')
    triplets = load_comet_dataset(data_path)
    if args.num_sentences < 0:
        args.num_sentences = 10e20
    for triplet in triplets:
        if len(all_captions) >= args.num_sentences:
            break
        e1, rel, e2, score = triplet
        if as_tuple:
            all_captions.append((e1, rel, e2))
        else:
            s = e1 + ' ' + rel + ' ' + e2 + '.'
            all_captions.append(s)
    print ("Concept Net 100k: Number of captions: {}".format(len(all_captions)))
    return all_captions


def get_physical_cs_concept_net_100k(args, data_path=None, as_tuple=False):
    all_captions = []
    if data_path is None:
        data_path = os.path.join(args.data_root,
            'concept_net/physical-cs-n100000.txt')
    if args.num_sentences < 0:
        args.num_sentences = 10e20
    with open(data_path, 'r') as data_file:
        for line in data_file:
            if len(all_captions) >= args.num_sentences:
                break
            s = line.strip() + '.'
            all_captions.append(s)
    print ("Physical CS - 100k: Number of captions: {}".format(len(all_captions)))
    return all_captions


def get_person_cs_concept_net_100k(args, data_path=None, as_tuple=False):
    all_captions = []
    if data_path is None:
        data_path = os.path.join(args.data_root,
            'concept_net/person-cs-n100000.txt')
    if args.num_sentences < 0:
        args.num_sentences = 10e20
    with open(data_path, 'r') as data_file:
        for line in data_file:
            if len(all_captions) >= args.num_sentences:
                break
            s = line.strip() + '.'
            all_captions.append(s)
    print ("Person CS - 100k: Number of captions: {}".format(len(all_captions)))
    return all_captions


def get_random_cs_concept_net_100k(args, data_path=None, as_tuple=False):
    all_captions = []
    if data_path is None:
        data_path = os.path.join(args.data_root,
            'concept_net/random-cs-n100000.txt')
    if args.num_sentences < 0:
        args.num_sentences = 10e20
    with open(data_path, 'r') as data_file:
        for line in data_file:
            if len(all_captions) >= args.num_sentences:
                break
            s = line.strip() + '.'
            all_captions.append(s)
    print ("Random CS - 100k: Number of captions: {}".format(len(all_captions)))
    return all_captions


def get_concept_net(args, tuples=False, as_tuple=False):
    all_captions = []
    cn_csv_file = os.path.join(args.data_root,
        'concept_net/original/simplified_english_conceptnet.csv')
    f = open(cn_csv_file, "r")
    if args.num_sentences < 0:
        args.num_sentences = 10e20
    for line in f:
        if len(all_captions) >= args.num_sentences:
            break
        content = line.split()
        sub, rel, obj, score = content
        sub = ' '.join(sub.split("_")) if "_" in sub else sub
        rel = ' '.join(rel.split("_")) if "_" in rel else rel
        obj = ' '.join(obj.split("_")) if "_" in obj else obj
        s = sub + ' ' + rel + ' ' + obj + '.'
        if as_tuple:
            t = (sub, rel, obj)
            all_captions.append(t)
        else:
            all_captions.append(s)
    return all_captions


def get_densecap_videos(args):
    all_captions = []
    data_path = os.path.join(args.data_root, 'densecap_videos')
    train_path = os.path.join(data_path, 'train.json')
    val1_path = os.path.join(data_path, 'val_1.json')
    val2_path = os.path.join(data_path, 'val_2.json')

    for path in [train_path, val1_path, val2_path]:
        with open(path, 'r') as f:
            data = json.load(f)
            for video in data:
                if len(all_captions) >= args.num_sentences:
                    break
                info = data[video]
                captions = [x.strip() for x in info['sentences']]
                all_captions += captions
    print ("Dense Cap Video: Number of captions: {}".format(len(all_captions)))
    return all_captions


def get_coin(args):
    all_captions = []
    data_path = os.path.join(args.data_root, 'coin', 'COIN.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
        root_key = "database"
        ids = sorted(list(data[root_key].keys()))
        for vid_id in ids:
            if len(all_captions) >= args.num_sentences:
                break
            annots = data[root_key][vid_id]["annotation"]
            for annot in annots:
                tkns = word_tokenize(annot["label"])
                if len(tkns) < 2:
                    continue
                all_captions.append(annot["label"])
    print ("COIN: Number of captions: {}".format(len(all_captions)))
    return all_captions


def get_whats_cookin(args):
    data_path = os.path.join(args.data_root, 'whats_cookin', 'exportable_action_objects_youtube.csv')
    all_captions = []
    dating_df = pd.read_csv(data_path, sep=',', header=None)
    dating_df = dating_df.drop([0, 1, 2], axis=1)
    dating_np = dating_df.values
    for i in range(len(dating_np[:, 0])):
        if len(all_captions) >= args.num_sentences:
            break
        if str(dating_np[i, 1]) == 'nan':
            tkns = word_tokenize(str(dating_np[i, 0]))
            if len(tkns) < 2:
                continue
            all_captions.append(str(dating_np[i, 0]))
        else:
            tkns = word_tokenize(str(str(dating_np[i, 0]) + ' ' + str(dating_np[i, 1])))
            if len(tkns) < 2:
                continue
            all_captions.append(str(dating_np[i, 0]) + ' ' + str(dating_np[i, 1]))
    print ("Whats Cookin: Number of captions: {}".format(len(all_captions)))
    return all_captions


def get_conceptual_captions(args):
    data_path = os.path.join(args.data_root, 'conceptual_captions', 'Train_GCC-training.tsv')
    all_captions = []
    dating_df = pd.read_csv(data_path, sep='\t', header=None)
    dating_df = pd.read_csv(data_path, sep='\t', header=None)
    dating_df = dating_df.drop([1], axis=1)
    # print(dating_df)
    dating_np = dating_df.values
    for i in range(len(dating_np[:,0])):
        if len(all_captions) >= args.num_sentences:
            break
        all_captions.append((dating_np[i,:][0]))
    print ("Conceptual Captions: Number of captions: {}".format(len(all_captions)))
    return all_captions


def get_20BN_something_to_something(args):
    all_captions = []
    for samples in ['something-something-v2-train.json','something-something-v2-validation.json']:
        data_path = os.path.join(args.data_root, '20BN_something_to_something','V2',str(samples))
        with open(data_path, 'r') as f:
            data = json.load(f)
            for i in range(len(data)):
                if len(all_captions) >= args.num_sentences:
                    break
                for num in range(len(data[i]["placeholders"])):
                    template = data[i]["template"]
                    a = template.find('[')
                    b = template.find(']')
                    template = template.replace(template[a:b + 1], data[i]["placeholders"][num])
                    tkns = word_tokenize(template)
                    if len(tkns) < 2:
                        continue
                    all_captions.append(template)
    print ("Smth-Smth: Number of captions: {}".format(len(all_captions)))
    return all_captions


def get_crosstalk(args):
    ## TODO: write this function
    raise NotImplementedError("Not Implemented Yet!")
    all_captions = []
    print ("CrossTalk: Number of captions: {}".format(len(all_captions)))
    return all_captions


def get_how2(args):
    all_captions = []
    for samples in [os.path.join('how2', 'sum_cv'),
                    os.path.join('how2', 'sum_devtest'),
                    os.path.join('how2', 'sum_train')]:
        for samples_1 in ['desc.tok.txt', 'tran.tok.txt']:
            data_path = os.path.join(args.data_root, samples, samples_1)
            f = open(data_path, "rb")
            line = f.readline()
            while line:
                line = f.readline()
                line = line[:-1]
                line = line.decode()
                template = line.replace(line[0:12], '')  # delete useless stuff
                sent_tokenize_list = sent_tokenize(template)
                for i in range(len(sent_tokenize_list)):
                    if len(all_captions) >= args.num_sentences:
                        break
                    tkns = word_tokenize(sent_tokenize_list[i])
                    if len(tkns) < 2:
                        continue
                    else:
                        all_captions.append(sent_tokenize_list[i])
            f.close()
    print("How2: Number of captions: {}".format(len(all_captions)))
    return all_captions


def get_coco(args):
    all_captions = []
    for samples in ['captions_val2017.json','captions_train2017.json']:
        data_path = os.path.join(args.data_root, 'coco',str(samples))
        with open(data_path, 'r') as f:
            data = json.load(f)
            for i in range(len(data["annotations"])):
                if len(all_captions) >= args.num_sentences:
                    break
                result = (data["annotations"][i]["caption"])
                tkns = word_tokenize(result)
                if len(tkns) < 2:
                    continue
                else:
                    all_captions.append(result)
    print ("Coco: Number of captions: {}".format(len(all_captions)))
    return all_captions


def get_wikihow(args):
    all_captions = []
    fi = open(os.path.join(args.data_root, 'wikihow/wikihow_summaries_train.txt'))
    for line in fi:
        if len(all_captions) >= args.num_sentences:
            break
        text = line.strip()
        all_captions.append(text)

    print ("WikiHow: Number of captions: {}".format(len(all_captions)))
    return all_captions


def get_data(args, save=True):
    datasets = args.datasets
    if type(datasets) != list:
        datasets = [datasets]
    all_data = []
    if args.num_sentences < 0:
        args.num_sentences = 1e20
    for dataset in sorted(datasets):
        assert dataset in DATA_SOURCES_LIST
        assert 'get_'+dataset in globals()
        # get function by name
        func = globals()['get_'+dataset]
        data = func(args)
        all_data += data

    # save txt file
    if save:
        print ("\nALL: Number of captions: {}".format(len(all_data)))
        if args.save_path is None:
            save_name = "all_" + "_".join(sorted(datasets)) +\
                        "_{}".format(len(all_data)) + ".txt"
            save_path = os.path.join(args.data_root, "multimodal", save_name)
        else:
            save_path = args.save_path
        if args.train_test_split:
            save_path_train = save_path.split(".")[0] + "_train.txt"
            save_path_test = save_path.split(".")[0] + "_test.txt"
            train_num = int(float(len(all_data)) * args.split_ratio)
            # shuffling
            random.shuffle(all_data)
            train_data = all_data[:train_num]
            test_data = all_data[train_num:]
            
            ''' eval, train, test split'''
            if args.eval_num_sentences > 0: # default = 300
                eval_data = train_data[:args.eval_num_sentences]
                train_data = train_data[args.eval_num_sentences:]
                save_path_eval = save_path.split(".")[0] + "_eval.txt"
                fo = open(save_path_eval, "w")
                for line in eval_data:
                    l = line.strip()
                    fo.write(l+'\n')
                fo.close()
                print ("saving data file at {}".format(save_path_eval))
                print("Eval: {} samples".format(len(eval_data)))
            ''' end '''
            
            fo = open(save_path_train, "w")
            for line in train_data:
                l = line.strip()
                fo.write(l+'\n')
            fo.close()
            print ("saving data file at {}".format(save_path_train))
            fo = open(save_path_test, "w")
            for line in test_data:
                l = line.strip()
                fo.write(l+'\n')
            fo.close()
            print ("saving data file at {}".format(save_path_test))
            print ("Train: {} samples  Test: {} samples".format(
                len(train_data), len(test_data)))
        else:
            ''' eval, train split'''
            random.shuffle(all_data)
            train_data = all_data
            save_path_train = save_path.split(".")[0] + "_train.txt"
            
            if args.eval_num_sentences > 0: # default = 300
                eval_data = train_data[:args.eval_num_sentences]
                train_data = train_data[args.eval_num_sentences:]
                save_path_eval = save_path.split(".")[0] + "_eval.txt"
                fo = open(save_path_eval, "w")
                for line in eval_data:
                    l = line.strip()
                    fo.write(l+'\n')
                fo.close()
                print ("saving data file at {}".format(save_path_eval))
                print("Eval: {} samples".format(len(eval_data)))
            ''' end '''
            
            fo = open(save_path_train, "w")
            for line in train_data:
                l = line.strip()
                fo.write(l+'\n')
            fo.close()
            print ("saving data file at {}".format(save_path_train))
    return data


if __name__ == "__main__":
    args = argparser()
    get_data(args)
