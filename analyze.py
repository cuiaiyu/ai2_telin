import os
import sys

import torch
import yaml
import argparse
import sys
import json
import jsonlines


def get_parser():
    def str2bool(v):
        v = v.lower()
        assert v == 'true' or v == 'false'
        return v.lower() == 'true'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     conflict_handler='resolve')
    parser.add_argument('--file_gt', default='', type=str,
                        help="the path to the ground truth file")
    parser.add_argument('--file_pred', default='', type=str, nargs="+",
                        help="the path to the predictions file")
    parser.add_argument('--prob_pred', default='', type=str, nargs="+",
                        help="the path to the predicted probabilities file")
    parser.add_argument('--file_data', default='', type=str,
                        help="the path to the original testing set file")
    parser.add_argument('--top_k', default=10, type=int,
                        help="printing top k analysis")
    return parser


def compute_acc(in_f_pd, in_f_gt):
    f_gt = open(in_f_gt, "r")
    f_pd = open(in_f_pd, "r")

    gts = []
    pds = []
    for line in f_gt:
        gt_ = int(line.strip())
        gts.append(gt_)
    for line in f_pd:
        pd_ = int(line.strip())
        pds.append(pd_)
    
    acc = 0.0
    cnt = 0.0
    assert len(gts) == len(pds)
    for i in range(len(gts)):
        gt_ = gts[i]
        pd_ = pds[i]
        if gt_ == pd_:
            acc += 1.0
        cnt += 1.0
    acc = acc / cnt * 100.0

    model_weight = in_f_pd.split("/")[-2].split("-")[1]
    print ("Accuracy of {} = {:.4f}%".format(model_weight, acc))

    f_gt.close()
    f_pd.close()
    return acc


def compare_preds_physicalIQA(in_f_pds, in_f_gt, in_data, in_probs, top_k=10):
    if len(in_f_pds) > 2:
        print ("Too many files to compare, please just use two!")
        return None
    
    f_pd1 = open(in_f_pds[0], "r")
    f_pd2 = open(in_f_pds[1], "r")
    f_pr1 = open(in_probs[0], "r")
    f_pr2 = open(in_probs[1], "r")
    f_gt = open(in_f_gt, "r")
    f_data = jsonlines.open(in_data)
    
    gts = []
    pds1 = []
    pds2 = []
    probs1 = []
    probs2 = []
    for line in f_gt:
        gt_ = int(line.strip())
        gts.append(gt_)
    for line in f_pd1:
        pd_ = int(line.strip())
        pds1.append(pd_)
    for line in f_pd2:
        pd_ = int(line.strip())
        pds2.append(pd_)
    for line in f_pr1:
        pr_ = line.strip().split()
        pr_ = [float(x) for x in pr_]
        probs1.append(pr_)
    for line in f_pr2:
        pr_ = line.strip().split()
        pr_ = [float(x) for x in pr_]
        probs2.append(pr_)
    assert len(pds1) == len(pds2) == len(gts)
    
    print ("---- Comparison on {} ----".format(in_data))
    model1 = in_f_pds[0].split("/")[-2].split("-")[1]
    model2 = in_f_pds[1].split("/")[-2].split("-")[1]

    print_cnt = 0
    to_print = True
    agreed_cnt = 0
    disagreed_cnt = 0
    agreed_correct_cnt = 0
    agreed_wrong_cnt = 0
    disagreed_model1_correct = 0
    disagreed_model2_correct = 0
    for i in range(len(gts)):
        if print_cnt >= top_k:
            to_print = False
        data = f_data.read()
        goal = data["goal"]
        sol1 = data["sol1"]
        sol2 = data["sol2"]
        pred1 = pds1[i]
        pred2 = pds2[i]
        gt = gts[i]
        if pred1 == pred2:
            agreed_cnt += 1
            if pred1 == gt:
                agreed_correct_cnt += 1
            elif pred1 != gt:
                agreed_wrong_cnt += 1
        else:
            disagreed_cnt += 1
            if pred1 == gt:
                disagreed_model1_correct += 1
            elif pred2 == gt:
                disagreed_model2_correct += 1
        if to_print:
            if pred1 == gt and pred2 != gt or \
              pred1 != gt and pred2 == gt:
                print ("Sample {}".format(print_cnt+1))
                print ("Goal: {}".format(goal))
                print ("Sol1: {}".format(sol1))
                print ("Sol2: {}".format(sol2))
                probs1[i] = [round(x, 5) for x in probs1[i]]
                probs2[i] = [round(x, 5) for x in probs2[i]]
                if gt == 0:
                    print ("GT: Sol1")
                else:
                    print ("GT: Sol2")
                if pred1 == gt and pred2 != gt:
                    print ("{} correct, prob: {}".format(model1, probs1[i]))
                    print ("{} wrong, prob: {}".format(model2, probs2[i]))
                elif pred1 != gt and pred2 == gt:
                    print ("{} wrong, prob: {}".format(model1, probs1[i]))
                    print ("{} correct, prob: {}".format(model2, probs2[i]))
                print ("-"*50)
                print_cnt += 1
    total_cnt = len(gts)
    agreed_avg = float(agreed_cnt) / float(total_cnt) * 100.0
    disagreed_avg = float(disagreed_cnt) / float(total_cnt) * 100.0
    agreed_correct_avg = float(agreed_correct_cnt) / float(total_cnt) * 100.0
    agreed_wrong_avg = float(agreed_wrong_cnt) / float(total_cnt) * 100.0
    disagreed_model1_correct_avg = float(disagreed_model1_correct) / float(total_cnt) * 100.0
    disagreed_model2_correct_avg = float(disagreed_model2_correct) / float(total_cnt) * 100.0
    print ("Model 1: {}".format(model1))
    print ("Model 2: {}".format(model2))
    print ("Agreed Percentage:                   {:.3f}%".format(agreed_avg))
    print ("Agreed Correct Percentage:           {:.3f}%".format(agreed_correct_avg))
    print ("Agreed Wrong Percentage:             {:.3f}%".format(agreed_wrong_avg))
    print ("Disagreed Percentage:                {:.3f}%".format(disagreed_avg))
    print ("Disagreed Model1 Correct Percentage: {:.3f}%".format(disagreed_model1_correct_avg))
    print ("Disagreed Model2 Correct Percentage: {:.3f}%".format(disagreed_model2_correct_avg))
    print ("-"*50)
    return None


def compare_preds_socialIQA(in_f_pds, in_f_gt, in_data, in_probs, top_k=10):
    if len(in_f_pds) > 2:
        print ("Too many files to compare, please just use two!")
        return None
    
    f_pd1 = open(in_f_pds[0], "r")
    f_pd2 = open(in_f_pds[1], "r")
    f_pr1 = open(in_probs[0], "r")
    f_pr2 = open(in_probs[1], "r")
    f_gt = open(in_f_gt, "r")
    f_data = jsonlines.open(in_data)
    
    gts = []
    pds1 = []
    pds2 = []
    probs1 = []
    probs2 = []
    for line in f_gt:
        gt_ = int(line.strip())
        gts.append(gt_)
    for line in f_pd1:
        pd_ = int(line.strip())
        pds1.append(pd_)
    for line in f_pd2:
        pd_ = int(line.strip())
        pds2.append(pd_)
    for line in f_pr1:
        pr_ = line.strip().split()
        pr_ = [float(x) for x in pr_]
        probs1.append(pr_)
    for line in f_pr2:
        pr_ = line.strip().split()
        pr_ = [float(x) for x in pr_]
        probs2.append(pr_)
    assert len(pds1) == len(pds2) == len(gts)
    
    print ("---- Comparison on {} ----".format(in_data))
    model1 = in_f_pds[0].split("/")[-2].split("-")[1]
    model2 = in_f_pds[1].split("/")[-2].split("-")[1]

    print_cnt = 0
    to_print = True
    agreed_cnt = 0
    disagreed_cnt = 0
    agreed_correct_cnt = 0
    agreed_wrong_cnt = 0
    disagreed_model1_correct = 0
    disagreed_model2_correct = 0
    for i in range(len(gts)):
        if print_cnt >= top_k:
            to_print = False
        data = f_data.read()
        goal = data["context"]
        sol1 = data["answerA"]
        sol2 = data["answerB"]
        sol3 = data["answerC"]
        pred1 = pds1[i]
        pred2 = pds2[i]
        gt = gts[i]
        if pred1 == pred2:
            agreed_cnt += 1
            if pred1 == gt:
                agreed_correct_cnt += 1
            elif pred1 != gt:
                agreed_wrong_cnt += 1
        else:
            disagreed_cnt += 1
            if pred1 == gt:
                disagreed_model1_correct += 1
            elif pred2 == gt:
                disagreed_model2_correct += 1
        if to_print:
            if pred1 == gt and pred2 != gt or \
              pred1 != gt and pred2 == gt:
                print ("Sample {}".format(print_cnt+1))
                print ("Context: {}".format(goal))
                print ("answerA: {}".format(sol1))
                print ("answerB: {}".format(sol2))
                print ("answerC: {}".format(sol3))
                probs1[i] = [round(x, 5) for x in probs1[i]]
                probs2[i] = [round(x, 5) for x in probs2[i]]
                if gt == 1:
                    print ("GT: answerA")
                elif gt == 2:
                    print ("GT: answerB")
                else:
                    print ("GT: answerC")
                if pred1 == gt and pred2 != gt:
                    print ("{} correct, prob: {}".format(model1, probs1[i]))
                    print ("{} wrong, prob: {}".format(model2, probs2[i]))
                elif pred1 != gt and pred2 == gt:
                    print ("{} wrong, prob: {}".format(model1, probs1[i]))
                    print ("{} correct, prob: {}".format(model2, probs2[i]))
                print ("-"*50)
                print_cnt += 1
    total_cnt = len(gts)
    agreed_avg = float(agreed_cnt) / float(total_cnt) * 100.0
    disagreed_avg = float(disagreed_cnt) / float(total_cnt) * 100.0
    agreed_correct_avg = float(agreed_correct_cnt) / float(total_cnt) * 100.0
    agreed_wrong_avg = float(agreed_wrong_cnt) / float(total_cnt) * 100.0
    disagreed_model1_correct_avg = float(disagreed_model1_correct) / float(total_cnt) * 100.0
    disagreed_model2_correct_avg = float(disagreed_model2_correct) / float(total_cnt) * 100.0
    print ("Model 1: {}".format(model1))
    print ("Model 2: {}".format(model2))
    print ("Agreed Percentage:                   {:.3f}%".format(agreed_avg))
    print ("Agreed Correct Percentage:           {:.3f}%".format(agreed_correct_avg))
    print ("Agreed Wrong Percentage:             {:.3f}%".format(agreed_wrong_avg))
    print ("Disagreed Percentage:                {:.3f}%".format(disagreed_avg))
    print ("Disagreed Model1 Correct Percentage: {:.3f}%".format(disagreed_model1_correct_avg))
    print ("Disagreed Model2 Correct Percentage: {:.3f}%".format(disagreed_model2_correct_avg))
    print ("-"*50)
    return None


def analyze(args):
    accs = []
    print ("---------- Accuracies ----------")
    print ("Ground Truth: {}".format(args.file_gt))
    for f_pd in args.file_pred:
        acc = compute_acc(f_pd, args.file_gt)
        accs.append(acc)
    print ("--------------------------------")

    # comparisons
    if len(args.file_pred) > 1:
        if "physicaliqa" in args.file_data:
            res = compare_preds_physicalIQA(args.file_pred,
                args.file_gt, args.file_data, args.prob_pred,
                top_k=args.top_k)
        elif "socialiqa" in args.file_data:
            res = compare_preds_socialIQA(args.file_pred,
                args.file_gt, args.file_data, args.prob_pred,
                top_k=args.top_k)
        else:
            raise NotImplementedError("Not implemented yet!")

    # other analysis
    pass


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    analyze(args)
