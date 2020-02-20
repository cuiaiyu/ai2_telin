""" 
Given dev-labels.lst, compare two model's prediction
dev-predictions.lst and dev-probabilities.lst
and find where two predictions are inconsistent
"""

import argparse
import jsonlines
def compare(label_path, jsonl_path,
            exp_name_1, pred_1_path, prob_1_path,
            exp_name_2, pred_2_path, prob_2_path,
            exp_1_strictly_better):
    print(jsonl_path)
    labels = [int(i) for i in open(label_path).readlines()]
    qa_reader = jsonlines.open(jsonl_path)
    pred_1 = [int(i) for i in open(pred_1_path).readlines()]
    prob_1 = [[float(j) for j in i.split()] for i in open(prob_1_path).readlines()]
    pred_2 = [int(i) for i in open(pred_2_path).readlines()]
    prob_2 = [[float(j) for j in i.split()] for i in open(prob_2_path).readlines()]
    assert len(labels) == len(pred_1) == len(pred_2) == len(prob_1) == len(prob_2)
    
    n_samples = len(labels)
    count_disagree = 0
    count_disagree_1_correct = 0
    count_agree_correct = 0

    for i in range(len(labels)):
        sample = qa_reader.read()
        label = labels[i]
        # two experiment results do not agree
        if pred_1[i] != pred_2[i]:
            count_disagree += 1
            if not exp_1_strictly_better or pred_1[i] == label:
                print()
                print("---------------------------")
                print("Disagreement on", str(i+1)+"-th sample:")
                for key in sample:
                    if sample[key] != "":
                        print("{}: {}".format(key, sample[key]))
                print("The correct answer is", label)
                print()
                print("Experiment " + exp_name_1 + "'s choice: " + str(pred_1[i]))
                print("\twith probabilities for each choice:", *prob_1[i])
                print()
                print("Experiment " + exp_name_2 + "'s choice: " + str(pred_2[i]))
                print("\twith probabilities for each choice:", *prob_2[i])
                print("---------------------------")
                print()
            if pred_2[i] != label:
                count_disagree_1_correct += 1
        # two results agree
        else:
            if pred_1[i] == label:
                count_agree_correct += 1

    print("=======================")
    count_1_correct = count_agree_correct + count_disagree_1_correct
    count_2_correct = count_agree_correct + (count_disagree - count_disagree_1_correct)
    print("{}'s accuracy is {:.3f}% - {:4} out of {:4} samples.".format(exp_name_1, count_1_correct / n_samples * 100, count_1_correct, n_samples))
    print("{}'s accuracy is {:.3f}% - {:4} out of {:4} samples.".format(exp_name_2, count_2_correct / n_samples * 100, count_2_correct, n_samples))
    print()
    count_agree                       = n_samples - count_disagree
    count_agree_wrong                 = count_agree - count_agree_correct
    count_disagree_2_correct          = count_disagree - count_disagree_1_correct
    
    agreed_percent                    = 100 * count_agree / n_samples
    agreed_correct_percent            = 100 * count_agree_correct / n_samples
    agreed_wrong_percent              = 100 * count_agree_wrong / n_samples
    disagreed_percent                 = 100 * count_disagree / n_samples
    disagreed_model1_correct_percent  = 100 * count_disagree_1_correct / n_samples
    disagreed_model2_correct_percent  = 100 * count_disagree_2_correct / n_samples
    
    print ("Agreed Percentage:                   {:.3f}%".format(agreed_percent))
    print ("Agreed Correct Percentage:           {:.3f}%".format(agreed_correct_percent))
    print ("Agreed Wrong Percentage:             {:.3f}%".format(agreed_wrong_percent))
    print ("Disagreed Percentage:                {:.3f}%".format(disagreed_percent))
    print ("Disagreed Model1 Correct Percentage: {:.3f}%".format(disagreed_model1_correct_percent))
    print ("Disagreed Model2 Correct Percentage: {:.3f}%".format(disagreed_model2_correct_percent))
    print("=======================")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", required=True)
    parser.add_argument("--exp_name_1", required=True)
    parser.add_argument("--exp_name_2", required=True)
    parser.add_argument("--pred_1_path", required=True)
    parser.add_argument("--pred_2_path", required=True)
    parser.add_argument("--prob_1_path", required=True)
    parser.add_argument("--prob_2_path", required=True)
    parser.add_argument("--jsonl_path", required=True)
    parser.add_argument("--exp_1_strictly_better", action="store_true")
    args = parser.parse_args()
    if args.exp_1_strictly_better:
        compare(args.label_path, args.jsonl_path,
                args.exp_name_1, args.pred_1_path, args.prob_1_path,
                args.exp_name_2, args.pred_2_path, args.prob_2_path,
                True)
    else:
        compare(args.label_path, args.jsonl_path,
                args.exp_name_1, args.pred_1_path, args.prob_1_path,
                args.exp_name_2, args.pred_2_path, args.prob_2_path,
                False)

if __name__ == "__main__":
    main()
