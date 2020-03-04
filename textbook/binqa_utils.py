import os
import sys
import json
import glob
import jsonlines


# Alter T, for indicated percentage
def binqa_true_percentage(f, true_percentage):
    print ('-'*50)
    print ("Data File Name:  {}".format(f))
    print ("True Percentage: {}".format(true_percentage))

    raw_data = []
    raw_labels = []
    label_offset = 10e6

    raw_f = open(f, 'r')
    raw_f_name = f.split("/")[-1].split(".")[0]
    task = f.split("/")[-2].split("-")[0]
    raw_dir = '/'.join(f.split("/")[:-1])
    raw_label_f_name = raw_f_name + '-labels.lst'
    raw_label_f_name = os.path.join(raw_dir, raw_label_f_name)
    raw_label_f = open(raw_label_f_name, 'r')
    
    for line in raw_f:
        raw_ = json.loads(line.strip())
        raw_data.append(raw_)

    for line in raw_label_f:
        label_ = int(line.strip())
        raw_labels.append(label_)
        label_offset = min(label_offset, label_)

    for i in range(len(raw_labels)):
        raw_labels[i] -= label_offset

    assert sum(raw_labels) == len(raw_labels) // 2

    print ("Total original T/F data poins: {}".format(len(raw_labels)))
    print ("Total original T   data poins: {}".format(sum(raw_labels)))

    # let T be X, we have X / X + F = true percentage (tp)
    # tp*X + tp* F = X => X = tp*F / (1-tp)
    if true_percentage > 0.5:
        len_T = len(raw_labels) // 2
        false_percentage = 1.0 - true_percentage
        len_F = false_percentage * len_T / (1.0 - false_percentage)
    else:
        len_F = len(raw_labels) // 2
        len_T = true_percentage * len_F / (1.0 - true_percentage)
    len_T, len_F = int(round(len_T, 0)), int(round(len_F, 0))
    print ("New Length of T: {}".format(len_T))
    print ("New Length of F: {}".format(len_F))
    new_total = len_T + len_F
    print ("New total T/F data points: {}".format(new_total))
    T_percentage = float(len_T) / float(new_total) * 100.0
    print ("Percentage of T of total:  {}%".format(T_percentage))

    cut_data = []
    cut_labels = []

    cnt_T, cnt_F = 0, 0
    for i in range(len(raw_data)):
        curr_data = raw_data[i]
        curr_label = raw_labels[i]
        if curr_label == 1:
            if cnt_T >= len_T:
                pass
            else:
                cut_data.append(curr_data)
                cut_labels.append(curr_label)
                cnt_T += 1
        elif curr_label == 0:
            if cnt_F >= len_F:
                pass
            else:
                cut_data.append(curr_data)
                cut_labels.append(curr_label)
                cnt_F += 1
        else:
            raise
         
    assert sum(cut_labels) == len_T
    cnt_T, cnt_F = 0, 0
    for label in cut_labels:
        if label == 1:
            cnt_T += 1
        elif label == 0:
            cnt_F += 1
        else:
            raise
    assert cnt_T == sum(cut_labels) == len_T
    assert cnt_F == len_F

    print ("Total data points of new T/F dataset: {}".format(len(cut_data)))
    
    out_dir = './cache/percentage_exps/{}_{}_true_percentage_{}'.format(
        task, raw_f_name, int(true_percentage*100.0))
    print ("Saving to :{}".format((out_dir)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_json_path = os.path.join(out_dir, raw_f_name+'.jsonl')
    save_label_path = os.path.join(out_dir, raw_f_name+'-labels.lst')

    with jsonlines.open(save_json_path, mode='w') as writer:
         writer.write_all(cut_data)
    with open(save_label_path, 'w') as out_label:
        for label_ in cut_labels:
            out_label.write(str(label_)+'\n')
    print ("jsonl saved at:      {}".format(save_json_path))
    print ("labels.lst saved at: {}".format(save_label_path))

    print ('-'*50)
    return save_json_path, save_label_path, out_dir


def balance_tp(f, tp_list=[20, 40, 50, 60, 80]):
    assert tp_list is not None
    assert max(tp_list) <= 100
    max_tp = max(tp_list)
    min_tp = min(min(tp_list), 100-max_tp)
    print ("Minimum tp is {}%".format(min_tp))
    print ('-'*50)
    print ("Data File Name:  {}".format(f))

    raw_data = []
    raw_labels = []
    label_offset = 10e6

    raw_f = open(f, 'r')
    raw_f_name = f.split("/")[-1].split(".")[0]
    task = f.split("/")[-2].split("-")[0]
    raw_dir = '/'.join(f.split("/")[:-1])
    raw_label_f_name = raw_f_name + '-labels.lst'
    raw_label_f_name = os.path.join(raw_dir, raw_label_f_name)
    raw_label_f = open(raw_label_f_name, 'r')
    
    for line in raw_f:
        raw_ = json.loads(line.strip())
        raw_data.append(raw_)

    for line in raw_label_f:
        label_ = int(line.strip())
        raw_labels.append(label_)
        label_offset = min(label_offset, label_)

    for i in range(len(raw_labels)):
        raw_labels[i] -= label_offset

    assert sum(raw_labels) == len(raw_labels) // 2

    print ("Total original T/F data poins: {}".format(len(raw_labels)))
    print ("Total original T   data poins: {}".format(sum(raw_labels)))

    min_tp = float(min_tp) / 100.0
    if min_tp > 0.5:
        len_T = len(raw_labels) // 2
        min_tp_f = 1.0 - min_tp
        len_F = min_tp_f * len_T / (1.0 - min_tp_f)
    else:
        len_F = len(raw_labels) // 2
        len_T = min_tp * len_F / (1.0 - min_tp)
    len_T, len_F = int(round(len_T, 0)), int(round(len_F, 0))
    print ("New Length of T: {}".format(len_T))
    print ("New Length of F: {}".format(len_F))
    new_total = len_T + len_F
    print ("New total T/F data points: {}".format(new_total))
    T_percentage = float(len_T) / float(new_total) * 100.0
    print ("Percentage of T of total:  {}%".format(T_percentage))

    max_total = new_total  # the max all the dataset can have
    print ("#"*50)
    print ("For balanced tp data, max and fixed len is: {}".format(max_total))

    for tp in tp_list:
        tp_f = float(tp) / 100.0
        len_T = int(round(max_total * tp_f, 0))
        len_F = max_total - len_T
        print ("TP: {}% Len: {}".format(tp, len_T))
        cut_data = []
        cut_labels = []

        cnt_T, cnt_F = 0, 0
        for i in range(len(raw_data)):
            curr_data = raw_data[i]
            curr_label = raw_labels[i]
            if cnt_T + cnt_F >= max_total:
                break
            if curr_label == 1:
                if cnt_T >= len_T:
                    pass
                else:
                    cut_data.append(curr_data)
                    cut_labels.append(curr_label)
                    cnt_T += 1
            elif curr_label == 0:
                if cnt_F >= len_F:
                    pass
                else:
                    cut_data.append(curr_data)
                    cut_labels.append(curr_label)
                    cnt_F += 1
            else:
                raise
         
        assert sum(cut_labels) == len_T
        cnt_T, cnt_F = 0, 0
        for label in cut_labels:
            if label == 1:
                cnt_T += 1
            elif label == 0:
                cnt_F += 1
            else:
                raise
        print ("T/F : {}/{}".format(cnt_T, cnt_F))
        assert cnt_T == sum(cut_labels) == len_T

        print ("Total data points of new T/F dataset: {}".format(len(cut_data)))
        
        out_dir = './cache/balanced_percentage_exps/{}_{}_true_percentage_{}'.format(
            task, raw_f_name, int(tp_f*100.0))
        print ("Saving to :{}".format((out_dir)))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        save_json_path = os.path.join(out_dir, raw_f_name+'.jsonl')
        save_label_path = os.path.join(out_dir, raw_f_name+'-labels.lst')

        with jsonlines.open(save_json_path, mode='w') as writer:
             writer.write_all(cut_data)
        with open(save_label_path, 'w') as out_label:
            for label_ in cut_labels:
                out_label.write(str(label_)+'\n')
        print ("jsonl saved at:      {}".format(save_json_path))
        print ("labels.lst saved at: {}".format(save_label_path))

        print ('-'*50)

    return


def tp_demo():
    f = './cache/physicalbinqa-train-dev/physicalbinqa-train-dev/dev.jsonl'

    true_percentage = 0.8
    binqa_true_percentage(f, true_percentage)

    """
    true_percentage = 0.2
    binqa_true_percentage(f, true_percentage)

    true_percentage = 0.5
    binqa_true_percentage(f, true_percentage)
    """


if __name__ == "__main__":
    f = './cache/physicalbinqa-train-dev/physicalbinqa-train-dev/train.jsonl'
    # f = './cache/physicalbinqa-train-dev/physicalbinqa-train-dev/dev.jsonl'
    balance_tp(f)
