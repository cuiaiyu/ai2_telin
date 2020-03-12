import os
import sys
import json
import glob
import argparse
import jsonlines


def get_parser():
    def str2bool(v):
        v = v.lower()
        assert v == 'true' or v == 'false'
        return v.lower() == 'true'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     conflict_handler='resolve')
    parser.add_argument('--input_dir', type=str,
                        default='./physicaliqa-train-dev/physicaliqa-train-dev/',
                        help='path to the datasets')
    parser.add_argument('--output_dir', type=str,
                        default='./physicalbinqa-train-dev/physicalbinqa-train-dev/',
                        help='path to the output binary datasets')
    parser.add_argument('--json_extension', type=str, default='jsonl',
                        help='the extension of the json files')
    parser.add_argument('--task', type=str, default='physicaliqa',
                        choices=["physicaliqa", "socialiqa"],
                        help='the task name')
    parser.add_argument('--verbose', type=str2bool, default=False,
                        help='if verbose')
    return parser


def transform_piqa_jsonl(args):
    print ("Input Dir: {}".format(args.input_dir))
    json_paths = os.path.join(args.input_dir, '*.'+args.json_extension)
    print ('-'*50)

    for json_path in sorted(glob.glob(json_paths)):
        label_offset = 10e9
        json_path = json_path.split('\n')[0]
        jsonl = open(json_path, 'r')

        raw_data = []
        raw_labels = []

        # json dara points
        print (json_path)
        for line in jsonl:
            raw = json.loads(line)
            raw_data.append(raw)

        # labels
        phase = json_path.split('/')[-1].split('.')[0]
        label_file_name = phase + '-labels.lst'
        label_file_name = os.path.join(args.input_dir, label_file_name)
        label_file = open(label_file_name, 'r')
        for line in label_file:
            label = int(line.strip())
            raw_labels.append(label)
            label_offset = min(label_offset, label)

        # pre-processing
        print ("Label Offset: {}".format(label_offset))

        new_data = []
        new_labels = []
        for i in range(len(raw_data)):
            data_ = raw_data[i]
            label_ = raw_labels[i]
            if args.verbose:
                print (data_)
                print (label_)

            for sol in ['sol1', 'sol2']:
                new_data_ = {}
                new_data_['id'] = data_['id']
                new_data_['goal'] = data_['goal']
                new_data_['sol'] = data_[sol]
                new_data.append(new_data_)

            # TODO: for label, 1: yes 0: no
            if label_ - label_offset == 0: # yes/no for sol1/sol2
                new_labels.append(1)
                new_labels.append(0)
            elif label_ - label_offset == 1: # no/yes for sol1/sol2
                new_labels.append(0)
                new_labels.append(1)

            if args.verbose:
                print (new_data)
                print (new_labels)

        assert len(new_data) == len(new_labels)

        # save files
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        save_json_path = os.path.join(args.output_dir, phase+'.jsonl')
        save_label_path = os.path.join(args.output_dir, phase+'-labels.lst')
        with jsonlines.open(save_json_path, mode='w') as writer:
            writer.write_all(new_data)
        with open(save_label_path, 'w') as out_label:
            for label_ in new_labels:
                out_label.write(str(label_)+'\n')
        print (save_json_path)
        print (save_label_path)
        print ('-'*50)
    
    print ("Binary PIQA datasets saved to: {}".format(args.output_dir))


# python3 mcq2binq.py \
#     --input_dir=./socialiqa-train-dev/socialiqa-train-dev \
#     --output_dir=./socialbinqa-train-dev/socialbinqa-train-dev/ \
#     --task=socialiqa \
#     --verbose=False \
def transform_siqa_jsonl(args):
    print ("Input Dir: {}".format(args.input_dir))
    json_paths = os.path.join(args.input_dir, '*.'+args.json_extension)
    print ('-'*50)

    for json_path in sorted(glob.glob(json_paths)):
        label_offset = 10e9
        json_path = json_path.split('\n')[0]
        jsonl = open(json_path, 'r')

        raw_data = []
        raw_labels = []

        # json dara points
        print (json_path)
        for line in jsonl:
            raw = json.loads(line)
            raw_data.append(raw)

        # labels
        phase = json_path.split('/')[-1].split('.')[0]
        label_file_name = phase + '-labels.lst'
        label_file_name = os.path.join(args.input_dir, label_file_name)
        label_file = open(label_file_name, 'r')
        for line in label_file:
            label = int(line.strip())
            raw_labels.append(label)
            label_offset = min(label_offset, label)

        # pre-processing
        print ("Label Offset: {}".format(label_offset))

        new_data = []
        new_labels = []
        for i in range(len(raw_data)):
            data_ = raw_data[i]
            label_ = raw_labels[i]
            if args.verbose:
                print (data_)
                print (label_)

            for sol in ['answerA', 'answerB', 'answerC']:
                new_data_ = {}
                new_data_['context'] = data_['context']
                new_data_['question'] = data_['question']
                new_data_['answer'] = data_[sol]
                new_data.append(new_data_)

            # TODO: for label, 1: yes 0: no
            for label_idx in [0+label_offset, 1+label_offset, 2+label_offset]:
                if label_idx == label_:
                    new_labels.append(1)
                else:
                    new_labels.append(0)

            if args.verbose:
                print (new_data)
                print (new_labels)

        assert len(new_data) == len(new_labels)

        # save files
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        save_json_path = os.path.join(args.output_dir, phase+'.jsonl')
        save_label_path = os.path.join(args.output_dir, phase+'-labels.lst')
        with jsonlines.open(save_json_path, mode='w') as writer:
            writer.write_all(new_data)
        with open(save_label_path, 'w') as out_label:
            for label_ in new_labels:
                out_label.write(str(label_)+'\n')
        print (save_json_path)
        print (save_label_path)
        print ('-'*50)
    
    print ("Binary SIQA datasets saved to: {}".format(args.output_dir))


def transform(args):
    if args.json_extension == 'jsonl':
        if args.task == 'physicaliqa':
            transform_piqa_jsonl(args)
        elif args.task == 'socialiqa':
            transform_siqa_jsonl(args)
        else:
            raise NotImplementedError("Not able to deal with task: {} yet!".format(args.task))
    elif args.json_extension == 'json':
        raise NotImplementedError("Not able to deal with extension: {} yet!".format(args.json_extension))
    else:
        raise NotImplementedError("Not able to deal with extension: {} yet!".format(args.json_extension))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    transform(args)
