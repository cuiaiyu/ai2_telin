import argparse
import warnings
import torch
from tqdm import tqdm
from argparse import Namespace
from ai2.utility import load_config
from ai2.model import Classifier
from run_darpa import MODELS, CONFIGS, TOKENIZERS
from loguru import logger

warnings.simplefilter(action='ignore', category=FutureWarning)


def load_from_metrics(base, weights_path, on_gpu, map_location=None, **kargs):
    """
    Primary way of loading model from csv weights path
    :param weights_path:
    :param tags_csv:
    :param on_gpu:
    :param map_location: dic for mapping storage {'cuda:1':'cuda:0'}
    :return:
    """
    # hparams = Namespace(**kargs)
    # hparams.__setattr__('on_gpu', on_gpu)

    if on_gpu:
        if map_location is not None:
            checkpoint = torch.load(weights_path, map_location=map_location)
        else:
            checkpoint = torch.load(weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

    # load the state_dict on the model automatically
    # print(hparams)
    model = base(**kargs)
    model.__setattr__('on_gpu', on_gpu)
    model.load_state_dict(checkpoint['state_dict'])

    # give model a chance to load something
    model.on_load_checkpoint(checkpoint)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Eval ai2 darpa tasks with pytorch-transformers')
    parser.add_argument('--task', '-t', choices=['vcrqa', 'vcrqar', 'anli', 'hellaswag', 'physicaliqa', 'socialiqa'],
                        help='DARPA task, see https://leaderboard.allenai.org/?darpa_offset=0', required=True)

    parser.add_argument('--train_config', help='Training config file', required=True)
    parser.add_argument('--model_type', choices=MODELS, help='Model type', required=True)
    parser.add_argument('--tokenizer_type', choices=TOKENIZERS, help='Tokenizer type', required=True)
    parser.add_argument('--model_config_type', choices=CONFIGS, help='Model configuration type', required=False)
    parser.add_argument('--model_weight', help='Model weight from huggingface', required=True)
    parser.add_argument('--tokenizer_weight', help='Pretrained tokenizer from huggingface', required=True)
    parser.add_argument('--model_config_weight', help='Predefined configuration', required=False)
    parser.add_argument('--weights_path', help='Saved model weights file')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--train_eval', action='store_true', default=False, help='Eval on train or not')

    args = parser.parse_args()

    TASK = load_config("ai2/tasks.yaml", args.task)

    if args.model_config_weight is None or args.model_config_type is None:
        args.model_config_weight = args.model_weight
        args.model_config_type = args.model_type

    pretrained_model = load_from_metrics(
        base=Classifier,
        weights_path=args.weights_path,
        on_gpu=torch.cuda.is_available(),
        map_location=None,
        task_config=TASK,
        train_config=load_config(args.train_config),
        model_class=MODELS[args.model_type],
        model_path=args.model_weight,
        tokenizer_class=TOKENIZERS[args.tokenizer_type],
        tokenizer_path=args.tokenizer_weight,
        model_config_class=CONFIGS[args.model_config_type],
        model_config_path=args.model_config_weight
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # predict
    pretrained_model.eval()
    pretrained_model.freeze()
    pretrained_model.to(device)

    dataloader = pretrained_model.tng_dataloader if args.train_eval else pretrained_model.val_dataloader 

    outputs = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        for key, val in batch.items():
            batch[key] = val.to(device)
        res = pretrained_model.validation_step(batch, i)
        outputs.append(res)

    truth = torch.cat([x['truth'] for x in outputs], dim=0).reshape(-1).cpu().detach().numpy().tolist()
    pred = torch.cat([x['pred'] for x in outputs], dim=0).reshape(-1).cpu().detach().numpy().tolist()
    prob = torch.cat([x['prob'] for x in outputs], dim=0).cpu().detach().numpy().tolist()

    # assert truth == list(map(lambda x: int(x.decode("utf-8").strip('\n') if not isinstance(x, int) else x) - TASK['start'], pretrained_model.dev_y))

    with open(args.output, "w") as output:
        output.write('\n'.join(map(lambda l: '\t'.join(map(str, l)), prob)))
    with open(args.output.replace(".tsv", ".truth"), "w") as output:
        output.write('\n'.join(map(str, truth)))
