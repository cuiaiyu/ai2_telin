import math
from typing import List, Dict, Any

import numpy
import torch

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter
from allennlp.predictors import Predictor
from allennlp.nn import util
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
from transformers import BertTokenizer, BertForMaskedLM

import argparse
import sys


class SmoothGradient(object):
    """
    Interprets the prediction using SmoothGrad (https://arxiv.org/abs/1706.03825)
    """
    def __init__(self, model, embedding_layer=None, get_loss_func=None, args=None):
        # Hyperparameters
        self.stdev = 0.01
        self.num_samples = 10
        self.model = model
        self.args = args
        self.embedding_layer = embedding_layer
        self.get_loss_func = get_loss_func

    def saliency_interpret(self, inputs, **kwargs):
        grads = self._smooth_grads(inputs, **kwargs)
        new_grads = {}

        # Normalize results
        for key, grad in grads.items():
            # TODO (@Eric-Wallace), SmoothGrad is not using times input normalization.
            # Fine for now, but should fix for consistency.

            # The [0] here is undo-ing the batching that happens in get_gradients.
            for i in range(len(grad)):
                embedding_grad = numpy.sum(grad[i], axis=1)
                norm = numpy.linalg.norm(embedding_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in embedding_grad]
                # grads[key] = normalized_grad
                new_grads[key+"_{}".format(i)] = normalized_grad

        # print (grads['grad_input_1'])
        return new_grads

    def _register_forward_hook(self, stdev: float):
        """
        Register a forward hook on the embedding layer which adds random noise to every embedding.
        Used for one term in the SmoothGrad sum.
        """
        def forward_hook(module, inputs, output):
            # Random noise = N(0, stdev * (max-min))
            scale = output.detach().max() - output.detach().min()
            noise = torch.randn(output.shape).to(output.device) * stdev * scale

            # Add the random noise
            output.add_(noise)

        # Register the hook
        # embedding_layer = self.model.roberta.embeddings
        # embedding_layer = self.model.bert.embeddings
        handle = self.embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _smooth_grads(self, inputs, **kwargs):
        total_gradients: Dict[str, Any] = {}
        for _ in range(self.num_samples):
            handle = self._register_forward_hook(self.stdev)
            # grads = self.get_gradients(inputs, **kwargs)[0]
            grads = self.get_gradients(inputs, **kwargs)
            handle.remove()

            # Sum gradients
            if total_gradients == {}:
                total_gradients = grads
            else:
                for key in grads.keys():
                    total_gradients[key] += grads[key]

        # Average the gradients
        for key in total_gradients.keys():
            total_gradients[key] /= self.num_samples

        return total_gradients

    def get_gradients(self, inputs, **kwargs):
        embedding_gradients = []
        hooks = self._register_embedding_gradient_hooks(embedding_gradients)

        loss = self.get_loss_func(self.model, inputs, **kwargs)
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        for hook in hooks:
            hook.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embedding_gradients):
            key = "grad_input_" + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()

        # return grad_dict, outputs
        return grad_dict

    def _register_embedding_gradient_hooks(self, embedding_gradients):
        def hook_layers(module, grad_in, grad_out):
            embedding_gradients.append(grad_out[0])

        backward_hooks = []
        # embedding_layer = self.model.roberta.embeddings
        # embedding_layer = self.model.bert.embeddings
        backward_hooks.append(self.embedding_layer.register_backward_hook(hook_layers))
        return backward_hooks


class IntegratedGradient(object):
    """
    Interprets the prediction using Integrated Gradients (https://arxiv.org/abs/1703.01365)
    """
    def __init__(self, model, embedding_layer=None, get_loss_func=None, args=None):
        # Hyperparameters
        self.model = model
        self.args = args
        self.embedding_layer = embedding_layer
        self.get_loss_func = get_loss_func

    def saliency_interpret(self, inputs, **kwargs):
        # Run integrated gradients
        grads = self._integrate_gradients(inputs, **kwargs)
        new_grads = {}

        # Normalize results
        for key, grad in grads.items():
            # The [0] here is undo-ing the batching that happens in get_gradients.
            for i in range(len(grad)):
                embedding_grad = numpy.sum(grad[i], axis=1)
                norm = numpy.linalg.norm(embedding_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in embedding_grad]
                new_grads[key+"_{}".format(i)] = normalized_grad

        return new_grads

    def _register_forward_hook(self, alpha: int, embeddings_list: List):
        """
        Register a forward hook on the embedding layer which scales the embeddings by alpha. Used
        for one term in the Integrated Gradients sum.

        We store the embedding output into the embeddings_list when alpha is zero.  This is used
        later to element-wise multiply the input by the averaged gradients.
        """
        def forward_hook(module, inputs, output):
            # Save the input for later use. Only do so on first call.
            if alpha == 0:
                embeddings_list.append(output.squeeze(0).clone().detach().cpu().numpy())

            # Scale the embedding by alpha
            output.mul_(alpha)

        # Register the hook
        # embedding_layer = util.find_embedding_layer(self.predictor._model)
        handle = self.embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _integrate_gradients(self, inputs, **kwargs):
        """
        Returns integrated gradients for the given :class:`~allennlp.data.instance.Instance`
        """
        ig_grads: Dict[str, Any] = {}

        # List of Embedding inputs
        embeddings_list: List[numpy.ndarray] = []

        # Use 10 terms in the summation approximation of the integral in integrated grad
        steps = 10

        # Exclude the endpoint because we do a left point integral approximation
        for alpha in numpy.linspace(0, 1.0, num=steps, endpoint=False):
            # Hook for modifying embedding value
            handle = self._register_forward_hook(alpha, embeddings_list)

            grads = self.get_gradients(inputs, **kwargs)
            handle.remove()

            # Running sum of gradients
            if ig_grads == {}:
                ig_grads = grads
            else:
                for key in grads.keys():
                    ig_grads[key] += grads[key]

        # Average of each gradient term
        for key in ig_grads.keys():
            ig_grads[key] /= steps

        # Gradients come back in the reverse order that they were sent into the network
        embeddings_list.reverse()

        # Element-wise multiply average gradient by the input
        for idx, input_embedding in enumerate(embeddings_list):
            key = "grad_input_" + str(idx + 1)
            ig_grads[key] *= input_embedding

        return ig_grads

    def get_gradients(self, inputs, **kwargs):
        embedding_gradients = []
        hooks = self._register_embedding_gradient_hooks(embedding_gradients)

        loss = self.get_loss_func(self.model, inputs, **kwargs)
        self.model.zero_grad()
        loss.backward()

        for hook in hooks:
            hook.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embedding_gradients):
            key = "grad_input_" + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()

        # return grad_dict, outputs
        return grad_dict

    def _register_embedding_gradient_hooks(self, embedding_gradients):
        def hook_layers(module, grad_in, grad_out):
            embedding_gradients.append(grad_out[0])

        backward_hooks = []
        # embedding_layer = self.model.roberta.embeddings
        # embedding_layer = self.model.bert.embeddings
        backward_hooks.append(self.embedding_layer.register_backward_hook(hook_layers))
        return backward_hooks


def test_sg_roberta(args, model, tokenizer):
    from k_hop_models.run_lm_finetuning_with_graphbert import mask_tokens

    sent = "<s> I am a really good boy."
    # sent = "I am a really good boy."
    inputs = tokenizer.encode(sent)
    inputs = torch.tensor(inputs)
    inputs = inputs.unsqueeze(0)
    inputs, labels = mask_tokens(inputs, tokenizer, args)
    masked_sent = tokenizer.decode(inputs[0].tolist())
    print (inputs)
    print (labels)
    print (masked_sent)
    inputs, labels = inputs.to(args.device), labels.to(args.device)
    outputs = model(inputs, masked_lm_labels=labels)
    preds = outputs[1]
    pred_tkns = torch.argmax(preds, dim=-1)
    mask = (labels != -1)
    print (pred_tkns[mask])
    inputs_ = inputs
    inputs_[mask] = pred_tkns[mask]
    print (inputs_)
    print (tokenizer.decode(inputs_[0].tolist()))

    def get_roberta_loss(model, inputs, **kwargs):
        target = kwargs['labels']
        outputs = model(inputs, masked_lm_labels=target)
        loss = outputs[0]
        return loss

    sg = SmoothGradient(model, embedding_layer=model.roberta.embeddings,
        get_loss_func=get_roberta_loss, args=args)
    sg.saliency_interpret(inputs, labels=labels)


def test_ig_roberta(args, model, tokenizer):
    from k_hop_models.run_lm_finetuning_with_graphbert import mask_tokens

    sent = "<s> I am a really good boy."
    # sent = "I am a really good boy."
    inputs = tokenizer.encode(sent)
    inputs = torch.tensor(inputs)
    inputs = inputs.unsqueeze(0)
    inputs, labels = mask_tokens(inputs, tokenizer, args)
    masked_sent = tokenizer.decode(inputs[0].tolist())
    print (inputs)
    print (labels)
    print (masked_sent)
    inputs, labels = inputs.to(args.device), labels.to(args.device)
    outputs = model(inputs, masked_lm_labels=labels)
    preds = outputs[1]
    pred_tkns = torch.argmax(preds, dim=-1)
    mask = (labels != -1)
    print (pred_tkns[mask])
    inputs_ = inputs
    inputs_[mask] = pred_tkns[mask]
    print (inputs_)
    print (tokenizer.decode(inputs_[0].tolist()))

    def get_roberta_loss(model, inputs, **kwargs):
        target = kwargs['labels']
        outputs = model(inputs, masked_lm_labels=target)
        loss = outputs[0]
        return loss

    ig = IntegratedGradient(model, embedding_layer=model.roberta.embeddings,
        get_loss_func=get_roberta_loss, args=args)
    ig.saliency_interpret(inputs, labels=labels)


def interpret():
    from interprets.dataset import ClassificationDataset
    dataset = ClassificationDataset.load(cache_dir=cache_dirs[0] if isinstance(cache_dirs, list) else cache_dirs,
                                         file_mapping=self.task_config[self.hparams.task_name]['file_mapping'][dataset_name],
                                         task_formula=self.task_config[self.hparams.task_name]['task_formula'],
                                         type_formula=self.task_config[self.hparams.task_name]['type_formula'],
                                         preprocessor=self.tokenizer,
                                         pretokenized=self.task_config[self.hparams.task_name].get('pretokenized', False),
                                         label_formula=self.task_config[self.hparams.task_name].get('label_formula', None),
                                         label_offset=self.task_config[self.hparams.task_name].get('label_offset', 0),
                                         label_transform=self.task_config[self.hparams.task_name].get('label_transform', None),
                                         shuffle=self.task_config[self.hparams.task_name].get('shuffle', False),
                                         )


def load_ai2_models(model, ai2_state_dict):
    pooler_exists = False
    lm_head_exists = False
    linear_exists = False
    for k in ai2_state_dict:
        if 'weight' in k:
            print (k, torch.sum(ai2_state_dict[k].data))
        else:
            print (k)
        if 'pooler' in k:
            pooler_exists = True
        if 'lm_head' in k:
            lm_head_exists = True
    print ('-'*50)
    print ('pooler_exists', pooler_exists)
    print ('lm_head_exists', lm_head_exists)

    model_state_dict = model.state_dict()
    for k, params in model_state_dict.items():
        if 'roberta.encoder' in k or 'roberta.embeddings' in k or 'roberta.pooler' in k:
            if 'roberta.pooler' in k and not pooler_exists:
                continue
            param_name = k[8:]
            ai2_param_name = 'encoder.model.' + param_name
            assert ai2_param_name in ai2_state_dict
            model_state_dict[k].copy_(ai2_state_dict[ai2_param_name].data)
            print ("Loaded {}".format(param_name))
        elif 'lm_head' in k and lm_head_exists:
            param_name = k
            ai2_param_name = param_name
            assert ai2_param_name in ai2_state_dict
            model_state_dict[k].copy_(ai2_state_dict[ai2_param_name].data)
            print ("Loaded {}".format(param_name))
        else:
            print ("Unloaded weights: {}!".format(k))


def get_parser():
    def str2bool(v):
        v = v.lower()
        assert v == 'true' or v == 'false'
        return v.lower() == 'true'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     conflict_handler='resolve')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--data_jsonfile', type=str, default=None)
    return parser


def argparser():
    parser = get_parser()
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argparser()

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.device = device

    tokenizer = RobertaTokenizer.from_pretrained('large_roberta')
    model = RobertaForMaskedLM.from_pretrained('large_roberta')

    # path = '/lfs1/telinwu/courses/csci699_dial_rl/notebooks/bert_models/'
    # tokenizer = BertTokenizer.from_pretrained(path)
    # model = BertForMaskedLM.from_pretrained(path)

    model = model.to(args.device)

    ai2_ckpt = torch.load(args.model_path)['state_dict']
    load_ai2_models(model, ai2_ckpt)

    interpret()

    # test_sg_roberta(args, model, tokenizer)
    # test_ig_roberta(args, model, tokenizer)
