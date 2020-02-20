import argparse
import sys


def get_parser():
    def str2bool(v):
        v = v.lower()
        assert v == 'true' or v == 'false'
        return v.lower() == 'true'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     conflict_handler='resolve')

    ########################
    ## DP TREE PARAMETERS ##
    ########################
    parser.add_argument('--datasets', nargs="+", default='densecap_videos',
                        help="gather the datasets under datasets/ folder, now we have:"
                        " densecap_videos, crosstask, how2, whats_cookin, conceptual_captions,"
                        " atomic, concept_net_100k (COMET), concept_net (large)")
    parser.add_argument('--kg_datasets', nargs="+", default='concept_net_100k',
                        help="gather the datasets under datasets/ folder, now we have:"
                        " concept_net_100k, concept_net, ...")
    parser.add_argument('--data_root', type=str, default='datasets/',
                        help='root path to the datasets')
    parser.add_argument('--allennlp_path', type=str,
                        choices=[
                            "../AllenNLPModels/biaffine-dependency-parser-ptb-2018.08.23.tar.gz",
                            "../AllenNLPModels/biaffine-dependency-parser-ud-2018.08.23.tar.gz"
                        ],
                        default='../AllenNLPModels/biaffine-dependency-parser-ptb-2018.08.23.tar.gz',
                        help='the path to AllenNLP models')
    parser.add_argument('--num_sentences', type=int, default=-1,
                        help='number of sentences to process in a dataset, -1 means takes all')
    parser.add_argument('--save_path', type=str, default=None,
                        help='path to save the gathered annotations')
    parser.add_argument('--dp_input_txt', type=str, default=None,
                        help='path to input txt file to dp constructor')
    parser.add_argument('--train_test_split', type=str2bool, default=False,
                        help='if performing test trainn split')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                        help='how many percentages for training')
    parser.add_argument('--dp_graph_dict', type=str, default=None,
                        help='the path to the graph dict for dependency parsing')
    parser.add_argument('--train_dp_dict', type=str, default=None,
                        help='the path to the training dict for dependency parsing')
    parser.add_argument('--eval_dp_dict', type=str, default=None,
                        help='the path to the testing dict for dependency parsing')
    parser.add_argument('--dp_repr_method', type=str, default="glove",
                        choices=["glove", "word2vec", "bert", "random", "roberta"],
                        help='the method for representation of graph nodes')
    parser.add_argument('--merge_nodes', type=str2bool, default=True,
                        help='if merging the nodes with the same tokens')
    parser.add_argument('--dp_batch', type=int, default=1,
                        help='how many sentences are constructed in the dp graph,'
                              '-1 means all sentences, and 1 means single sentence')
    parser.add_argument('--graphbert_model', type=str,
                        default="basic_message_passing",
                        help='the name of the graph pretraining model class')
    parser.add_argument('--num_gat_heads', type=int, default=8,
                        help='number of attention heads in GAT model')
    parser.add_argument('--mask_degree_lb', type=int, default=None,
                        help='The degree lower bound to mask out a node')
    parser.add_argument('--mask_degree_ub', type=int, default=20,
                        help='The degree upper bound to mask out a node')
    parser.add_argument('--wordpiece_method', type=str, default="decode",
                        choices=["decode", "segment"],
                        help='The method to deal with multiple tokens in one word node')
    parser.add_argument('--out_dp_repr_dim', type=int, default=768,
                        help='inter representation dimension of dp graph module')
    parser.add_argument('--train_edge_embedding', type=str2bool, default=False,
                        help='if train edge embedding network as well')
    parser.add_argument('--load_graphbert_checkpoints', type=str, default=None,
                        help='the path to load the graph model checkpoints')
    parser.add_argument('--graph_decoder_configs', type=str, default=None,
                        help='the configs for the graph bert decoder transformer')
    parser.add_argument('--use_default_configs', type=str2bool, default=False,
                        help='if enforcing using default configs')
    parser.add_argument('--do_link_prediction', type=str2bool, default=False,
                        help='if doing link prediction')
    parser.add_argument('--eval_num_sentences', type=int, default=300,
                        help='take first K sentences as eval.txt')
    parser.add_argument('--naturalize_atomic', type=str2bool, default=False,
                        help='if naturalizing atomic relation types')
    parser.add_argument('--naturalize_cn', type=str2bool, default=False,
                        help='if naturalizing concept net relation types')
    parser.add_argument('--num_cn_graphs', type=int, default=100,
                        help='number of concept net graphs to sample')
    parser.add_argument('--train_cn_dict', type=str, default=None,
                        help='the path to the training dict for concept net graphs')
    parser.add_argument('--eval_cn_dict', type=str, default=None,
                        help='the path to the testing dict for concept net graphs')
    parser.add_argument('--cn_mother_dict', type=str, default=None,
                        help='the path to the mother dict for concept net graphs')
    parser.add_argument('--graph_type', type=str, default='dp', choices=['dp', 'cn'],
                        help='the type that the model is training on')
    parser.add_argument('--lm_layer_for_graph', type=int, default=None,
                        help='which layer of lm encoder for graph input representation')
    parser.add_argument('--lm_graph_merge_training', type=str2bool, default=False,
                        help='if merge training our graphbert and lm models')
    parser.add_argument('--dp_fully_conn', type=str2bool, default=False,
                        help='if dp under fully connected experiment')
    parser.add_argument('--max_num_graph_mask', type=int, default=None,
                        help='the maximum number of graph bert masked tokens')
    parser.add_argument('--lm_loss_with_graph', type=str2bool, default=False,
                        help='if lm_loss_with_graph')
    parser.add_argument('--org_roberta_repr', type=str2bool, default=True,
                        help='if using original roberta representations')
    parser.add_argument('--ner', type=str2bool, default=False,
                        help='if with ner')
    parser.add_argument('--enhanced_fully_conn', type=str2bool, default=False,
                        help='if dp under fully connected plus original dp')
    parser.add_argument('--dp_undirected', type=str2bool, default=False,
                        help='if dp with undirected graphs')
    parser.add_argument('--special_masking', type=str, default=None,
                        choices=[None, 'entity', 'relation', 'entity_nokg', 'relation_nokg'],
                        help='special masking techique')
    parser.add_argument('--max_num_kg_sample', type=int, default=3,
                        help='the maximum number of kg triplets to sample')
    parser.add_argument('--kg_augment_with_pad', type=str2bool, default=True,
                        help='if adding <pad> when doing kg augmentation')
    parser.add_argument('--standard_masking_prob', type=float, default=0,
                        help='to what probability to use special masking')
    parser.add_argument('--kg_multiword_matching', type=str2bool, default=False,
                        help='turn on to sample kg triplets if matching at least one of the tokens')


    ########################
    ## CONCEPTNET DATASET ##
    ########################
    parser.add_argument('--conceptnet_file_name', type=str, default=None,
                        help='the name of the training txt file for k hop')
    parser.add_argument('--khop_file_name', type=str,
                        default='datasets/concept_net/2hop_train_single_token.txt',
                        help='the name of the training txt file for k hop')
    parser.add_argument('--single_file_name', type=str, default='',
                        help='the name of the testing txt file for single triplet')
    parser.add_argument('--pos_neg_file_name', type=str, default='',
                        help='the name of the positive-negative mixed training file')
    parser.add_argument('--khop_k', type=int, default=2,
                        help='the k for the k-hop experiment')
    parser.add_argument('--num_negative', type=int, default=100,
                        help='number of negative samples')
    parser.add_argument('--single_neg_sampling', type=str2bool, default=False,
                        help='to perform negative sampling on single triplet')
    parser.add_argument('--generate_test', type=str2bool, default=False,
                        help='indicate to generate the test set')
    parser.add_argument('--generate_khop', type=str2bool, default=True,
                        help='indicate to generate the k-hop dataset')
    parser.add_argument('--single_mask_token', type=str2bool, default=True,
                        help='filter the k-hop datasets to have single masking token')
    parser.add_argument('--neg_sampling', type=str2bool, default=False,
                        help='to perform negative sampling')
    parser.add_argument('--if_csv', type=str2bool, default=False,
                        help='if generating the csv file for human annotation')
    parser.add_argument('--human_csv_file', type=str,
                        default='datasets/concept_net/pos_neg.csv',
                        help='human evaluations csv file')
    parser.add_argument('--strategy_prob', type=float, default=0.5,
                        help='mixing probability of negative sampling')
    parser.add_argument('--num_test', type=int, default=10,
                        help='number of testing khop paths or triplets')
    parser.add_argument('--generate_selfqa', type=str2bool, default=False,
                        help='if generating self qa dataset')
    parser.add_argument('--generate_selfqa_single', type=str2bool, default=False,
                        help='if generating self qa dataset')
    

    #####################
    ## BERT PARAMETERS ##
    #####################
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions"\
                             " and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate"\
                             " the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not"
                             " the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not"
                             " the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models"
                             " downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs"
                             " (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", type=str2bool, default=False,
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before"\
                             " performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to"\
                             " perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=5,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same"\
                             " prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    # add argument of max_len for each entity/rel
    parser.add_argument('--max_s1', type=int, default=20)
    parser.add_argument('--max_r1', type=int, default=10)
    parser.add_argument('--max_s2', type=int, default=20)
    parser.add_argument('--max_r2', type=int, default=10)
    parser.add_argument('--max_o', type=int, default=20)
    parser.add_argument("--rel_lang", type=str2bool,
                        help="Use natural language to represent relations.")
    parser.add_argument('--mask_parts', type=str,
                        help='e1, r, or e2, which part to mask and predict')
    parser.add_argument('--num_selfqa_choices', type=int, default=5,
                        help='number of choices for self qa task')
    parser.add_argument('--do_selfqa', type=str2bool, default=False,
                        help='if performing self qa task')
    parser.add_argument('--linebyline', type=str2bool, default=False,
                        help='if performing MLM line by line from data')
    parser.add_argument('--prefix', type=str, default=None,
                        help='the prefix for MLM roberta model')
    parser.add_argument('--ewc', type=str2bool, default=False,
                        help='if using ewc during fine-tuning')
    parser.add_argument('--ewc_importance', type=float, default=1.0,
                        help='ewc weight penalty importance')
    parser.add_argument('--global_step', type=int, default=None,
                        help='starting global step')

    #####################
    ##### CLASSIFIER ####
    #####################
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument("--train_dataset", type=str, nargs="+", default=None)
    parser.add_argument("--eval_dataset", type=str, nargs="+", default=None)
    parser.add_argument("--test_dataset", type=str, nargs="+", default=None)
    parser.add_argument("--toy", type=str2bool, help="Test mode.")
    parser.add_argument("--add_prefix", type=str2bool, default=False,
                        help="add a prefix at the beginning of each input when"
                             " train with multiple dataset")
    parser.add_argument("--add_separator", type=str2bool, default=False,
                        help="add <sep> between sub/rel/obj")
    parser.add_argument("--fix_weights", type=str2bool, default=False,
                        help="fix weight except for the last MLP layer")
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--eval_per_steps', type=int, default=500)
    parser.add_argument('--preprocess_method', choices=['2hop', 'single'],
                        default='2hop', help='the dataset type to preprocess')

    #####################
    ##### GENERALS #####
    #####################
    parser.add_argument('--tb_dir', type=str, default=None,
                        help='the dir to save tbX events')
    parser.add_argument('--word2vec_path', type=str,
                        default='datasets/Word2Vec/GoogleNews-vectors-negative300.bin',
                        help='the path to word2vec bin')
    parser.add_argument('--glove_path', type=str,
                        default=None,
                        help='the path to glove txt file')

    #####################
    ####### MISC ########
    #####################
    return parser


def argparser():
    parser = get_parser()
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    for k in vars(args):
        print (k, getattr(args, k))
