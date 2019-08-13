import argparse
from utils import str2bool, prepare_dirs_loggers, get_time, process_config, plot_conf_mat
import os
from dataset import corpora, data_loaders
import logging
from models.keyword_based import KeywordBased
arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_path', type=str, default='data/sto/sto_data.json')
data_arg.add_argument('--log_dir', type=str, default='logs')
data_arg.add_argument('--max_utt_len', type=int, default=100)
data_arg.add_argument('--max_vocab_cnt', type=int, default=20000)
data_arg.add_argument('--enable_pad', type=str2bool, default=False)
data_arg.add_argument('--use_dict', type=str, default="bow")
data_arg.add_argument('--tfidf', type=str2bool, default=False)
data_arg.add_argument('--remove_other', type=str2bool, default=True)
data_arg.add_argument('--lemm', type=str2bool, default=True)
data_arg.add_argument('--only_title', type=str2bool, default=False)
data_arg.add_argument('--embed_path', type=str, default="C:\Users\jichuanzeng\Documents\glove.6B\glove.6B.200d.txt")
data_arg.add_argument('--use_major_label', type=str2bool, default=False)

# Model
model_arg = add_argument_group('Model')
model_arg.add_argument('--add_manual', type=float, default=0.2)
model_arg.add_argument('--token', type=str, default="")
model_arg.add_argument('--forward_only', type=str2bool, default=False)
model_arg.add_argument('--top_keyword', type=int, default=-200)


def main(config):
    prepare_dirs_loggers(config, os.path.basename(__file__))
    logger = logging.getLogger()

    corpus_client = corpora.STOCorpus(config)

    corpus = corpus_client.get_corpus()
    non_valid_corpus, valid_corpus, vocab, label_dict = corpus['non_valid'], \
                                            corpus['valid'], corpus['vocab'], corpus["label_dict"]
    print("label_dict is ", label_dict)
    valid_feed = data_loaders.KnnDataLoader("valid", valid_corpus, vocab, config)  # for keyword extraction
    # valid_feed = data_loaders.KnnDataLoader("valid", valid_corpus, vocab, config)
    non_valid_feed = data_loaders.KnnDataLoader("non_valid", non_valid_corpus, vocab, config)


    ### for keyword extraction
    model = KeywordBased(config)
    model.run(valid_feed, label_dict, vocab, config.add_manual)
    model.predict(non_valid_feed)

if __name__ == '__main__':
    config, unparsed = get_config()
    config = process_config(config)
    main(config)
