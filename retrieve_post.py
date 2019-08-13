### Retrieve posts related to "Debugging: runtime inspection", "Architecture: CPU/GPU", "Architecture: distributed"


import argparse
from utils import str2bool, prepare_dirs_loggers, get_time, process_config, plot_conf_mat
import os
from dataset import corpora, data_loaders
import logging, csv
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
model_arg.add_argument('--k', type=int, default=11)
model_arg.add_argument('--sample_size', type=int, default=20)
model_arg.add_argument('--confident', type=float, default=0.5)
model_arg.add_argument('--add_manual', type=float, default=0.12)
model_arg.add_argument('--forward_only', type=str2bool, default=False)
model_arg.add_argument('--load_sess', type=str, default="2018-10-20T15-50-54-topic_disc.py")
model_arg.add_argument('--token', type=str, default="")
model_arg.add_argument('--top_keyword', type=int, default=-200)
model_arg.add_argument('--target_label', type=str, default='Architecture:distributed')

def get_manual_keyword():
    manual_keyword_dict = {}
    manual_fr = open(os.path.join('.', 'data', 'sto', 'tianyi.csv'))
    csv_reader = csv.reader(manual_fr, delimiter='\t')
    for id, row in enumerate(csv_reader):
        if id == 0:
            continue
        label = row[0]
        keyword_list = row[2].split(',')
        if label in manual_keyword_dict:
            manual_keyword_dict[label] += list(set(keyword_list))
        else:
            manual_keyword_dict[label] = keyword_list
    return manual_keyword_dict

def _get_relates(target_label, data_list, keyword_dict):
    related_scores = []
    post_ids = []
    for line in data_list:
        post_raw = line['raw']
        post_id = line['id']
        keyword_list = keyword_dict[target_label]
        score = (set(keyword_list)&set(post_raw)).__len__()
        related_scores.append(score)
        post_ids.append(post_id)
    top_ids = sorted(range(len(related_scores)), key=lambda i: related_scores[i])[-100:]
    return [post_ids[id] for id in top_ids]


def main(config):
    prepare_dirs_loggers(config, os.path.basename(__file__))
    logger = logging.getLogger()

    corpus_client = corpora.STOCorpus(config)

    corpus = corpus_client.get_corpus()
    non_valid_corpus, valid_corpus, vocab, label_dict = corpus['non_valid'], corpus['valid'], corpus['vocab'], corpus["label_dict"]
    manual_keyword_dict = get_manual_keyword()

    target_label = config.target_label
    related_posts = _get_relates(target_label, non_valid_corpus['corpus'], manual_keyword_dict)
    print(related_posts)


if __name__ == "__main__":
    config, unparsed = get_config()
    config = process_config(config)
    main(config)
