import os
import re
import logging
import json
import sys
from datetime import datetime
from collections import Counter
import nltk
import numpy as np
from argparse import Namespace

import matplotlib.pyplot as plt

def process_config(config):
    if config.forward_only:
        load_sess = config.load_sess
        beam_size = config.beam_size
        gen_type = config.gen_type

        load_path = os.path.join(config.log_dir, load_sess, "params.json")
        config = load_config(load_path)
        config.forward_only = True
        config.load_sess = load_sess
        config.beam_size = beam_size
        config.gen_type = gen_type
        config.batch_size = 20
    return config


def prepare_dirs_loggers(config, script=""):
    logFormatter = logging.Formatter("%(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    if config.forward_only:
        return

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    dir_name = "{}-{}".format(get_time(), script) if script else get_time()
    if config.token:
        config.session_dir = os.path.join(config.log_dir, dir_name + "_" + config.token)  # append token
    else:
        config.session_dir = os.path.join(config.log_dir, dir_name)
    os.mkdir(config.session_dir)

    fileHandler = logging.FileHandler(os.path.join(config.session_dir,
                                                   'session.log'))
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # save config
    param_path = os.path.join(config.session_dir, "params.json")
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)
        print("Save params in "+param_path)


def get_embed_mat(path, vocab):
    with open(path) as fin:
        embedding_index = {}
        first_line = True
        l_id = 0
        for line in fin:
            if l_id % 100000 == 0:
                print("loaded %d words embedding..." % l_id)
            if first_line:
                first_line = False
                continue
            line = line.rstrip()
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
            l_id += 1
        embedding_mat = np.zeros((len(vocab), len(embedding_index.values()[0])), dtype='float32')  # 0 is for padding
        for i, word in vocab.items():
            embedding_vec = embedding_index.get(word)
            if embedding_vec is not None:
                embedding_mat[i] = embedding_vec
    return embedding_mat


def plot_conf_mat(conf_mat, label_dict):
    label_str_lst = []

    for k, v in sorted(label_dict.items(), key=lambda kv: kv[1]):
        label_str_lst.append(k)

    # plot
    fig, ax = plt.subplots()
    im = ax.imshow(conf_mat)
    ax.set_xticks(np.arange(len(label_str_lst)))
    ax.set_yticks(np.arange(len(label_str_lst)))
    ax.set_xticklabels(label_str_lst)
    ax.set_yticklabels(label_str_lst)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.show()


def load_config(load_path):
    data = json.load(open(load_path, "rb"))
    config = Namespace()
    config.__dict__ = data
    return config


def get_time():
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

def str2bool(v):
    return v.lower() in ('true', '1')
