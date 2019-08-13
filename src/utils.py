import os
import re
import logging
from hierarchical_clustering import Vectorize
import json
from collections import Counter

logger = logging.getLogger(__name__)

def load_valid_data(fn):
    # do some filtering
    logger.info("Loading valid data")
    valid_json = json.load(open(fn))
    data_lst = []
    label_lst = []
    label_dict = {}
    label_counter = Counter()
    filter_label = []
    for it in valid_json['data']:
        for l in it["labels"]:
            label_counter[l] += 1
    for key, val in label_counter.items():
        if val < 5:
            filter_label.append(key)

    for it in valid_json['data']:
        skip = False
        for l in it["labels"]:
            if l in filter_label:   # filter labels
                skip = True
                break
            if l not in label_dict:
                label_dict[l] = len(label_dict)
        if not skip:
            data_lst.append(it['text'])
            label_lst.append(list(map(label_dict.get, it["labels"])))
    return data_lst, label_lst, label_dict


def _get_label(label):
    label = label.strip()
    if label in ['Architecutre:CPU/GPU', 'Architecture:CPU', 'Architecture:GPU/CPU', 'Architecture:GPU']:
        return 'Architecture:CPU/GPU'
    elif label in ['build failure', 'install']:
        return 'build/install'
    elif label in ['Implemenation:API usage']:
        return 'Implementation:API_usage'
    else:
        return re.sub(r'\s', '_', label)


def build_valid_file(fn):
    label_dict = {}
    valid_json = {"dataset": "valid"}
    data = []
    with open(fn) as fin:
        lines = fin.readlines()
        for line in lines:
            text, label_str = line.strip().split('\t')
            if label_str.count(':') < 2:    # only one label
                labels = [_get_label(label_str)]
            else:   # two labels
                labels = list(map(_get_label, label_str.split(" ", 1)))
            for label in labels:
                if label not in label_dict:
                    label_dict[label] = len(label_dict)
            data.append({
                "text": text,
                "labels": labels
            })
    for key in label_dict.keys():
        print(key)
    valid_json['data'] = data
    with open("valid.json", "w") as fout:
        json.dump(valid_json, fout, indent=4)

def get_tfidf_feat(input_docs):
    """
    get tf-idf features
    :param input_docs: list of
    :return:
    """
    doc_tfidfs = Vectorize(input_docs).doc_vecs
    return doc_tfidfs