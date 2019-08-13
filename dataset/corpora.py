import numpy as np
import json
from gensim.corpora import Dictionary
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import TfidfModel
from pymining import itemmining

import copy
import logging
import os
import utils
import nltk
import re

DIGIT = "<digit>"
VARIABLE = "<var>"
FUNCTION = "<func>"
PAD = "<pad>"

class STOCorpus(object):
    def __init__(self, config):
        self.config = config
        self._path = config.data_path
        self.max_utt_len = config.max_utt_len
        self.tokenize = self._get_tokenize()
        self.lemm = WordNetLemmatizer().lemmatize
        self.label_dict = {}
        self.valid, self.non_valid = self._read_file(self._path)
        # self.phrases = self._build_phrase()
        self._build_vocab(config.max_vocab_cnt)

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for data in self.valid + self.non_valid:
            all_words.append(data["title"] + data["content"])
        vocab = Dictionary(all_words)
        raw_vocab_size = len(vocab)

        vocab.filter_extremes(no_below=5)
        vocab.filter_extremes(keep_n=max_vocab_cnt)
        len_1_words = list(filter(
            lambda w: len(w) == 1 and re.match(r"[\x00-\x7f]", w) and w not in ["a", "i"] and True or False,
            vocab.values()))
        vocab.filter_tokens(list(map(vocab.token2id.get, len_1_words)))
        if self.config.use_dict == "seq" and self.config.enable_pad:
            vocab.token2id[PAD] = len(vocab)
            vocab.compactify()
            self.pad_wid = vocab.token2id.get(PAD)
        self.vocab_seq = vocab  # seq dictionary
        # build bow dictionary
        self.vocab_bow = copy.deepcopy(vocab)
        self.vocab_bow.filter_tokens(map(self.vocab_bow.token2id.get, STOPWORDS))   # filter stop words
        self.vocab_bow.compactify()
        if self.config.tfidf:
            tfidf_corpus = [self.vocab_bow.doc2bow(line) for line in all_words]
            self.tfidf_model = TfidfModel(tfidf_corpus)
        print("Load corpus with non_valid size %d, valid size %d, "
              "raw vocab size %d seq vocab size %d, bow vocab size %d"
              % (len(self.non_valid), len(self.valid),
                 raw_vocab_size, len(self.vocab_seq), len(self.vocab_bow)))

    def _build_phrase(self):
        input_list = {}
        for post in self.valid:
            for cate  in post["category"]:
                if cate not in input_list:
                    input_list[cate] = []
                input_list[cate].append(post["title"] + post["content"])

        phrases = {}
        for cate, posts in input_list.items():
            relim_input = itemmining.get_relim_input(posts)
            fis = itemmining.relim(relim_input, min_support=2)
            phrases[cate] = {}
            for phrase, count in fis.items():
                new_p = list(phrase)
                if len(new_p) >= 2:
                    phrases[cate]['_'.join(new_p)] = count
            print(cate)
            print(phrases[cate])

        phrase_json = open(os.path.join("../data/sto/phrase_dict.json"), 'w')
        json.dump(phrase_json, phrases)
        phrase_json.close()
        return phrases

    def _process_data(self, data_lst):
        all_data = []
        all_lens = []
        cate_count = 0
        for data in data_lst:
            if self.config.lemm:
                title = list(map(lambda x: self.lemm(x, "v"), self.tokenize(data["title"].lower())))
                content = list(map(lambda x: self.lemm(x, "v"), self.tokenize(data["content"].lower())))
            else:
                title = self.tokenize(data["title"].lower())
                content = self.tokenize(data["content"].lower())
            id = data["id"]

            if "category" in data:
                labels = []
                for cate in data["category"]:
                    if self.config.remove_other and cate in ["Other", "Architecture:interaction"]:
                        continue
                    if self.config.use_major_label:
                        cate = cate.split(":")[0]   # major label
                    if cate not in self.label_dict:
                        self.label_dict[cate] = len(self.label_dict)
                    labels.append(self.label_dict[cate])
                    cate_count += 1
                category = labels
            else:
                category = None
            all_data.append({"title": title,
                             "content": content,
                             "id": id,
                             "category": category})
            all_lens.append(len(title) + len(content))
        if not self.config.remove_other:
            self.config.other_id = self.label_dict["Other"]     # record other_id
        print("Max post len %d, mean post len %.2f" % (np.max(all_lens), float(np.mean(all_lens))))
        print("No. of data with label is %d" % cate_count)
        return all_data

    def _sent2id(self, sent, vocab):
        return filter(lambda x: x is not None, [vocab.token2id.get(t) for t in sent])

    def _sent2id_bow(self, sent, vocab):
        return vocab.doc2bow(sent)

    def _to_corpus_seq(self, data, vocab):
        word_cnt = 0
        msg_cnt = 0
        curpus = []
        for post in data:
            new_post = {
                "title": self._sent2id(post["title"], vocab),
                "content": self._sent2id(post["content"], vocab),
                "category": post["category"],
                "id": post["id"],
                "raw": post["title"] + post["content"]
            }
            if new_post["content"] or new_post["title"]:
                curpus.append(new_post)
                word_cnt += len(new_post["content"]) + len(new_post["title"])
                msg_cnt += 1

        print("Load seq with %d posts, %d words" % (msg_cnt, word_cnt))
        return {"corpus": curpus, "msg_cnt": msg_cnt, "word_cnt": word_cnt}


    def _to_corpus_bow(self, data, vocab):
        word_cnt = 0
        msg_cnt = 0
        curpus = []
        for post in data:
            new_post = {
                "title": self._sent2id_bow(post["title"], vocab),
                "content": self._sent2id_bow(post["content"], vocab),
                "category": post["category"],
                "id": post["id"],
                "raw": post["title"] + post["content"]
            }
            if new_post["content"] or new_post["title"]:
                curpus.append(new_post)
                word_cnt += len(new_post["content"]) + len(new_post["title"])
                msg_cnt += 1
        if self.config.tfidf:
            for post in curpus:
                post["title"] = self.tfidf_model[post["title"]]
                post["content"] = self.tfidf_model[post["content"]]

        print("Load bow with %d posts, %d words" % (msg_cnt, word_cnt))
        return {"corpus": curpus, "msg_cnt": msg_cnt, "word_cnt": word_cnt}

    def _read_file(self, path):
        with open(path) as f:
            data = json.load(f)
        return self._process_data(data["valid"]), self._process_data(data["non_valid"])

    def _get_tokenize(self):
        return nltk.RegexpTokenizer(r'\w+|<digit>|<var>|<func>').tokenize


    def get_corpus(self):
        if self.config.use_dict == "seq":
            vocab = self.vocab_seq
            valid_corpus = self._to_corpus_seq(self.valid, vocab)
            non_valid_corpus = self._to_corpus_seq(self.non_valid, vocab)
        else:
            vocab = self.vocab_bow
            valid_corpus = self._to_corpus_bow(self.valid, vocab)
            non_valid_corpus = self._to_corpus_bow(self.non_valid, vocab)
        return {"valid": valid_corpus, "non_valid": non_valid_corpus, "vocab": vocab, "label_dict": self.label_dict}