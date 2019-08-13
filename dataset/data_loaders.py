from __future__ import print_function
import numpy as np
import copy
import itertools
from dataset.dataloader_bases import DataLoader
from utils import get_embed_mat
from sklearn import cross_validation
import pickle, os



class KnnDataLoader(object):
    def __init__(self, name, data, vocab, config):
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.config = config
        self.vocab = vocab
        data_posts, m_cnt, w_cnt = data["corpus"], data["msg_cnt"], data["word_cnt"]
        self.data_size = len(data_posts)
        new_data = []
        all_labels = []

        if config.use_dict == "seq":
            # use embedding
            embed_mat = get_embed_mat(config.embed_path, vocab)
            for post in data_posts:
                if post["category"] == []:
                    continue
                if config.only_title:
                    data_post = self._seq2vec(post["title"], embed_mat)
                else:
                    data_post = self._seq2vec(post["title"] + post["content"], embed_mat)

                new_data.append({"text": data_post, "label": post["category"]})
        elif config.use_dict == "phrase":
            # use phrase sets
            pass
        else:
            # use bow
            for post in data_posts:
                if post["category"] == []:
                    continue
                if config.only_title:
                    data_post = self._bow2vec(post["title"], len(vocab))
                else:
                    data_post = self._bow2vec(post["title"] + post["content"], len(vocab))
                try:
                    if len(post["category"])  > 1:
                        for label in post["category"]:
                            all_labels.append(label)
                            new_data.append({"text": data_post, "label": post["category"], "raw": post["raw"]})
                    else:
                        all_labels.append(post["category"][0])
                    new_data.append({"text": data_post, "label": post["category"], "raw": post["raw"], "id": post["id"]})
                except TypeError:
                    new_data.append({"text": data_post, "label": None, "raw": post["raw"], "id": post["id"]})


        if name == "valid":
            # split into train and test
            self.data, self.data_test = self._split_shuffle(new_data, all_labels)
            # # balance sampling
            # data_dict = {}
            # sampled_data = []
            # for post in self.data:
            #     for label in post["label"]:
            #         if label not in data_dict:
            #             data_dict[label] = []
            #         data_dict[label].append(post)
            # for label, posts in data_dict.items():
            #     new_posts = np.random.choice(posts, config.sample_size)
            #     sampled_data.append(list(new_posts))
            # self.data = list(itertools.chain.from_iterable(sampled_data))  # no need to shuffle
        elif name == "manual":
            self.data, self.data_test = self._get_expdata()
        else:
            self.data = new_data

    def _get_expdata(self):
        exp_fr = open(os.path.join(".", "data", "sto", "exp_data.json"), "rb")
        exp_data = pickle.load(exp_fr)
        exp_fr.close()
        data_train, data_test = exp_data["data_train"], exp_data["data_test"]
        print("Len of training data is ", len(data_train), "; Len of test data is ", len(data_test))
        return data_train, data_test

    def _split_shuffle(self, data, all_labels):
        # find out other
        other_indices = []
        ## extract ids of posts classified as "Other"
        if not self.config.remove_other:
            for i, ele in enumerate(data):
                for label in ele["label"]:
                    if label == self.config.other_id:
                        other_indices.append(i)
        all_indices = np.arange(len(data))
        re_indices = list(set(all_indices) - set(other_indices))
        ## extract samples according to distributions of classes
        skf = cross_validation.StratifiedShuffleSplit(all_labels, 2, test_size=0.4, random_state=0)
        for train_index, test_index in skf:
            # X_train, X_test = re_indices[train_index], re_indices[test_index]
            data_train = [data[id] for id in train_index]
            data_test = [data[id] for id in test_index]
        exp_data = {"data_train": data_train, "data_test": data_test}
        exp_fw = open(os.path.join(".", "data", "sto", "exp_data.json"), "wb")
        pickle.dump(exp_data, exp_fw)
        exp_fw.close()

        train_dict = {}
        for data_ele in data_train:
            for sub_label in data_ele["label"]:
                if sub_label not in train_dict:
                    train_dict[sub_label] = 0
                train_dict[sub_label] += 1
        print("training data ", train_dict)

        test_dict = {}
        for data_ele in data_test:
            for sub_label in data_ele["label"]:
                if sub_label not in test_dict:
                    test_dict[sub_label] = 0
                test_dict[sub_label] += 1
        print("test data ", test_dict)


        print("Len of training data is ", len(data_train), "; Len of test data is ", len(data_test))

        ## random shuffle
        # np.random.shuffle(re_indices)
        # num_test = int(len(re_indices) * 0.2)
        #
        # data_train = [data[id] for id in re_indices[:-num_test]]
        # data_test = [data[id] for id in re_indices[-num_test:] + other_indices]

        return data_train, data_test


    def _bow2vec(self, bow, vec_size):
        vec = np.zeros(vec_size, dtype=float)
        for id, val in bow:
            vec[id] = val
        return vec

    def _seq2vec(self, seq, embed_mat):  # average embedding
        lst = []
        for wid in seq:
            lst.append(embed_mat[wid])
        return np.mean(np.array(lst), 0)


    def _pad(self, max_len, token, pad_wid):
        pad_vec = np.full(max_len, pad_wid)
        pad_vec[:len(token)] = token[:min(len(token), max_len)]
        return pad_vec