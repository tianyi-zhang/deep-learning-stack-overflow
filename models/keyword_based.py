## implement keyword-based method
import numpy as np
import logging
import csv, os

logger = logging.getLogger()

class KeywordBased:
    def __init__(self, config):
        self.config = config
        self.use_major_label = config.use_major_label
        self.top_keyword = config.top_keyword

    def valid_manual(self, label_dict, vocab):
        vocab_ids = {y: x for x, y in vocab.iteritems()}
        # id_labels = {y:x for x,y in label_dict.iteritems()}
        manual_keyword_dict = {}
        manual_fr = open(os.path.join('.', 'data', 'sto', 'keyword.csv'))
        csv_reader = csv.reader(manual_fr, delimiter='\t')
        for id, row in enumerate(csv_reader):
            if id == 0:
                continue
            else:
                if row[0] == "Comprehension:API":
                    label = label_dict["Comprehension:concept"]
                elif row[0] == "Implementation:API_usage":
                    label = label_dict["Implementation:functionality"]
                elif row[0] == "Performance_Bug":
                    label = label_dict["Performance"]
                elif row[0] == "Migration":
                    label = label_dict["Model_Migration"]
                else:
                    label = label_dict[row[0]]

                keyword_list = []
                try:
                    for i in row[2].split(','):
                        try:
                            keyword_list.append(vocab_ids[i])
                            # manual_keyword_dict[label] = [vocab_ids[i] for i in row[2].split(',')]
                        except KeyError:
                            # print(i)
                            continue
                    if label in manual_keyword_dict:
                        manual_keyword_dict[label] += list(set(keyword_list))
                    else:
                        manual_keyword_dict[label] = keyword_list
                except:
                    manual_keyword_dict[label] = []
        return manual_keyword_dict


    def run(self, data_feed, label_dict, vocab, add_manual=False):
        label_freq_mat = {}
        # new_label_dict = map(label_dict.pop, ["Other", "Architerture:interaction"])
        self.id_label = {y:x for x,y in label_dict.iteritems()}
        for data_ele in data_feed.data:
            # if id_label[data_ele["label"][0]] in ["Other", "Architecture:interaction"]:
            #     continue
            if data_ele["label"] == []:
                continue
            for sub_label in data_ele["label"]:
                if sub_label not in label_freq_mat:
                    label_freq_mat[sub_label] = data_ele["text"]
                else:
                    label_freq_mat[sub_label] += data_ele["text"]


        label_keyword_dict = {}
        label_vocab_weight = {}
        for label, mat in label_freq_mat.items():
            top_ids = mat.argsort()[self.top_keyword:][::-1]
            top_vocabs = [vocab[top_id] for top_id in top_ids]
            label_keyword_dict[label] = top_vocabs
            if label not in label_vocab_weight:
                label_vocab_weight[label] = {}
            label_vocab_weight[label]["vocab_id"] = top_ids
            weight_mat = np.sort(mat)[::-1][:self.top_keyword*(-1)]
            label_vocab_weight[label]["weight_mat"] = weight_mat / np.linalg.norm(weight_mat)

            if add_manual:
                manual_keyword_dict = self.valid_manual(label_dict, vocab)
                manual_ids = manual_keyword_dict[label]
                # label_vocab_weight[label]["manual_mat"] = np.zeros(self.top_keyword*(-1))
                for manual_id in manual_ids:
                    try:
                        label_vocab_weight[label]["weight_mat"][list(top_ids).index(manual_id)] += add_manual
                    except ValueError:
                        print(manual_id)
                        continue
        self.label_keyword_dict = label_keyword_dict
        self.label_vocab_weight = label_vocab_weight



        pred_labels = []
        true_labels = []
        for data_ele in data_feed.data_test:
            # if id_label[data_ele["label"][0]] in ["Other", "Architecture:interaction"]:
            #     continue
            true_labels.append(data_ele["label"])
            pred_label = self._get_vote(data_ele["text"], self.label_vocab_weight)
            if self.id_label[data_ele["label"][0]] in ["Debugging:performance"]:
                print(self.id_label[pred_label[0]])
                # print(data_ele["raw"])
            pred_labels.append(pred_label)

        self._save_pred(data_feed.data_test, pred_labels, self.id_label, valid=True, true_labels=true_labels)
        return self._my_metric(true_labels, pred_labels, label_dict)


    def predict(self, data_feed):
        pred_labels = []
        for data_ele in data_feed.data:
            pred_label = self._get_vote(data_ele["text"], self.label_vocab_weight)
            pred_labels.append(pred_label)
        self._save_pred(data_feed.data, pred_labels,self.id_label)


    def _save_pred(self, test_data, pred_labels, id_label, valid=False, true_labels=None):
        fw = open(os.path.join('.', 'data', 'sto', 'pred_cuiyun.txt'), 'w')
        for idx, data_ele in enumerate(test_data):
            raw = data_ele["raw"]
            # id = data_ele["id"]
            pred_cla = "+".join([id_label[i] for i in pred_labels[idx]])
            if valid:
                true_cla = "+".join([id_label[i] for i in true_labels[idx]])
                fw.write(true_cla+","+pred_cla+","+" ".join(raw)+"\n")
            else:
                fw.write(pred_cla+"\n")   #pred_cla+","+id+"\n"
        fw.close()



    def _get_vote(self, post_bow, label_vocab_weight):
        label_scores = {}
        for label, attrs in label_vocab_weight.items():
            label_scores[label] = sum([post_bow[attrs["vocab_id"][i]]*attrs["weight_mat"][i] for i in range(len(attrs["vocab_id"]))])
        max_value = max(label_scores.values())
        # max_value = sorted(label_scores.values())[-2:]

        ## if no overlap between manual_mat and the bow
        # if max_value == 0:
        #     for label, attrs in label_vocab_weight.items():
        #         label_scores[label] = sum(
        #             [post_bow[attrs["vocab_id"][i]] * attrs["weight_mat"][i] for i in range(len(attrs["vocab_id"]))])
        # max_value = max(label_scores.values())

        pred_label = [key for key , value in label_scores.items() if value == max_value]
        return pred_label

    def _my_metric(self, true_labels, pred_labels, label_dict):
        assert len(true_labels) == len(pred_labels)
        label_len = len(label_dict)
        conf_matrix = np.zeros((label_len, label_len), dtype=float)
        for i in range(len(true_labels)):
            for j in range(len(pred_labels[i])):
                if pred_labels[i][j] in true_labels[i]:
                    conf_matrix[pred_labels[i][j], pred_labels[i][j]] += 1
                else:
                    # for label in true_labels:
                        # conf_matrix[label, pred_labels[i][j]] += 1.0/len(true_labels)
                    if not conf_matrix[true_labels[i][0], pred_labels[i][j]]:
                        conf_matrix[true_labels[i][0], pred_labels[i][j]] += 1.0


        precisions = []
        recalls = []
        for j in range(conf_matrix.shape[0]):
            if np.sum(conf_matrix[:, j]) != 0:
                precisions.append(conf_matrix[j,j] / np.sum(conf_matrix[:, j]))
            else:
                precisions.append(1.0)
            if np.sum(conf_matrix[j]) != 0:
                recalls.append(conf_matrix[j,j] / np.sum(conf_matrix[j]))
            else:
                recalls.append(1.0)
        precision = float(np.mean(precisions))
        recall = float(np.mean(recalls))
        # for i in range(len(true_labels)):
        #     if true_labels[i] == pred_labels[i]:
        #         precisions.append(1)
        #         recalls.append(1)
        #     else:
        #         inters_len = (set(true_labels[i]) & set(pred_labels[i])).__len__()
        #         precisions.append(inters_len/float(len(pred_labels[i])))
        #         recalls.append(inters_len/float(len(true_labels[i])))
        # precision = sum(precisions)/float(len(precisions))
        # recall = sum(recalls)/float(len(recalls))
        f1 = 2 * precision * recall / (precision + recall)
        self._save_file(label_dict, precisions, recalls)
        logger.info("precision: %.6f, recall: %.6f, f1: %.6f" % (precision, recall, f1))

    def _save_file(self, label_dict, precisions, recalls):
        id_labels = {y:x for x,y in label_dict.iteritems()}
        result_fw = open(os.path.join('.', 'data', 'sto', 'result_cuiyun.csv'), 'w')
        csv_fields = ['label', 'keyword', 'precision', 'recall', 'fmeasure']
        writer = csv.DictWriter(result_fw, fieldnames=csv_fields)
        writer.writeheader()
        for key, value in self.label_keyword_dict.items():
            row = {'label': id_labels[key], 'keyword': ','.join(value), 'precision':precisions[key], 'recall':recalls[key], 'fmeasure':2 * precisions[key] * recalls[key] / (precisions[key] + recalls[key])}
            writer.writerow(row)
        result_fw.close()




