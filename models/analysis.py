import os, pickle

fr = open(os.path.join('..', 'data', 'sto', 'exp_data.json'))
exp_data = pickle.load(fr)
fr.close()

data_train = exp_data['data_train']
data_test = exp_data['data_test']

train_label_num = {}
for data_ele in data_train:
    label = data_ele['label']
    for l in label:
        if l not in train_label_num:
            train_label_num[l] = 0
        train_label_num[l] += 1

test_label_num = {}
for data_ele in data_test:
    label = data_ele['label']
    for l in label:
        if l not in test_label_num:
            test_label_num[l] = 0
        test_label_num[l] += 1

print(train_label_num)
print(test_label_num)