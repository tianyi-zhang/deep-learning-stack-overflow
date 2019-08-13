import os
import csv, sys
import json
import codecs

input_fr = open(os.path.join('..', 'data', 'topic_doc_dict.json'), 'rb')
input_dict = json.load(input_fr)
input_fr.close()

for topic_id, docs in input_dict.items():
    topic_fw = codecs.open(os.path.join('..', 'data', str(topic_id)+'.txt'), "w", encoding='utf-8')
    topic_fw.writelines([doc+'\n' for doc in docs])
    topic_fw.close()