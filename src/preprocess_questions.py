import os
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from itertools import groupby
from openpyxl import load_workbook
from process_data import DataLoader
import json

special_words = ['prosses', 'acsses']
stop_words = set(stopwords.words('english'))

class ProcessQues():
    def __init__(self, fr, ques_id, ques_cate):
        self.ques_dict = {}
        self.questions = fr.readlines()
        self.ques_dict['ques_id'] = ques_id
        self.ques_dict['ques_cate'] = ques_cate
        self.title = re.sub('Title: ', '', self.questions[0])
        self.body = self.questions[2:]

    def _check_pattern(self, token):
        method_patterns = [re.compile('\w+\.+'), re.compile('\w+\_+')]
        for method_pattern in method_patterns:
            if re.search(method_pattern, token):
                return '<method>'
            elif re.search('\_', token):
                return '<method>'
            else:
                return token

    def _basic_process(self, terms):
        for term_id, term in enumerate(terms):
            terms[term_id] = re.sub('[\"\(\)\`\=\:\[\]\-\,\?\+\/]*', ' ', term)
            if re.search(re.compile("(\w+)\'s"), terms[term_id]):
                terms[term_id] = terms[term_id].split("'")[0]
            terms[term_id] = re.sub("[\']*", '', terms[term_id])
            terms[term_id] = re.sub(' +', '', terms[term_id])
            # terms[term_id] = self._check_pattern(terms[term_id])
            terms[term_id] = WordNetLemmatizer().lemmatize(terms[term_id], 'v')
            if terms[term_id] not in special_words:
                temp_token = WordNetLemmatizer().lemmatize(terms[term_id], 'n')
                if terms[term_id] != temp_token and not re.search(r'ss$', terms[term_id]):
                    terms[term_id] = temp_token
            if terms[term_id] in stop_words or len(terms[term_id]) == 1:
                terms[term_id] = ''
            if re.match(re.compile("(\d+)d$"), terms[term_id]):  # $ is the end of string
                terms[term_id] = '<dim>'
            elif re.match(re.compile("(\d+)gb$"), terms[term_id]):  # $ is the end of string
                terms[term_id] = '<mem>'
            elif terms[term_id].isdigit():
                terms[term_id] = '<digit>'

        new_terms = [x[0] for x in groupby(terms)]
        new_terms = ' '.join(new_terms)
        new_terms = re.sub(r"[^0-9a-z,.?!<>' ]", ' ', new_terms)
        new_terms = re.sub('\.', ' ', new_terms)
        new_terms = re.sub(' +', ' ', new_terms)
        return new_terms

    def _process_body(self):
        lines = ' '.join([x.strip() for x in self.body])
        line = lines.lower()
        line = re.sub(r'<code[^>]*>([^<]+)</code>', '', line)   # remove <code> block
        new_line = re.findall(r'>([^<>]*)<', line)
        new_line = ' '.join(new_line)
        new_line = re.sub(r'[^\x00-\x7F]+', '', new_line)
        new_title = re.sub(r'[^\x00-\x7F]+', '', self.title.strip().lower())
        self.ques_dict['body'] = self._basic_process(new_line.split()).strip()
        self.ques_dict['title'] = self._basic_process(new_title.split())
        return self.ques_dict


def _get_question_ids():
    wb = load_workbook(filename=os.path.join('..', 'data', 'Manual_Inspection.xlsx'))
    _, _, tf_id_cate_dict1 = DataLoader(wb, 'TensorFlow(random)')._get_questions()
    _, _, tf_id_cate_dict2 = DataLoader(wb, 'TenserFlow(top-view)')._get_questions()
    _, _, pt_id_cate_dict1 = DataLoader(wb, 'PyTorch(random)')._get_questions()
    _, _, pt_id_cate_dict2 = DataLoader(wb, 'PyTorch(top-view)')._get_questions()
    _, _, dj_id_cate_dict1 = DataLoader(wb, 'DeepLearning4J(random)')._get_questions()
    _, _, dj_id_cate_dict2 = DataLoader(wb, 'DeepLearning4J(top-view)')._get_questions()
    tf_id_cate_dict = tf_id_cate_dict1 + tf_id_cate_dict2
    pt_id_cate_dict = pt_id_cate_dict1 + pt_id_cate_dict2
    dj_id_cate_dict = dj_id_cate_dict1 + dj_id_cate_dict2

    return tf_id_cate_dict, pt_id_cate_dict, dj_id_cate_dict


if __name__ == "__main__":
    tf_id_cate_dict, pt_id_cate_dict, dj_id_cate_dict = _get_question_ids()
    id_cate_dict = tf_id_cate_dict + pt_id_cate_dict + dj_id_cate_dict
    question_dir = os.path.join('..', 'data', 'questions')

    ques_dict = []
    count = 0
    for (ques_id, ques_cate) in id_cate_dict:
        question_fr = open(os.path.join(question_dir, ques_id+'.txt'))
        ques_data = ProcessQues(question_fr, ques_id, ques_cate)._process_body()
        ques_dict.append(ques_data)
        question_fr.close()
        count+=1
        if count%20 == 0:
            print(ques_data)
            print('Finish %d questions.'%count)
            break

    fw = open(os.path.join('..', 'result', 'ques_dict.json'), 'w')
    json.dump(ques_dict, fw, indent=4)
    fw.close()



