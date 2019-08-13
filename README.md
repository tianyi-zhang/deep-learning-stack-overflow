# deep-learning-stack-overflow
Dataset and replication package for the empirical study of programming challenges of building deep learning applications (ISSRE 2019)

## Usage
Run the code with
```
$ python run_model.py
```

You can change the configures in `run_model.py`, including the `data_path`, `log_dir`, `max_utt_len`, etc. The important parameters are 
```
add_manual -- defines the weights of manual keywords in class prediction
top_keyword -- means the number of top words to consider (ranked by tf-idf)
only_title -- means whether or not involve post content or only title
use_major_label -- indicates whether use the first label or all the labels for a post
lemm -- whether or not do lemmatization
remove_other -- whether remove the posts labeled with "Other" category
```

## Dataset

All the data used in this study can be downloaded from this [link](https://drive.google.com/open?id=1bKCX2raszVo7BVGhIkY_Xlt3EoXgHq0k). You can put the folder in the master directory, i.e., `../data`. The `dl_framework_stackoverflow_questions` file contains all the 39,628 studied deap-learning-related questions on Stack Overflow. The `Manual_Inspection.xlsx` file iincludes the labeled 715 questions on three deep learning platforms. The `meta-data.csv` file saves the meta data for each question, such as the response time. The manually labeled keywords are listed in `../data/sto/keyword.csv`. Other files are generated files during running.

## Citation
Please cite our ISSRE paper if you use our method / datasets in your work

```
@inproceedings{zgmlk2019stoissre,
    title={An empirical study of common challenges in developing deep learning applications},
    author={Tianyi Zhang and Cuiyun Gao and Lei Ma and Michael R. Lyu and Miryung Kim},
    booktitle={Proceedings of the 30th International Symposium on Software Reliability Engineering, {ISSRE} 2019},
    year={2019}
}
```