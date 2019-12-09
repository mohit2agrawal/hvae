# coding=utf-8
from nltk import jaccard_distance as jd
from nltk.translate.bleu_score import sentence_bleu as sb
import numpy as np
from tqdm import tqdm
import sys
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))
weights = (0.5, 0.5, 0, 0)

# input_sents_fn = 'pos/data_mod.txt'
# input_para_fn = 'generated_sentences.txt'
# input_para_labels_fn = 'generated_labels.txt'

input_sents_fn, input_para_fn, input_para_labels_fn = sys.argv[1:4]

output_fn = input_para_fn.replace('.txt', '.selected.txt')
output_labels_fn = input_para_labels_fn.replace('.txt', '.selected.txt')

NUM_SENTS = 2500
NUM_PARA = 100
OUPUT_NUM = 1  ## top N paras per sentence

input_sents = []
input_sents_orig = []
with open(input_sents_fn) as f:
    for i in range(NUM_SENTS):
        s = f.readline().strip().split(' ')
        input_sents_orig.append(s)
        # print(s)
        input_sents.append(
            [
                lemmatizer.lemmatize(x.replace('\xe2\x80\x99', "'")) for x in s
                if x not in stopWords
            ]
        )

input_paras = []
input_labels = []
similarities = []
means = []
maxs = []
with open(input_para_fn) as f, open(input_para_labels_fn) as lf:
    for i in tqdm(range(NUM_SENTS)):
        i_sent = input_sents[i]
        i_sent_set = set(i_sent)

        paras = []
        labels = []
        sims = []

        ## original sentences are also written. escape them first.
        f.readline()
        lf.readline()
        for j in range(NUM_PARA):
            para_sent = f.readline().strip().split(' ')
            para_sent_ = [
                lemmatizer.lemmatize(x.replace('\xe2\x80\x99', "'"))
                for x in para_sent if x not in stopWords
            ]
            paras.append(para_sent)
            labels.append(lf.readline())
            sims.append(sb([i_sent], para_sent_, weights=weights))

        input_paras.append(paras)
        input_labels.append(labels)
        similarities.append(sims)
        mean = np.mean(sims)
        max_ = max(sims)
        means.append(mean)
        maxs.append(max_)
        print(i, mean, max_)

        ## should get a blank line after NUM_PARA sentences
        if i != NUM_SENTS - 1:
            blank = f.readline().strip()
            assert blank == ''
            blank = lf.readline().strip()
            assert blank == ''

print('mean of means:', np.mean(means))
print('std of means:', np.std(means))
print()
print('mean of maxs:', np.mean(maxs))
print('std of maxs:', np.std(maxs))

## similarities, higher is better
# with open(output_fn, 'w') as f, open(output_labels_fn, 'w') as lf:
#     for i in range(NUM_SENTS):
#         scores = similarities[i]
#         indices = np.argsort(scores)[::-1]
#         for j in range(OUPUT_NUM):
#             f.write(' '.join(input_paras[i][indices[j]]))
#             f.write('\n')
#             lf.write(input_labels[i][indices[j]])