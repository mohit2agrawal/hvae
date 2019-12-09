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

dict_fn, input_sents_fn, input_para_fn = sys.argv[1:4]

output_fn = input_para_fn.replace('.txt', '.selected.txt')

NUM_SENTS = 2500
NUM_PARA = 100
OUPUT_NUM = 1  ## top N paras per sentence

word2idx = dict()
lines = open(dict_fn).readlines()
for line in lines:
    tokens = line.split()
    idx = int(tokens[1])
    word = tokens[0].strip()
    word2idx[word] = idx

input_sents = []
input_sents_orig = []
with open(input_sents_fn) as f:
    for i in range(NUM_SENTS):
        s = f.readline().strip().split(' ')
        input_sents_orig.append(s)
        # print(s)
        input_sents.append(s)

input_sents_unk = [
    [y if y in word2idx else '<unk>' for y in x] for x in input_sents
]

input_sents = []
for sent in input_sents_unk:
    # print(sent)
    input_sents.append(
        [lemmatizer.lemmatize(x.replace('\xe2\x80\x99', "'").replace('\xe2\x80\x93', '-').replace('\xe2\x99\xa5', '<unk>')) \
        for x in sent if x not in stopWords]
    )
print(len(input_sents))

input_paras = []
similarities = []
means = []
maxs = []
with open(input_para_fn) as f:
    for i in tqdm(range(NUM_SENTS)):
        i_sent = input_sents[i]
        i_sent_set = set(i_sent)

        paras = []
        sims = []

        ## original sentences are also written. escape them first.
        # f.readline()
        for j in range(NUM_PARA):
            para_sent = f.readline().strip().split(' ')
            print(para_sent)
            para_sent_ = [
                lemmatizer.lemmatize(
                    x.replace('\xe2\x80\x99',
                              "'").replace('\xe2\x80\x93', '-').replace(
                                  '\xe2\x99\xa5', '<unk>'
                              ).replace('\xc2\xb4',
                                        "'").replace('\xe2\x98\xba', '<unk>').
                    replace('\xe2\x97\x99',
                            '<unk>').replace('\xc3\xb9',
                                             'u').replace('\xc3\xa9', 'e').replace('\xc3\xa7', 'c').replace('\xc3\xa0', 'a').replace('\xe2\x80\x9d', '"').replace('\xc3\xa8', 'e').replace('\xe2\x80\xa2','')
                ) for x in para_sent if x not in stopWords
            ]
            paras.append(para_sent)
            sims.append(sb([i_sent], para_sent_, weights=weights))

        input_paras.append(paras)
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

print('mean of means:', np.mean(means))
print('std of means:', np.std(means))
print()
print('mean of maxs:', np.mean(maxs))
print('std of maxs:', np.std(maxs))

## similarities, higher is better
# with open(output_fn, 'w') as f:
#     for i in range(NUM_SENTS):
#         scores = similarities[i]
#         indices = np.argsort(scores)[::-1]
#         for j in range(OUPUT_NUM):
#             f.write(' '.join(input_paras[i][indices[j]]))
#             f.write('\n')
