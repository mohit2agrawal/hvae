from nltk import jaccard_distance as jd
from nltk.translate.bleu_score import sentence_bleu as sb
import numpy as np
from tqdm import tqdm
import sys
import os

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
with open(input_sents_fn) as f:
    for i in range(NUM_SENTS):
        input_sents.append(f.readline().strip().split(' '))

input_sents = [
    [y if y in word2idx else '<unk>' for y in x] for x in input_sents
]

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
            paras.append(para_sent)
            sims.append(sb([i_sent], para_sent, weights=weights))

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
with open(output_fn, 'w') as f:
    for i in range(NUM_SENTS):
        scores = similarities[i]
        indices = np.argsort(scores)[::-1]
        for j in range(OUPUT_NUM):
            f.write(' '.join(input_paras[i][indices[j]]))
            f.write('\n')
