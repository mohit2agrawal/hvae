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

input_sents_fn, input_para_fn = sys.argv[1:3]

output_fn = input_para_fn.replace('.txt', '.selected.txt')
# output_labels_fn = input_para_labels_fn.replace('.txt', '.selected.txt')

NUM_SENTS = 250 * 5
NUM_PARA = 100
OUPUT_NUM = 2  ## top N paras per sentence

input_sents = []
input_sents_orig = []
with open(input_sents_fn) as f:
    for i in range(NUM_SENTS):
        s = f.readline().strip().split(' ')
        input_sents_orig.append(s)
        input_sents.append(
            [lemmatizer.lemmatize(x) for x in s if x not in stopWords]
        )

input_paras = []
# input_labels = []
similarities = []
means = []
maxs = []
with open(input_para_fn) as f:
    for i in tqdm(range(NUM_SENTS)):
        i_sent = input_sents[i]
        i_sent_set = set(i_sent)

        paras = []
        labels = []
        sims = []

        ## original sentences are also written. escape them first.
        # f.readline()
        # lf.readline()
        for j in range(NUM_PARA):
            para_sent = f.readline().strip().split(' ')
            para_sent_ = [
                lemmatizer.lemmatize(x) for x in para_sent if x not in stopWords
            ]
            paras.append(para_sent)
            # labels.append(lf.readline())
            sims.append(sb([i_sent], para_sent_, weights=weights))

        input_paras.append(paras)
        # input_labels.append(labels)
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
            # blank = lf.readline().strip()
            # assert blank == ''

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
#             # lf.write(input_labels[i][indices[j]])
#         f.write('\n')

with open(output_fn, 'w') as f:
    for i in range(NUM_SENTS):
        scores = similarities[i]
        indices = np.argsort(scores)[::-1]
        paras = np.array(input_paras[i])[indices]

        orig_sent = ''.join(input_sents_orig[i])
        paras = [s for s in paras if orig_sent != ''.join(s)]

        if i == NUM_SENTS - 5 - 1:
            print(' '.join(input_sents_orig[i]))
            for x in paras:
                print(x)

        if not paras:
            print('got nothing')
            print(i)
            print(' '.join(input_sents[i]))
            f.write('\n')
            continue

        uniq_paras = [paras[0]]
        last_idx = 0
        for i, sent in enumerate(paras):
            if ''.join(sent) != ''.join(uniq_paras[last_idx]):
                uniq_paras.append(sent)
                last_idx += 1

        for j in range(OUPUT_NUM):
            if j < len(uniq_paras):
                f.write(' '.join(uniq_paras[j]))
                f.write('\n')
                # lf.write(input_labels[i][indices[j]])
        f.write('\n')
