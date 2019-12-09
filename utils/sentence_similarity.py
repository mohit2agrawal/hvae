from nltk import jaccard_distance as jd
from nltk.translate.bleu_score import sentence_bleu as sb
import numpy as np
from tqdm import tqdm
import sys
import os

bleu = False
bleu4 = False
if len(sys.argv) > 1:
    what = sys.argv[1]
    if what.lower() in ['blue', 'bleu', 'b', 'b2', 'blue2', 'bleu2']:
        bleu = True
        weights = (0.5, 0.5, 0, 0)
    if what.lower() in ['blue4', 'bleu4', 'b4']:
        bleu = True
        bleu4 = True
        weights = (0.25, 0.25, 0.25, 0.25)

input_sents_fn = 'DATA/ptb_pos_short/data_mod.txt'
input_para_fn = 'generated_sentences.txt'
input_para_labels_fn = 'generated_labels.txt'

if not os.path.exists('generated'):
    os.mkdir('generated')
output_fn = 'generated/sentences_jaccard.txt'
output_labels_fn = 'generated/labels_jaccard.txt'
if bleu:
    output_fn = 'generated/sentences_bleu.txt'
    output_labels_fn = 'generated/labels_bleu.txt'
if bleu4:
    output_fn = 'generated/sentences_bleu4.txt'
    output_labels_fn = 'generated/labels_bleu4.txt'

NUM_SENTS = 500
NUM_PARA = 1000
OUPUT_NUM = 100  ## top N paras per sentence

input_sents = []
with open(input_sents_fn) as f:
    for i in range(NUM_SENTS):
        input_sents.append(f.readline().strip().split(' '))

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

        for j in range(NUM_PARA):
            para_sent = f.readline().strip().split(' ')
            paras.append(para_sent)
            labels.append(lf.readline())
            if bleu:
                sims.append(sb([i_sent], para_sent, weights=weights))
            else:
                sims.append(1 - jd(i_sent_set, set(para_sent)))

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
with open(output_fn, 'w') as f, open(output_labels_fn, 'w') as lf:
    for i in range(NUM_SENTS):
        scores = similarities[i]
        indices = np.argsort(scores)[::-1]
        for j in range(OUPUT_NUM):
            f.write(' '.join(input_paras[i][indices[j]]))
            f.write('\n')
            lf.write(input_labels[i][indices[j]])
