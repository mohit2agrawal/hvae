'''get 5 random sets of lines from the input
'''
import sys
import numpy as np

# np.seed(0)

sent_file, label_file = sys.argv[1:3]

with open(sent_file) as f:
    sents = f.readlines()
with open(label_file) as f:
    labels = f.readlines()

for i in range(5):
    indices = np.random.choice(len(sents), 2500, replace=False)

    out_filename = 'random_sents_' + str(i) + '.txt'
    with open(out_filename, 'w') as f:
        for idx in indices:
            f.write(sents[idx])

    out_filename = 'random_labels_' + str(i) + '.txt'
    with open(out_filename, 'w') as f:
        for idx in indices:
            f.write(labels[idx])
