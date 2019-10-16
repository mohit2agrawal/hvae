'''add half of the lines (randomly) from input to output
'''

import sys
import numpy as np

# np.seed(0)

input_sents, input_labels, output_sents, output_labels = sys.argv[1:5]

with open(input_sents) as f:
    sents = f.readlines()
with open(input_labels) as f:
    labels = f.readlines()

indices = np.random.choice(len(sents), len(sents) // 2, replace=False)

with open(output_sents, 'a') as sf, open(output_labels, 'a') as lf:
    for idx in indices:
        sf.write(sents[idx])
        lf.write(labels[idx])
