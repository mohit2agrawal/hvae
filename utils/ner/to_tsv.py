''' convert sentences and labels from two input files to a single tsv
'''
import sys
from tqdm import tqdm

if len(sys.argv) < 3:
    print("provide 2 input and 1 output file name")
    sys.exit()
sents_file, labels_file, output_file = sys.argv[1:4]

# rep = {'0':'O', '1':'LOCATION', '2':'PERSON', '3':'ORGANIZATION'}
num_lines = sum([1 for line in open(sents_file)])
with open(sents_file) as sf, open(labels_file
                                  ) as lf, open(output_file, 'w') as of:
    for idx in tqdm(range(num_lines)):
        sentence = sf.readline()
        labels_sent = lf.readline()

        words = sentence.strip().split(' ')
        labels = labels_sent.strip().split(' ')
        # labels = [rep[label] for label in labels_sent.strip().split(' ')]

        for i in range(len(words)):
            of.write('\t'.join([words[i], labels[i]]))
            of.write('\n')
        of.write('\n')
