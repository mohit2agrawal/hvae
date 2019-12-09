from nltk import pos_tag

from collections import Counter, defaultdict
import sys
from tqdm import tqdm

if len(sys.argv) < 3:
    print("provide input and output file name")
    sys.exit()
input_file, output_file = sys.argv[1:3]

# words = defaultdict(list)
num_lines = sum([1 for line in open(input_file)])
c = Counter()
# i = 0
with open(input_file) as f, open(output_file, 'w') as of:
    for line in tqdm(f, total=num_lines):

        # i += 1
        # if i == 100:
        #     break

        # tokenized_text = word_tokenize(line)
        tokenized_text = line.strip().split(' ')
        classified_text = pos_tag(tokenized_text, tagset='universal')
        ## classified_text is a list of tuples [(word, tag), ...]

        # for word, label in classified_text:
        #     words[label].append(word)

        labels = [x[1] for x in classified_text]
        of.write(' '.join(labels))
        of.write('\n')
        c.update(labels)

print(c)
