import json
import sys
import numpy as np


def read_file(fn):
    return np.array(open(fn).readlines())


def write_file(fn, sentences):
    open(fn, 'w').writelines(sentences)


input_sents_fn, scores_fn = sys.argv[1:3]

output_fn = input_sents_fn.replace('.txt', '.selected.txt')

js = json.loads(open(scores_fn).read())
ppls = [x['ppl'] for x in js]

indices = np.argsort(ppls)  ## lower ppl is better
output_sents = read_file(input_sents_fn)[:2500]
write_file(output_fn, output_sents)

print('mean ppl', np.mean(ppls))
print('std ppl', np.std(ppls))
print()

sp = sorted(ppls)[:2500]
print('mean ppl for top 2500', np.mean(sp))
print('std ppl for top 2500', np.std(sp))
