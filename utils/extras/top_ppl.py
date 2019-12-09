import json
import sys
import numpy as np

fn = sys.argv[1]

js = json.loads(open(fn).read())
ppls = [x['ppl'] for x in js]

print('mean ppl', np.mean(ppls))
print('std ppl', np.std(ppls))
print()

sp = sorted(ppls)[:2500]
print('mean ppl for top 2500', np.mean(sp))
print('std ppl for top 2500', np.std(sp))

import matplotlib as mpl

# mpl.use("qt5Agg")
mpl.use("Agg")
import matplotlib.pyplot as plt

# n, bins, patches = plt.hist(sp, 50, density=True, facecolor='g', alpha=0.75)
# # plt.xlabel('Smarts')
# # plt.ylabel('Probability')
# # plt.title('Histogram of IQ')
# # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# # plt.xlim(40, 160)
# # plt.ylim(0, 0.03)
# plt.grid(True)
# plt.savefig('ppl.png')

scores = []
for x in js:
    scores.append((x['ppl'], len(x['tokens'])))

sorted_scores = sorted(scores)[:2500]
word_ppl = [float(x[0]) / x[1] for x in sorted_scores]
n, bins, patches = plt.hist(
    word_ppl, 50, density=True, facecolor='g', alpha=0.75
)
plt.grid(True)
plt.savefig('word_ppl.png')
