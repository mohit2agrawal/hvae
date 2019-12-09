import sys
from collections import Counter
# import numpy as np
import matplotlib as mpl

# mpl.use("qt5Agg")
mpl.use("Agg")
import matplotlib.pyplot as plt

fn = sys.argv[1]

c = Counter()

data = [len(x.strip().split(' ')) for x in open(fn).readlines()]
c.update(data)

print(c.items())

total = float(sum(c.values()))

for k, v in c.items():
    c[k] = v / total
    c[k] *= 100
print(c.items())

plt.figure(figsize=(11.69, 8.27))

plt.hist(data)
plt.savefig('a.png')

n, bins, patches = plt.hist(data, 50, density=True, facecolor='g', alpha=0.75)
# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title('Histogram of IQ')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.xlim(40, 160)
# plt.ylim(0, 0.03)
plt.grid(True)
plt.savefig('a.png')
# plt.show()
