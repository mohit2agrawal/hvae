'''get the accuracy given two label files
'''

import sys
from collections import defaultdict

if len(sys.argv) < 3:
    print("provide 2 files to be compared")
    sys.exit()
file_a, file_b = sys.argv[1:3]

correct = defaultdict(int)
total = defaultdict(int)
# correct = 0
# total = 0
with open(file_a) as fa, open(file_b) as fb:
    while True:
        a = fa.readline()
        if not a:
            break
        a = a.strip().split(' ')
        b = fb.readline().strip().split(' ')

        for x in zip(a, b):
            total[x[1]] += 1
            if x[0] == x[1]:
                correct[x[1]] += 1

        # correct += sum([x[0]==x[1] for x in zip(a,b)])
        # total += len(a)

# print(correct/total)
for x in total:
    print(x, total[x] / (sum(total.values())), correct[x] / total[x])
print(sum(correct.values()) / sum(total.values()))
