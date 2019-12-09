import sys

pred_fn, orig_fn = sys.argv[1:3]

pred = [int(x) for x in open(pred_fn).readlines()]
orig = [int(x) for x in open(orig_fn).readlines()]

# total = 2000
# assert total == len(pred)

assert len(orig) == len(pred)

neg_correct = 0
pos_correct = 0
neg_count = 0
pos_count = 0

for i in range(len(orig)):
    if orig[i] == 0:
        neg_count += 1
        if orig[i] == pred[i]:
            neg_correct += 1
    if orig[i] == 1:
        pos_count += 1
        if orig[i] == pred[i]:
            pos_correct += 1

neg_acc = float(neg_correct) / neg_count
print('neg_acc:', neg_acc)

pos_acc = float(pos_correct) / pos_count
print('pos_acc:', pos_acc)

print('acc:', (neg_acc + pos_acc) / 2)

print(neg_count)
print(pos_count)
