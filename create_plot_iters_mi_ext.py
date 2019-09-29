import sys
import os

filename = "mi_ext_values.txt"
if len(sys.argv) > 1:
    filename = sys.argv[1]
out_graph_name = filename.replace(".txt", ".png")
if len(sys.argv) > 2:
    out_graph_name = sys.argv[2]

with open(filename) as f:
    smi = [float(x) for x in f.readline().strip().split(" ")]
    gmi = [float(x) for x in f.readline().strip().split(" ")]
    tmi = [float(x) for x in f.readline().strip().split(" ")]
    indicator = [float(x) for x in f.readline().strip().split(" ")]

import matplotlib as mpl

# mpl.use("qt5Agg")
mpl.use("Agg")
import matplotlib.pyplot as plt

# fig = plt.figure()
# size = fig.get_size_inches()
# sys.exit()

plt.figure(figsize=(11.69, 8.27))

# plt.subplot(3, 1, 1)
plt.plot(tmi, color="black", label="total_mi")
# plt.ylabel("Total MI")

# plt.subplot(3, 1, 2)
plt.plot(gmi, color="blue", label="label_mi")
# plt.ylabel("Label MI")

# plt.subplot(3, 1, 3)
plt.plot(smi, color="red", label="sentence_mi")
plt.plot(indicator, color="cyan", label="indicator")
plt.legend()
plt.xlabel("Iterations")
# plt.ylabel("Sentence MI")
plt.ylabel("Mutual Information")

plt.savefig(out_graph_name)
