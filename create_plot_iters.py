import sys
import os

filename = "test_plot.txt"
if len(sys.argv) > 1:
    filename = sys.argv[1]
out_graph_name = filename.replace(".txt", ".png")
if len(sys.argv) > 2:
    out_graph_name = sys.argv[2]

with open(filename) as f:
    alpha_arr = [float(x) for x in f.readline().strip().split(" ")]
    beta_arr = [float(x) for x in f.readline().strip().split(" ")]
    tlb_arr = [float(x) for x in f.readline().strip().split(" ")]
    klw_arr = [float(x) for x in f.readline().strip().split(" ")]
    kld_zg_arr = [float(x) for x in f.readline().strip().split(" ")]
    kld_zs_arr = [float(x) for x in f.readline().strip().split(" ")]

import matplotlib as mpl

# mpl.use("qt5Agg")
mpl.use("Agg")
import matplotlib.pyplot as plt

# fig = plt.figure()
# size = fig.get_size_inches()
# sys.exit()

plt.figure(figsize=(11.69, 8.27))

plt.subplot(5, 1, 1)
plt.plot(alpha_arr, color="red", label="alpha")
plt.plot(beta_arr, color="blue", label="beta")
plt.legend()
# plt.xlabel('Epochs')
plt.ylabel("Alpha and Beta")
# plt.savefig('./graph_alpha_epochs_150.png')

plt.subplot(5, 1, 2)
plt.plot(tlb_arr, color="blue", label="Total lower bound")
# plt.xlabel('Epochs')
plt.ylabel("ELBO")
# plt.savefig('./graph_elbo_epochs_150.png')

plt.subplot(5, 1, 3)
plt.plot(klw_arr, color="blue", label="KL")
# plt.legend()
# plt.title("KL Term Value vs Epochs")
# plt.xlabel('Epochs')
plt.ylabel("KL")
# plt.savefig('./graph_klw_epochs_150.png')

plt.subplot(5, 1, 4)
plt.plot(kld_zg_arr, color="green", label="kld_zl")
# plt.xlabel('Epochs')
plt.ylabel("KL zl")
# plt.savefig('./graph_kld_zg_epochs_150.png')

plt.subplot(5, 1, 5)
plt.plot(kld_zs_arr, color="red", label="kld_zc")
plt.xlabel("Iterations")
plt.ylabel("KL zc")
# plt.savefig('./graph_kld_zs_epochs_150.png')

plt.savefig(out_graph_name)
