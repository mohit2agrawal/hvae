import sys
import os
import numpy as np

filename = "test_plot.txt"
if len(sys.argv) > 1:
    filename = sys.argv[1]
if filename.endswith('txt'):
    out_graph_name = filename.replace(".txt", ".png")
else:
    out_graph_name = filename + ".png"

if len(sys.argv) > 2:
    out_graph_name = sys.argv[2]

with open(filename) as f:
    alpha_arr = np.array([float(x) for x in f.readline().strip().split(" ")])
    beta_arr = np.array([float(x) for x in f.readline().strip().split(" ")])
    tlb = np.array([float(x) for x in f.readline().strip().split(" ")])
    kl = np.array([float(x) for x in f.readline().strip().split(" ")])
    kls = np.array([float(x) for x in f.readline().strip().split(" ")])
    kld = np.array([float(x) for x in f.readline().strip().split(" ")])
    rl = np.array([float(x) for x in f.readline().strip().split(" ")])
    srl = np.array([float(x) for x in f.readline().strip().split(" ")])
    drl = np.array([float(x) for x in f.readline().strip().split(" ")])
    smi = np.array([float(x) for x in f.readline().strip().split(" ")])
    dmi = np.array([float(x) for x in f.readline().strip().split(" ")])

import matplotlib as mpl

# mpl.use("qt5Agg")
mpl.use("Agg")
import matplotlib.pyplot as plt

# fig = plt.figure()
# size = fig.get_size_inches()
# sys.exit()

plt.figure(figsize=(11.69, 8.27))

# ax = plt.subplot(5, 2, 1)
# ax.set_ylim([-.05, 1.05])
# plt.plot(alpha_arr, color="blue", label="alpha")
# plt.plot(beta_arr, color="red", label="beta")
# plt.legend()
# plt.ylabel("Alpha and Beta")

# ax = plt.subplot(5, 2, 2)
# ax.set_ylim([-.05, 1.05])
# plt.plot(alpha_arr, color="blue", label="alpha")
# plt.plot(beta_arr, color="red", label="beta")
# plt.legend()
# plt.ylabel("Alpha and Beta")

ax = plt.subplot(4, 1, 1)
# ax.set_ylim([0, max(tlb_arr)])
ax.set_ylim([-0.05, 30])
plt.plot(30 * .95 * alpha_arr, color="lightskyblue", label="alpha")
plt.plot(30 * .95 * beta_arr, color="lightcoral", label="beta")
plt.plot(tlb, color="blue", label="Total loss")
plt.ylabel("loss")

# ppl = [np.exp(x / 25.0) for x in tlb_arr]
ax = plt.subplot(4, 2, 3)
# ax.set_ylim([-0.05, max(klw_arr)])
ax.set_ylim([-0.05, 10])
plt.plot(10 * .95 * alpha_arr, color="lightskyblue", label="alpha")
plt.plot(10 * .95 * beta_arr, color="lightcoral", label="beta")
plt.plot(kl, color="g", label="KL")
plt.ylabel("weighted KL")

ax = plt.subplot(4, 2, 5)
# ax.set_ylim([-0.05, max(kls)])
#plt.plot(max(kld_zg_arr) * .95 * alpha_arr, color="lightskyblue", label="alpha")
#plt.plot(max(kld_zg_arr) * .95 * beta_arr, color="lightcoral", label="beta")
ax.set_ylim([-0.05, 15])
plt.plot(15 * .95 * alpha_arr, color="lightskyblue", label="alpha")
plt.plot(15 * .95 * beta_arr, color="lightcoral", label="beta")
plt.plot(kls, color="blue", label="kls")
# plt.legend()
plt.ylabel("KL seq")

ax = plt.subplot(4, 2, 7)
# ax.set_ylim([-0.05, max(kld)])
#plt.plot(max(kld_zs_arr) * .95 * alpha_arr, color="lightskyblue", label="alpha")
#plt.plot(max(kld_zs_arr) * .95 * beta_arr, color="lightcoral", label="beta")
ax.set_ylim([-0.05, 15])
plt.plot(15 * .95 * alpha_arr, color="lightskyblue", label="alpha")
plt.plot(15 * .95 * beta_arr, color="lightcoral", label="beta")
plt.plot(kld, color="red", label="kld")
# plt.legend()
plt.xlabel("Iterations")
plt.ylabel("KL doc")

ax = plt.subplot(4, 2, 4)
# ax = plt.subplot(4, 1, 2)
# ax.set_ylim([0, 15])
plt.plot(rl, color="green", label="rl")
plt.ylabel("reconstruction loss")

ax = plt.subplot(4, 2, 6)
# ax = plt.subplot(5, 1, 2)
ax.set_ylim([0, 15])
plt.plot(srl, color="green", label="srl")
plt.ylabel("seq rl")

#lppl = [np.exp(x / 25.0) for x in lrl_arr]
# ax = plt.subplot(5, 1, 3)
# # ax.set_ylim([0, 40])
# plt.plot(lppl, color="green", label="lppl")
# plt.ylabel("label perplexity")

ax = plt.subplot(4, 2, 8)
# ax = plt.subplot(5, 1, 4)
# ax.set_ylim([0, 10])
plt.plot(drl, color="green", label="drl")
plt.ylabel("doc rl")

plt.savefig(out_graph_name)
