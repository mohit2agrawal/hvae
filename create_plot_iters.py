import sys
import os
import numpy as np

filename = "test_plot.txt"
if len(sys.argv) > 1:
    filename = sys.argv[1]
out_graph_name = filename.replace(".txt", ".png")
if len(sys.argv) > 2:
    out_graph_name = sys.argv[2]

with open(filename) as f:
    alpha_arr = np.array([float(x) for x in f.readline().strip().split(" ")])
    beta_arr = np.array([float(x) for x in f.readline().strip().split(" ")])
    tlb_arr = np.array([float(x) for x in f.readline().strip().split(" ")])
    klw_arr = np.array([float(x) for x in f.readline().strip().split(" ")])
    kld_zg_arr = np.array([float(x) for x in f.readline().strip().split(" ")])
    kld_zs_arr = np.array([float(x) for x in f.readline().strip().split(" ")])
    rl_arr = np.array([float(x) for x in f.readline().strip().split(" ")])
    lrl_arr = np.array([float(x) for x in f.readline().strip().split(" ")])
    wrl_arr = np.array([float(x) for x in f.readline().strip().split(" ")])

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
ax.set_ylim([0, max(tlb_arr)])
plt.plot(tlb_arr, color="blue", label="Total lower bound")
plt.ylabel("ELBO")

ax = plt.subplot(4, 2, 3)
# ax.set_ylim([-0.05, max(klw_arr)])
ax.set_ylim([-0.05, 10])
plt.plot(10 * .95 * alpha_arr, color="lightskyblue", label="alpha")
plt.plot(10 * .95 * beta_arr, color="lightcoral", label="beta")
plt.plot(klw_arr, color="g", label="KL")
plt.ylabel("weighted KL")

ax = plt.subplot(4, 2, 5)
# ax.set_ylim([-0.05, max(kld_zg_arr)])
ax.set_ylim([-0.05, 5])
plt.plot(5 * .95 * alpha_arr, color="lightskyblue", label="alpha")
plt.plot(5 * .95 * beta_arr, color="lightcoral", label="beta")
plt.plot(kld_zg_arr, color="red", label="kld_zl")
# plt.legend()
plt.ylabel("KL zl")

ax = plt.subplot(4, 2, 7)
# ax.set_ylim([-0.05, max(kld_zs_arr)])
ax.set_ylim([-0.05, 30])
plt.plot(30 * .95 * alpha_arr, color="lightskyblue", label="alpha")
plt.plot(30 * .95 * beta_arr, color="lightcoral", label="beta")
plt.plot(kld_zs_arr, color="blue", label="kld_zc")
# plt.legend()
plt.xlabel("Iterations")
plt.ylabel("KL zc")

ax = plt.subplot(4, 2, 4)
ax.set_ylim([0, 15])
plt.plot(rl_arr, color="green", label="rl")
plt.ylabel("reconstruction loss")

ax = plt.subplot(4, 2, 6)
ax.set_ylim([0, 4])
plt.plot(lrl_arr, color="green", label="lrl")
plt.ylabel("label rl")

ax = plt.subplot(4, 2, 8)
ax.set_ylim([0, 10])
plt.plot(wrl_arr, color="green", label="wrl")
plt.ylabel("word rl")

plt.savefig(out_graph_name)
