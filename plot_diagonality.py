import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
file_path = "/home4/datpt/thuhb/XLSR-Ebranchformer/diagonality/2021_DF/DF__home4_datpt_thuhb_checkpoints_avg_5_best.pth.txt"

layers = []
means = []

with open(file_path, "r") as f:
    for line in f:
        layer_idx = int(re.search(r"layer_(\d+)", line).group(1))
        mean_str = re.search(r"mean=\[(.*?)\]", line).group(1)

        mean_vals = np.array([float(x) for x in mean_str.split(",")])
        mean_avg = mean_vals.mean()  # average over heads

        layers.append(layer_idx)
        means.append(mean_avg)

# sort theo layer
layers = np.array(layers)
means = np.array(means)
idx = np.argsort(layers)
plt.figure()
plt.plot(layers[idx], means[idx], marker='o')

plt.xlabel("Layer")
plt.ylabel("Diagonality")
plt.title("Diagonality vs Layer")
plt.grid()

# 🔥 FIX CHUẨN
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig("DF.png")
plt.show()