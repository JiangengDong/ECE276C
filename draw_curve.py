from matplotlib import pyplot as plt
import numpy as np
import os

def smooth(v):
    v = v.copy()
    n = v.shape[0]
    for i in range(1, n):
        v[i] = 0.6*v[i-1] + 0.4*v[i]
    return v

folder = "./data/DDPG-1-shaping-rand"
csv_path = os.path.join(folder, "run-log-tag-return.csv")
img_path = os.path.join(folder, "curve.pdf")

data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
t = data[:, 1]
r = data[:, 2]
r = smooth(r)

plt.plot(t, r)
plt.xlim(t[0], t[-1])
plt.ylim(2, 14)
plt.tight_layout()
plt.savefig(img_path, bbox_inches="tight")
plt.show()