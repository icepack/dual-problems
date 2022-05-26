import numpy as np
import matplotlib.pyplot as plt

ts = np.linspace(-1.0, 1.0, 201)

n = 3
zs = n / (n + 1) * abs(ts) ** (1 / n + 1)
dzs = abs(ts) ** (1 / n) * np.sign(ts)
ddzs = 1 / n * abs(ts) ** (1 / n - 1)

fontsize = 24

fig, ax = plt.subplots(constrained_layout=True)
ax.set_title("Primal form", fontsize=fontsize)
ax.set_xlabel("$\dot\\varepsilon$", fontsize=fontsize)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_visible(False)
ax.set_xlim((-1.0, 1.0))
ax.set_ylim((-0.05, 1.0))
ax.plot(ts, zs, label="$P(\dot\\varepsilon)$")
ax.plot(ts, ddzs, label="$d^2P(\dot\\varepsilon)$")
ax.legend(fontsize=fontsize, loc="upper right")
fig.savefig("primal.pdf", bbox_inches="tight")


ws = 1 / (n + 1) * abs(ts) ** (n + 1)
dws = abs(ts) ** n
ddws = n * abs(ts) ** (n - 1)

fig, ax = plt.subplots(constrained_layout=True)
ax.set_title("Dual form", fontsize=fontsize)
ax.set_xlabel("$\\tau$", fontsize=fontsize)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_visible(False)
ax.set_xlim((-1.0, 1.0))
ax.set_ylim((-0.05, 1.0))
ax.plot(ts, ws, label="$P(\\tau)$")
ax.plot(ts, ddws, label="$d^2P(\\tau)$")
ax.legend(fontsize=fontsize, loc="upper right")
fig.savefig("dual.pdf", bbox_inches="tight")
