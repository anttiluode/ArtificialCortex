"""
Two questions about the temporal upgrade, answered by measurement:

 Q1. Does a learned linear Koopman operator (Gemini's "z_{t+1} = K z_t") predict
     the next frame better than persistence (copy the current frame)?
 Q2. If not, what is the real payoff of going temporal?

Pure numpy DMD on a moving face video (translation+rotation+lighting+noise).
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import shift as ndshift, gaussian_filter
from skimage import data, color, transform

rng = np.random.default_rng(0)
N, T = 64, 60
base = transform.resize(color.rgb2gray(data.astronaut()), (N, N), anti_aliasing=True)
frames = []
for t in range(T):
    f = transform.rotate(base, 7 * np.sin(2 * np.pi * t / 40), mode="edge")
    f = ndshift(f, (4 * np.cos(2 * np.pi * t / 35), 6 * np.sin(2 * np.pi * t / 30)), mode="nearest")
    f = f * (0.9 + 0.1 * np.sin(2 * np.pi * t / 50))
    frames.append(np.clip(f + 0.01 * rng.standard_normal((N, N)), 0, 1))
X = np.stack([f.ravel() for f in frames])
Xm = X.mean(0); U, S, Vt = np.linalg.svd(X - Xm, full_matrices=False)
r = 24; Bmat = Vt[:r]; Z = (X - Xm) @ Bmat.T
decode = lambda z: np.clip(z @ Bmat + Xm, 0, 1)
psnr = lambda a, b: 99. if np.mean((a - b) ** 2) < 1e-12 else 10 * np.log10(1. / np.mean((a - b) ** 2))
tr = 40
A, Bn = Z[:tr][:-1], Z[:tr][1:]
Kfree = (np.linalg.pinv(A) @ Bn).T
I_ = np.eye(r)
test = range(tr, T - 1)

def mean_psnr(predfn): return float(np.mean([psnr(predfn(t), X[t + 1]) for t in test]))

free = mean_psnr(lambda t: decode(Kfree @ Z[t]))
pixper = mean_psnr(lambda t: X[t])
latper = mean_psnr(lambda t: decode(Z[t]))
static = mean_psnr(lambda t: Xm)
# best identity-anchored operator: K = I + a*(Kfree-I), a in [0,1]
best_a, best_p = 0.0, -1
for a in np.linspace(0, 1, 21):
    p = mean_psnr(lambda t, Ka=I_ + a * (Kfree - I_): decode(Ka @ Z[t]))
    if p > best_p: best_a, best_p = a, p

print("=" * 72)
print("Q1  next-frame prediction (held-out), dB")
print("=" * 72)
print(f"  free linear Koopman  decode(Kfree z_t) : {free:5.2f}")
print(f"  pixel persistence    copy(f_t)         : {pixper:5.2f}   <- baseline to beat")
print(f"  latent persistence   decode(z_t)       : {latper:5.2f}")
print(f"  static mean prior                      : {static:5.2f}")
print(f"  best identity-anchored K=I+a(Kf-I)     : {best_p:5.2f}  at a={best_a:.2f}")
print(f"  => a={best_a:.2f} means the operator helps "
      f"{'not at all; persistence wins' if best_a < 0.05 else 'a little'}")
print(f"  => free Koopman vs persistence         : {free-pixper:+.2f} dB"
      f"  ({'WORSE' if free < pixper else 'better'})")

print("\nQ2  the real payoff: persistence-as-prediction makes SURPRISE = MOTION.")
# temporal surprise (what persistence misses) vs the static gate's spatial surprise
t = tr + 3
temp = np.abs(X[t + 1] - X[t]).reshape(N, N)
blur = gaussian_filter(X[t + 1].reshape(N, N), 3)
spat = np.abs(X[t + 1].reshape(N, N) - blur)               # static "every edge" surprise
topfrac = lambda m, p=0.10: np.sort(m.ravel())[::-1][:int(m.size * p)].sum() / (m.sum() + 1e-9)
print(f"  temporal surprise |f_t+1 - f_t|  : {topfrac(temp)*100:4.1f}% of mass in top 10% pixels (sparse=motion)")
print(f"  static surprise  |f - blur(f)|   : {topfrac(spat)*100:4.1f}% of mass in top 10% pixels (diffuse=every edge)")
print("=" * 72)

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
xs = list(test)
ax[0].plot(xs, [psnr(decode(Kfree @ Z[t]), X[t + 1]) for t in test], "-o", ms=3, label="free Koopman K")
ax[0].plot(xs, [psnr(X[t], X[t + 1]) for t in test], "-s", ms=3, label="persistence")
ax[0].plot(xs, [psnr(Xm, X[t + 1]) for t in test], ":", label="static mean")
ax[0].set_title("Q1: free linear K loses to persistence"); ax[0].set_ylabel("PSNR dB")
ax[0].set_xlabel("predict frame t+1"); ax[0].legend(fontsize=8)
ax[1].imshow(temp, cmap="magma"); ax[1].axis("off")
ax[1].set_title("temporal surprise = motion\n(sparse, localized)")
ax[2].imshow(spat, cmap="magma"); ax[2].axis("off")
ax[2].set_title("static surprise = every edge\n(diffuse — the old failure)")
plt.tight_layout(); fig.savefig("/home/claude/koopman_predict.png", dpi=110, bbox_inches="tight")
print("saved -> koopman_predict.png")
