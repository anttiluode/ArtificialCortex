"""
Why is the live percept ~= the gist, and the surprise map a field of stripes?
Diagnosis: the coherence gate only admits detail when the prior is REGISTERED to
the frame. A model gist (a canonical-ish face, not aligned to your actual frame)
disagrees with the frame nearly everywhere -> gate closes -> agreement layer is
starved -> you see the raw VAE prior. This probes that with measurement.
Same verified gate math as saccade_agreement_demo.py.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, shift as ndshift
from scipy.signal import fftconvolve
from skimage import data, color, transform

N = 256
frame = transform.resize(color.rgb2gray(data.astronaut()), (N, N), anti_aliasing=True)
other = transform.resize(data.camera(), (N, N), anti_aliasing=True)   # a DIFFERENT face

def gk(th, lam=8., sig=6., ks=25):
    r = ks // 2; yy, xx = np.mgrid[-r:r+1, -r:r+1]
    xr = xx*np.cos(th) + yy*np.sin(th); yr = -xx*np.sin(th) + yy*np.cos(th)
    k = np.exp(-(xr**2 + yr**2)/(2*sig**2)) * np.exp(1j*2*np.pi*xr/lam)
    return k - k.mean()
K = [gk(np.pi*o/4) for o in range(4)]
def analytic(im): return np.stack([fftconvolve(im, k.real, "same") + 1j*fftconvolve(im, k.imag, "same") for k in K], 0)
def gate(gist, zw):
    zg = analytic(gist); ph = zw*np.conj(zg); amp = np.abs(zw)*np.abs(zg)
    cos = np.real(ph)/(amp+1e-9); w = amp/(amp.sum(0, keepdims=True)+1e-9)
    g = np.clip((w*0.5*(1+cos)).sum(0), 0, 1); return g, 1-g
def ap(cy, cx, s=16.):
    yy, xx = np.mgrid[0:N, 0:N]; return np.exp(-(((yy-cy)**2 + (xx-cx)**2)/(2*s**2)))
def loop(prior, nfix=12, decay=0.5):
    zw = analytic(frame); g, surp = gate(prior, zw); res = frame - prior
    agree = np.zeros((N, N)); ior = np.zeros((N, N)); yy, xx = np.mgrid[0:N, 0:N]
    for _ in range(nfix):
        sal = gaussian_filter(surp, 4)*(1-np.clip(ior, 0, 1))
        sal[:18] = sal[-18:] = 0; sal[:, :18] = sal[:, -18:] = 0
        cy, cx = np.unravel_index(np.argmax(sal), sal.shape)
        agree = decay*agree + ap(cy, cx)*g*res
        ior = 0.9*ior + np.exp(-(((yy-cy)**2 + (xx-cx)**2)/(2*26.**2)))
    return np.clip(prior+agree, 0, 1), g, surp
def psnr(a, b): m = np.mean((a-b)**2); return 99. if m < 1e-12 else 10*np.log10(1./m)

priors = {
    "aligned blur (the working config)": gaussian_filter(frame, 5),
    "misaligned blur (shifted 14,17px)": ndshift(gaussian_filter(frame, 5), (14, 17), mode="nearest"),
    "other face (proxy for model gist)": gaussian_filter(other, 5),
}
print("="*78)
print("PRIOR-ALIGNMENT PROBE  --  does the agreement loop add anything?")
print("="*78)
rows = []
for name, pr in priors.items():
    perc, g, surp = loop(pr)
    rows.append((name, pr, surp, perc))
    print(f"  {name:36s} | mean gate gain {g.mean():.3f} | "
          f"prior {psnr(pr,frame):5.2f}dB -> percept {psnr(perc,frame):5.2f}dB | "
          f"loop adds {psnr(perc,frame)-psnr(pr,frame):+.2f}dB")
print("="*78)

fig, ax = plt.subplots(3, 4, figsize=(13, 9.5))
col = ["retina (frame)", "gist (prior)", "surprise -> saccade", "percept = gist + agreement"]
for r, (name, pr, surp, perc) in enumerate(rows):
    ims = [frame, pr, surp/(surp.max()+1e-9), perc]
    cmaps = ["gray", "gray", "magma", "gray"]
    for c in range(4):
        a = ax[r, c]
        a.imshow(ims[c], cmap=cmaps[c], vmin=0 if cmaps[c]=="gray" else None, vmax=1 if cmaps[c]=="gray" else None)
        a.axis("off")
        if r == 0: a.set_title(col[c], fontsize=10.5)
    ax[r, 1].text(4, 22, name, color="#5DCAA5", fontsize=9)
plt.tight_layout(); fig.savefig("/home/claude/prior_alignment.png", dpi=108, bbox_inches="tight")
print("saved -> prior_alignment.png")
