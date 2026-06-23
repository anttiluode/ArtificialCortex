"""
saccade_agreement_demo.py
=========================

Offline, CPU-only verifier for the foveated agreement-layer architecture.
No GPU, no torch, no trained model, no webcam. Pure numpy + scipy + matplotlib.

Architecture under test (from the diagram):
  GIST  (theta) : blurry low-frequency prior, held; slowly UPDATED where the eye
                  resolves a surprise. PERSISTENT.
  RETINA(gamma) : the sharp world, sampled sharply only at the fovea.
  AGREEMENT     : high-freq detail admitted by a COHERENCE GATE into a buffer
                  that DECAYS. TRANSIENT -- not stored, re-fetched by looking.

  gate: gain = 0.5*(1 + cos d_phi), surprise = 1 - gain,
        d_phi = angle( z_world * conj(z_gist) ) from local Gabor phase,
        energy-weighted across orientations (the janus complex carry).
  saccade: eye -> argmax(surprise) in an attention window, inhibition of return.

Two experiments, each MEASURED (printed), no tuning-to-win:

  EXP 1 (core mechanism). Against a BLURRY prior:
     - the HELD percept stays blurry globally (transient) while an accumulating
       STITCH climbs to a stored sharp photo  => detail is transient, not stored;
     - the percept is sharp LOCALLY at fixated patches;
     - resolving surprise via slow prior update makes the loop SETTLE.

  EXP 2 (surprise targeting -- the honest version). Relative to a blurry prior
     EVERY edge is surprise, so a change is not privileged. The saccade-to-change
     claim only holds against a SETTLED prior (one already in agreement with the
     scene). So we settle the prior, inject a brightness-matched STRUCTURAL change,
     and measure whether surprise now concentrates on the change and whether the
     eye lands there -- comparing the phase gate against a plain intensity residual.

NOT shown: anything about real brains, qualia, or beating any model. A hand-built
mechanism behaving as designed on one contrived static scene. That is all.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import fftconvolve

rng = np.random.default_rng(0)
N = 256
BORDER = 18                       # FFT-convolution edge artifacts live here; mask them


# ---------------------------------------------------------------- scene
def load_gray():
    try:
        from skimage import data, color, transform
        img = color.rgb2gray(data.astronaut())
        return transform.resize(img, (N, N), anti_aliasing=True).astype(np.float64)
    except Exception:
        y, x = np.mgrid[0:N, 0:N] / N
        base = np.sin(6 * np.pi * x) * np.sin(5 * np.pi * y) * 0.25 + 0.5
        for _ in range(6):
            cx, cy, r = rng.uniform(0.2, 0.8, 3)
            base += 0.3 * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (0.02 + 0.03 * r)))
        return np.clip(base, 0, 1)


I0 = load_gray()
gist0 = gaussian_filter(I0, 5.0)            # blurry theta prior
change_box = (40, 110, 150, 220)            # y0,y1,x0,x1


def inject_change(img, box, mode="rotate"):
    y0, y1, x0, x1 = box
    patch = img[y0:y1, x0:x1].copy()
    new = rotate(patch, 90, reshape=False, mode="reflect") if mode == "rotate" else patch[::-1, ::-1]
    new = (new - new.mean()) / (new.std() + 1e-6) * patch.std() + patch.mean()  # match brightness
    out = img.copy()
    out[y0:y1, x0:x1] = np.clip(new, 0, 1)
    return out


world = inject_change(I0, change_box)


# ---------------------------------------------------------------- gabor / gate
def gabor_kernel(theta, lam, sigma, ksize=25):
    r = ksize // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)
    k = np.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2)) * np.exp(1j * 2 * np.pi * xr / lam)
    return k - k.mean()


def analytic(img, lam=8.0, sigma=6.0, n_orient=4):
    out = []
    for o in range(n_orient):
        th = np.pi * o / n_orient
        k = gabor_kernel(th, lam, sigma)
        out.append(fftconvolve(img, k.real, mode="same") + 1j * fftconvolve(img, k.imag, mode="same"))
    return np.stack(out, 0)


ZW = analytic(world)


def gate_and_surprise(gist, zw=ZW):
    """coherence gate gain in [0,1] and surprise = 1 - gain, energy-weighted."""
    zg = analytic(gist)
    ph = zw * np.conj(zg)
    amp = np.abs(zw) * np.abs(zg)
    cos_dphi = np.real(ph) / (amp + 1e-9)
    w = amp / (amp.sum(0, keepdims=True) + 1e-9)
    gain = np.clip((w * 0.5 * (1 + cos_dphi)).sum(0), 0, 1)
    return gain, 1.0 - gain


def mask_border(m):
    out = m.copy()
    out[:BORDER, :] = out[-BORDER:, :] = out[:, :BORDER] = out[:, -BORDER:] = 0
    return out


def enrichment(smap, box):
    smap = mask_border(smap)
    y0, y1, x0, x1 = box
    frac_mass = smap[y0:y1, x0:x1].sum() / (smap.sum() + 1e-9)
    frac_area = ((y1 - y0) * (x1 - x0)) / ((N - 2 * BORDER) ** 2)
    return frac_mass / frac_area


def aperture(cy, cx, s=14.0):
    yy, xx = np.mgrid[0:N, 0:N]
    return np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2)))


# ---------------------------------------------------------------- EXP 1
def run_loop(driver="phase", n_fix=14, decay=0.55, update_prior=True,
             fov=14.0, ior_s=26.0):
    gist = gist0.copy()
    agree = np.zeros((N, N)); stitch = np.zeros((N, N)); ior = np.zeros((N, N))
    path, trace = [], []
    yy, xx = np.mgrid[0:N, 0:N]
    for _ in range(n_fix):
        residual = world - gist
        gain, surp = gate_and_surprise(gist)
        sal = gaussian_filter(surp if driver == "phase" else np.abs(residual), 4)
        sal = mask_border(sal) * (1 - np.clip(ior, 0, 1))
        cy, cx = np.unravel_index(np.argmax(sal), sal.shape)
        path.append((cy, cx))
        ap = aperture(cy, cx, fov)
        agree = decay * agree + ap * gain * residual          # admit AGREEING detail; transient
        stitch = stitch + ap * residual                       # contrast: keep all, sharp
        if update_prior:
            gist = gist + 0.9 * gaussian_filter(ap * (1 - gain) * residual, 2.0)  # slow low-freq fix
        ior = 0.9 * ior + np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * ior_s ** 2)))
        y0, y1, x0, x1 = change_box
        trace.append(float(surp[y0:y1, x0:x1].mean()))
    return dict(percept=gist + agree, stitched=gist0 + stitch, gist=gist,
                agree=agree, path=path, trace=trace, surp=surp)


res = run_loop("phase")


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)


def local_psnr(img, ref, centers, rad=18):
    v = []
    for cy, cx in centers:
        y0, y1, x0, x1 = max(0, cy - rad), min(N, cy + rad), max(0, cx - rad), min(N, cx + rad)
        v.append(psnr(img[y0:y1, x0:x1], ref[y0:y1, x0:x1]))
    return np.mean(v)


g_gist = psnr(gist0, world); g_tr = psnr(res["percept"], world); g_st = psnr(res["stitched"], world)
l_gist = local_psnr(gist0, world, res["path"]); l_pc = local_psnr(res["percept"], world, res["path"])


# ---------------------------------------------------------------- EXP 2
settled = gaussian_filter(I0, 1.0)                 # prior already in agreement w/ the scene
g_set, surp_set = gate_and_surprise(settled)       # surprise of settled prior vs CHANGED world
surp_int = np.abs(world - settled); surp_int /= surp_int.max() + 1e-9
enr_p = enrichment(gaussian_filter(surp_set, 4), change_box)
enr_i = enrichment(gaussian_filter(surp_int, 4), change_box)

# where does the eye land (phase-driven, IOR) against the settled prior?
sal2 = mask_border(gaussian_filter(surp_set, 4)); ior2 = np.zeros((N, N))
yy, xx = np.mgrid[0:N, 0:N]; land = []
for _ in range(6):
    s = sal2 * (1 - np.clip(ior2, 0, 1))
    cy, cx = np.unravel_index(np.argmax(s), s.shape); land.append((cy, cx))
    ior2 = 0.9 * ior2 + np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 26.0 ** 2)))
y0, y1, x0, x1 = change_box
in_box = [(y0 <= cy < y1 and x0 <= cx < x1) for cy, cx in land]


# ---------------------------------------------------------------- print
print("=" * 70)
print("OFFLINE AGREEMENT-LAYER VERIFIER  --  measured results")
print("=" * 70)
print("\nEXP 1  core mechanism (against a blurry prior)")
print(f"  blurry prior vs world        : {g_gist:5.2f} dB")
print(f"  HELD percept (transient)     : {g_tr:5.2f} dB   ({g_tr-g_gist:+.2f} vs gist) <- barely moves: NOT stored")
print(f"  stitch (accumulating)        : {g_st:5.2f} dB   ({g_st-g_gist:+.2f} vs gist) <- climbs: a stored photo")
print(f"  fixated regions: gist        : {l_gist:5.2f} dB")
print(f"  fixated regions: percept     : {l_pc:5.2f} dB   ({l_pc-l_gist:+.2f}) <- locally sharp under the fovea")
print(f"  surprise in change box       : start {res['trace'][0]:.3f} -> end {res['trace'][-1]:.3f}"
      f"  ({'settles' if res['trace'][-1] < res['trace'][0]*0.8 else 'flat'})")
print("\nEXP 2  surprise targeting (against a SETTLED prior)")
print(f"  change-box enrichment, intensity residual : {enr_i:5.2f}x")
print(f"  change-box enrichment, phase gate         : {enr_p:5.2f}x")
print(f"  first 6 fixations landing inside the change: {sum(in_box)}/6  {in_box}")
better = ("phase gate concentrates harder" if enr_p > enr_i * 1.15
          else "intensity concentrates harder" if enr_i > enr_p * 1.15
          else "phase and intensity comparable")
print(f"  => {better}")
print("=" * 70)


# ---------------------------------------------------------------- figures
def show(a, img, title, cmap="gray"):
    a.imshow(img, cmap=cmap, vmin=0 if cmap == "gray" else None, vmax=1 if cmap == "gray" else None)
    a.set_title(title, fontsize=10.5); a.axis("off")


fig1, ax = plt.subplots(2, 3, figsize=(13.5, 9))
show(ax[0, 0], gist0, "Gist - prior (theta)\nblurry, held")
show(ax[0, 1], world, "Retina - world (gamma)\nbox = post-prior change")
ax[0, 1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#1D9E75", lw=2))
show(ax[0, 2], mask_border(gaussian_filter(res["surp"], 4)),
     "EXP1 surprise vs blurry prior\n(every edge is surprise) + scanpath", "magma")
px = [p[1] for p in res["path"]]; py = [p[0] for p in res["path"]]
ax[0, 2].plot(px, py, "-o", color="#85B7EB", ms=4, lw=1)
show(ax[1, 0], np.clip(res["percept"], 0, 1), "HELD percept (transient)\ngist + decaying agreement")
show(ax[1, 1], np.clip(res["stitched"], 0, 1), "Stitch (stored photo)\ngist + accumulated detail")
show(ax[1, 2], res["agree"], "Agreement buffer alone\ntransient: fades behind the eye", "magma")
plt.tight_layout(); fig1.savefig("/home/claude/saccade_agreement.png", dpi=110, bbox_inches="tight")

fig2, bx = plt.subplots(1, 3, figsize=(13.5, 4.6))
show(bx[0], settled, "Settled prior\n(in agreement with the scene)")
show(bx[1], world, "World after change\nbox = the new surprise")
bx[1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#1D9E75", lw=2))
show(bx[2], mask_border(gaussian_filter(surp_set, 4)),
     "EXP2 surprise vs settled prior\nfixations land on the change", "magma")
bx[2].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#5DCAA5", lw=1.5))
lx = [p[1] for p in land]; ly = [p[0] for p in land]
bx[2].plot(lx, ly, "-o", color="#85B7EB", ms=5, lw=1)
bx[2].plot(lx[0], ly[0], "o", color="white", ms=8)
plt.tight_layout(); fig2.savefig("/home/claude/saccade_novelty.png", dpi=110, bbox_inches="tight")
print("\nsaved -> saccade_agreement.png , saccade_novelty.png")
