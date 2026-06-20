"""
grid_cells.py — hexagonal grid cells from the interference of held trajectories
===============================================================================
THE CLAIM (Gemini's, made testable): if a unit's skew operator HOLDS the directed
trajectory of the body (the WHERE), then the COINCIDENCE of three such held
trajectories, oriented 60 deg apart, is a hexagonal grid. No backprop, no training
— the hexagon is the interference geometry of three velocity-controlled phase
waves. This is the OSCILLATORY-INTERFERENCE model of entorhinal grid cells
(O'Keefe & Burgess; Burgess 2007), built here as a coincidence of skew-held phases.

WHAT IS HONEST AND WHAT IS NOT:
  - the HEXAGON is NOT free. It comes from CHOOSING three preferred directions
    60 deg apart. That choice is the model's central assumption, put in by hand.
  - single-oscillator OI has a real biological problem: phase noise decoheres the
    oscillators, so the grid degrades — Welinder et al. 2008; Giocomo et al. We
    SHOW that fragility (gridness vs phase-noise), we do not hide it.
  - the competing theory is the continuous-attractor network (Fuhs & Touretzky;
    McNaughton et al.). This script demonstrates ONE of two leading accounts.
  - grid cells: Hafting, Fyhn, Molden, Moser & Moser 2005 (discovery; Nobel 2014).

WHAT IS MEASURED: a virtual animal random-walks a 2 m box; three velocity-
controlled phase oscillators integrate its motion (each HOLDS a directed
trajectory); the cell fires on their coincidence. We bin a rate map, take its
spatial autocorrelogram, and compute the standard GRIDNESS score (the 60/120 vs
30/90/150 rotational symmetry). Then we sweep phase noise and watch gridness fall.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

def walk(T=200000, L=2.0, speed=0.012, turn=0.30, seed=1):
    rng = np.random.default_rng(seed)
    pos = np.zeros((T, 2)); p = np.array([L/2, L/2]); th = rng.uniform(0, 2*np.pi)
    for t in range(T):
        th += turn * rng.standard_normal()
        v = speed * np.array([np.cos(th), np.sin(th)])
        p = p + v
        for k in (0, 1):
            if p[k] < 0:   p[k] = -p[k];      th = -th if k == 0 else np.pi - th
            if p[k] > L:   p[k] = 2*L - p[k]; th = -th if k == 0 else np.pi - th
        pos[t] = p
    return pos

def grid_activity(pos, beta=2.6, thr=1.5, phase_noise=0.0, seed=2):
    rng = np.random.default_rng(seed)
    dirs = [np.array([np.cos(a), np.sin(a)]) for a in (0, np.pi/3, 2*np.pi/3)]
    T = len(pos); a = np.zeros(T)
    for d in dirs:
        sp = 2*np.pi*beta*(pos @ d)                 # exact path-integration phase
        if phase_noise > 0:
            sp = sp + np.cumsum(phase_noise*rng.standard_normal(T))   # phase diffusion
        a += np.cos(sp)
    return np.maximum(a - thr, 0.0)                  # coincidence of three held phases

def rate_map(pos, act, L=2.0, G=100):
    edges = np.linspace(0, L, G+1)
    occ, _, _ = np.histogram2d(pos[:,0], pos[:,1], bins=[edges, edges])
    tot, _, _ = np.histogram2d(pos[:,0], pos[:,1], bins=[edges, edges], weights=act)
    return np.divide(tot, occ, out=np.zeros_like(tot), where=occ > 0)

def autocorrelogram(R):
    Rz = R - R.mean()
    F = np.fft.fft2(Rz); ac = np.fft.ifft2(F*np.conj(F)).real
    ac = np.fft.fftshift(ac); ac /= (ac.max() + 1e-12)
    return ac

def gridness(ac, r_in, r_out):
    n = ac.shape[0]; c = n//2
    y, x = np.indices((n, n)); r = np.sqrt((x-c)**2 + (y-c)**2)
    mask = (r >= r_in) & (r <= r_out); base = ac[mask]
    def corr_at(angle):
        return float(np.corrcoef(base, rotate(ac, angle, reshape=False, order=1)[mask])[0, 1])
    return min(corr_at(60), corr_at(120)) - max(corr_at(30), corr_at(90), corr_at(150))

if __name__ == "__main__":
    print("="*74)
    print("THE GRID — hexagonal grid cells from the coincidence of held trajectories")
    print("="*74)
    L = 2.0; G = 100; beta = 2.6
    period_bins = G / (beta * L)
    r_in, r_out = 0.5*period_bins, 1.6*period_bins
    print(f"three velocity-controlled oscillators at 0/60/120 deg; beta={beta}")
    print(f"(grid spacing ~ {1/beta:.2f} m, ~{period_bins:.0f} bins in the autocorrelogram)\n")

    pos = walk(seed=1)
    print(f"  {'phase noise':>12}{'gridness':>11}")
    results = {}
    for pn in [0.0, 0.02, 0.05, 0.10]:
        act = grid_activity(pos, beta=beta, phase_noise=pn)
        R = rate_map(pos, act, L=L, G=G); ac = autocorrelogram(R)
        g = gridness(ac, r_in, r_out); results[pn] = (R, ac, g)
        tag = "clean (path integration exact)" if pn == 0 else "oscillators drifting"
        print(f"  {pn:>12.2f}{g:>11.2f}   {tag}")
    print("\n  -> clean interference gives a strongly hexagonal field (gridness > 0).")
    print("     As the held phases drift (the OI model's real flaw), gridness falls.")
    print("     The hexagon is the geometry of three coincident trajectories — and it")
    print("     is only as stable as the phases the units hold. Relative units.")

    Rc, acc, gc = results[0.0]; Rd, acd, gd = results[0.10]
    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    ax[0,0].imshow(Rc.T, origin="lower", cmap="jet"); ax[0,0].axis("off")
    ax[0,0].set_title(f"rate map — clean (gridness {gc:.2f})", fontsize=11)
    ax[0,1].imshow(acc.T, origin="lower", cmap="jet"); ax[0,1].axis("off")
    ax[0,1].set_title("autocorrelogram — hexagonal", fontsize=11)
    ax[1,0].imshow(Rd.T, origin="lower", cmap="jet"); ax[1,0].axis("off")
    ax[1,0].set_title(f"rate map — phases drifting (gridness {gd:.2f})", fontsize=11)
    ax[1,1].imshow(acd.T, origin="lower", cmap="jet"); ax[1,1].axis("off")
    ax[1,1].set_title("autocorrelogram — degraded", fontsize=11)
    plt.tight_layout(); plt.savefig("grid_cells.png", dpi=110, bbox_inches="tight")
    print("\n  saved grid_cells.png"); print("="*74)
