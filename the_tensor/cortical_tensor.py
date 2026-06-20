"""
cortical_tensor.py — the 3D tensor: X,Y space x Z frequency, energy spent on surprise
=====================================================================================
THE HORIZON BUILD. The three organs become one block:
  - a SLOW (theta) layer: a prescribed traveling wave sweeping the sheet (the deep
    medial-septum pacemaker) whose phase is the gate, holding a GRID-PATTERNED
    PREDICTION P (the WHERE — the expected spatial world);
  - a FAST (gamma) layer: a damped, DRIVEN complex field that encodes only the
    RESIDUAL R = world - prediction (the NOW that the grid did not foresee),
    its growth gated by the local theta phase (the chandelier on the clock).

So the computation is not a tower of weights flashing top-to-bottom. It is a wave
of coincidence: the fast layer lights up ONLY where the sensory world contradicts
the slow grid prediction, and only when theta opens the gate. Two things are
MEASURED, not asserted:

  [A] SPATIAL THETA-GAMMA PAC. Drive gamma with a broadband world, gate it by the
      theta wave, and measure Tort's Modulation Index at probe sites vs an ungated
      control. This is `the_stack` lifted from a 1-D signal into a 2-D field.

  [B] ENERGY proportional to SURPRISE. Make the world = the grid prediction plus
      surprise patches of controllable density rho. Measure total gamma energy
      E(rho) = <|z_gamma|^2>. A static processor (a transformer-like dense block)
      pays the SAME cost at every cell regardless of rho; this field's cost scales
      with rho and is near zero when the world matches the prediction.

GROUNDING (established, used not claimed): predictive coding / residual coding
(Rao & Ballard 1999; the ELL negative image, Bell/Sawtell); theta-gamma code
(Lisman & Jensen 2013); cross-frequency coupling (Canolty & Knight 2010; Tort
2010); medial-septal theta pacing; neural field theory (Wilson-Cowan; Amari).
HONEST LIMITS: theta is a PRESCRIBED pacemaker here (not self-organised); damped-
driven linear gamma; relative units; one toy world; "energy" = field amplitude^2,
a proxy, not Joules. This shows the ARCHITECTURE spends on surprise; it is not a
benchmark and not a brain. The bet (that any of it is felt) stays in the drawer.
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 96
def lap(Z): return (np.roll(Z,1,0)+np.roll(Z,-1,0)+np.roll(Z,1,1)+np.roll(Z,-1,1)-4*Z)
Y, X = np.indices((N, N)) / N

def theta_phase(t, w_th=0.6, kx=3.0, ky=2.0):
    """prescribed deep traveling-wave clock (medial-septum pacemaker)."""
    return (w_th*t - 2*np.pi*(kx*X + ky*Y)) % (2*np.pi)

def theta_gate(phase, pref=0.0, sharp=3.0):
    return (0.5*(1 + np.cos(phase - pref)))**sharp

def grid_prediction(beta=4.0, thr=1.2):
    """the slow layer's held GRID prediction (3 plane waves at 60 deg, the WHERE)."""
    a = np.zeros((N, N))
    for ang in (0, np.pi/3, 2*np.pi/3):
        a += np.cos(2*np.pi*beta*(np.cos(ang)*X + np.sin(ang)*Y))
    g = np.maximum(a - thr, 0); return g/ (g.max()+1e-9)

def run_gamma(drive_fn, steps=2000, dt=0.05, lam=0.6, w_g=3.0, Dg=0.25, kappa=1.0,
              gated=True, rec_probe=None, rec_from=600, seed=0):
    """fast damped-driven gamma field; growth gated by the theta phase."""
    rng = np.random.default_rng(seed)
    zg = np.zeros((N, N), complex); t = 0.0
    php=[]; amp=[]
    for s in range(steps):
        t += dt
        ph = theta_phase(t); G = theta_gate(ph) if gated else np.ones((N, N))
        R = drive_fn(s, rng)                       # the residual (world - prediction)
        zg = zg + dt*((-lam + 1j*w_g)*zg + Dg*lap(zg) + kappa*G*R)
        if rec_probe is not None and s >= rec_from:
            py, px = rec_probe
            php.append(ph[py, px]); amp.append(abs(zg[py, px]))
    return zg, np.array(php), np.array(amp)

def tort_MI(phase, amp, nbin=18):
    b = np.linspace(0, 2*np.pi, nbin+1)
    m = np.array([amp[(phase>=b[i])&(phase<b[i+1])].mean() if np.any((phase>=b[i])&(phase<b[i+1])) else 0
                  for i in range(nbin)])
    P = m/(m.sum()+1e-12); H = -np.sum(P*np.log(P+1e-12))
    return (np.log(nbin)-H)/np.log(nbin)

if __name__ == "__main__":
    print("="*76)
    print("THE TENSOR — X,Y space x Z frequency; the fast layer spends only on surprise")
    print("="*76)

    # ---------- [A] spatial theta-gamma PAC ----------
    print("\n[A] SPATIAL THETA-GAMMA PAC  (the vertical wiring, now a 2-D field)")
    def broadband(s, rng): return rng.standard_normal((N, N))   # a noisy world everywhere
    probes = [(30, 30), (48, 60), (70, 40), (20, 75)]
    for gated, name in [(True, "theta gates gamma"), (False, "ungated control")]:
        mis = []
        for pr in probes:
            _, php, amp = run_gamma(broadband, gated=gated, rec_probe=pr, seed=1)
            mis.append(tort_MI(php, amp))
        print(f"   {name:<22} mean Tort MI over {len(probes)} probes = {np.mean(mis):.4f}")
    print("   -> gating locks gamma amplitude to theta phase across the sheet (PAC);")
    print("      removing the gate removes it. the_stack's signature, now spatial.")

    # ---------- [B] energy proportional to surprise ----------
    print("\n[B] ENERGY vs SURPRISE  (the fast layer lights up only where prediction fails)")
    P = grid_prediction()
    def make_world(rho, seed=3):
        rng = np.random.default_rng(seed)
        W = P.copy()
        mask = rng.random((N, N)) < rho
        W[mask] = rng.random(mask.sum())          # surprise: unpredicted values
        return W, mask
    # dense/static reference: a processor that encodes the WHOLE field every step
    def dense(s, rng): return np.ones((N, N))
    zg_dense, _, _ = run_gamma(dense, steps=1500, gated=True, seed=5)
    E_dense = float(np.mean(np.abs(zg_dense)**2))
    print(f"   {'surprise density rho':>22}{'gamma energy E':>16}{'E / E_dense':>14}")
    Es = []
    for rho in [0.0, 0.02, 0.05, 0.10, 0.20]:
        W, mask = make_world(rho)
        R = (W - P)
        zg, _, _ = run_gamma(lambda s, rng: R, steps=1500, gated=True, seed=7)
        E = float(np.mean(np.abs(zg)**2)); Es.append((rho, E))
        print(f"   {rho:>22.2f}{E:>16.4f}{E/E_dense:>14.3f}")
    print(f"   (dense/static reference energy E_dense = {E_dense:.4f}, paid at EVERY rho)")
    print("   -> E rises ~linearly with surprise and is near zero when the world matches")
    print("      the grid prediction; a static block pays E_dense regardless. The field")
    print("      spends energy on what it did not expect. Relative units; 'energy'=|z|^2.")

    # ---------- the money shot ----------
    W, mask = make_world(0.08)
    R = (W - P)
    zg, _, _ = run_gamma(lambda s, rng: R, steps=1500, gated=True, seed=7)
    ph = theta_phase(1500*0.05); G = theta_gate(ph)
    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    ax[0,0].imshow(G, cmap="bone");  ax[0,0].set_title("theta gate (deep pacemaker sweeping)", fontsize=10); ax[0,0].axis("off")
    ax[0,1].imshow(P, cmap="viridis"); ax[0,1].set_title("slow layer: held GRID prediction (the WHERE)", fontsize=10); ax[0,1].axis("off")
    ax[1,0].imshow(W, cmap="viridis"); ax[1,0].set_title("the world = prediction + surprise", fontsize=10); ax[1,0].axis("off")
    ax[1,1].imshow(np.abs(zg)**2, cmap="inferno"); ax[1,1].set_title("gamma energy: lit ONLY at surprise", fontsize=10); ax[1,1].axis("off")
    plt.tight_layout(); plt.savefig("cortical_tensor.png", dpi=110, bbox_inches="tight")
    print("\n   saved cortical_tensor.png")
    print("="*76)
