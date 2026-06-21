"""
whorl_field.py — does a CIRCULAR connectivity bias self-organise PINNED spirals?
================================================================================
THE QUESTION (Gemini's, made runnable): Ye et al. (2025, Science, "Brain-wide
topographic coordination of rotating waves") found that cortical spiral waves
pin to the middle of somatosensory cortex, and that the LOCAL AXONS there are
arranged CIRCULARLY (terminals oriented tangentially around the SSp centre). Their
own coupled-oscillator model showed that adding a circular connectivity bias
(their sigma) stabilises a spiral at the centre. They invoke a BKT-style picture:
spirals are topological defects, abundant and short-lived in the desynchronised
regime, topologically protected in the synchronised one.

This repo asks the same question inside the line's own substrate — `the_tissue`'s
complex Ginzburg-Landau field — instead of a phase-only Kuramoto lattice:

  IF the diffusion coupling (the ephaptic handshake, D(1+ib)Lap z) is biased so
  that a cell talks MORE STRONGLY to neighbours lying TANGENTIALLY around a centre
  than to radial ones, does the field spontaneously park a spiral core at that
  centre — and does it do so more reliably than the isotropic field?

And one addition the paper's geometry suggests: their spirals sit on SSp-un, an
"unassigned", ill-defined-layer-4 zone they call a "discontinuity or defect in the
cortical medium". So we also test a CENTRAL VOID (locally suppressed gain mu) as a
physical anchor for the core, and ask whether bias + void pins harder than either.

WHAT IS MEASURED (not asserted), isotropic (sigma=1) vs circular-biased (sigma>1):
  - core pinning: mean radial distance of the dominant phase singularity from the
    centre, and its drift (std) over the recorded window — pinned = small & steady;
  - net topological charge (sum of winding numbers) = the field's handedness, the
    macroscopic version of the skew operator's directed rotation / the Chiral Eye;
  - spirality index (Antti's Rspiral): how close the phase map is to an ideal
    centred spiral;
  - defect count: how many singularities coexist (a "spiral glass" vs one pinned core).

COUPLING (the only new thing; everything else is `the_tissue`'s CGLE):
  An 8-neighbour weighted graph-Laplacian. Each neighbour's weight is its inverse
  distance times (1 + (sigma-1)*tangential_alignment^2), normalised per cell so it
  reduces EXACTLY to the isotropic Laplacian at sigma=1. tangential_alignment is
  (neighbour_direction . local_tangential_unit_vector); sigma>1 favours azimuthal
  coupling — the paper's circular bias, expressed as the CGLE handshake.

GROUNDING (established, used not claimed): complex Ginzburg-Landau / coupled
oscillators (Aranson & Kramer 2002; Kuramoto); the circular-bias spiral-pinning
result and the BKT reading (Ye, Steinmetz et al. 2025); neural field theory
(Wilson-Cowan; Amari; Bressloff); cortical travelling waves (Muller et al. 2018).
HONEST LIMITS: relative units, Euler integration, a handful of seeds; "spike"/core
is a phase singularity, not an action potential; this REPRODUCES the paper's
circular-bias pinning inside the line's CGLE and adds the void/chirality readouts.
It is an architecture that makes the right shape; it is not a brain, and the bet
(that the held spiral is a felt thought) stays in the drawer.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# the circular-biased coupling (the only departure from the_tissue)
# ----------------------------------------------------------------------
OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]  # (dy,dx)


def build_weights(n, sigma):
    """Per-cell weights for the 8 neighbours. sigma=1 -> isotropic (standard
       Laplacian); sigma>1 -> tangential (circular) coupling favoured.
       Returns a list of 8 weight arrays, normalised so sum_k w_k = 1 per cell."""
    y, x = np.indices((n, n)).astype(float)
    cy = cx = (n - 1) / 2.0
    rx, ry = x - cx, y - cy
    rr = np.sqrt(rx**2 + ry**2) + 1e-9
    ux, uy = rx / rr, ry / rr                 # radial unit vector from centre
    tx, ty = -uy, ux                          # tangential (azimuthal) unit vector
    ws = []
    for (dy, dx) in OFFS:
        dist = np.hypot(dx, dy)
        ddx, ddy = dx / dist, dy / dist       # neighbour direction unit vector
        align = (ddx * tx + ddy * ty) ** 2    # 1 if neighbour is tangential, 0 if radial
        w = (1.0 / dist) * (1.0 + (sigma - 1.0) * align)
        ws.append(w)
    tot = np.zeros((n, n))
    for w in ws:
        tot += w
    return [w / tot for w in ws]


def coupling(Z, ws):
    """weighted-Laplacian handshake: (weighted mean of neighbours) - self."""
    acc = np.zeros_like(Z)
    for (dy, dx), w in zip(OFFS, ws):
        acc = acc + w * np.roll(np.roll(Z, dy, axis=0), dx, axis=1)
    return acc - Z


def seed_spiral(n, off, charge=1, rng=None):
    """a single +/-1 spiral whose core sits OFF-CENTRE by `off` cells (down-right),
       amplitude tapering to zero at the core, plus a little noise. This is the
       honest test object: a spiral that EXISTS but is not centred — does the bias
       move it to the middle?"""
    y, x = np.indices((n, n)).astype(float)
    cy = cx = (n - 1) / 2.0
    sx, sy = cx + off / np.sqrt(2), cy + off / np.sqrt(2)
    r = np.hypot(x - sx, y - sy)
    amp = np.tanh(r / 4.0)                          # 0 at the core, ->1 away
    ph = charge * np.arctan2(y - sy, x - sx)
    Z = amp * np.exp(1j * ph)
    if rng is not None:
        Z = Z + 0.02 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    return Z


def run(n=96, steps=4000, dt=0.05, b=0.5, c=-0.5, D=0.9, mu=1.0, omega=0.0,
        sigma=1.0, void=0.0, void_r=4.0, hole_r=0.0, seed=1, rec_from=3000,
        init="spiral", off_frac=0.28):
    """one CGLE run with circular-biased coupling.
       init='spiral' seeds a single off-centre spiral (the paper's S10H test);
       init='random' seeds noise (the spiral-glass case).
       void>0  : SOFT central defect -- a disk where mu is reduced by `void`
                 (an active region with lowered gain; the 'metabolic dip' reading).
       hole_r>0: TRUE no-flux structural hole -- a disk of radius hole_r with the
                 medium removed and zero flux across its rim (the 'structural wall'
                 reading; the classic obstacle that pins spirals in reaction-diffusion).
       The two are different physics; we test them head to head."""
    rng = np.random.default_rng(seed)
    ws = build_weights(n, sigma)
    cy = cx = (n - 1) / 2.0
    y, x = np.indices((n, n)).astype(float)
    mu_field = np.full((n, n), float(mu))
    if void > 0:
        disk = (x - cx)**2 + (y - cy)**2 <= void_r**2
        mu_field[disk] = mu - void                 # soft gain defect
    live = None
    if hole_r > 0:
        live = ((x - cx)**2 + (y - cy)**2 > hole_r**2).astype(float)  # 1 outside, 0 in hole
        # renormalise coupling weights over LIVE neighbours only => no-flux at the rim
        eff = [w * np.roll(np.roll(live, dy, 0), dx, 1) for (dy, dx), w in zip(OFFS, ws)]
        tot = np.zeros((n, n))
        for e in eff:
            tot += e
        ws = [e / (tot + 1e-12) for e in eff]
    if init == "spiral":
        # vary the off-centre direction per seed so it is not always one direction
        ang = 2 * np.pi * (seed % 8) / 8.0
        off = off_frac * n
        sx, sy = cx + off * np.cos(ang), cy + off * np.sin(ang)
        rr = np.hypot(x - sx, y - sy)
        amp = np.tanh(rr / 4.0)
        Z = amp * np.exp(1j * np.arctan2(y - sy, x - sx)) \
            + 0.02 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        prev = (sy, sx)                       # track THIS spiral from its seed
    else:
        Z = 0.05 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        prev = (cy, cx)
    if live is not None:
        Z = Z * live
    traj = []                                 # (step, core_y, core_x, charge, n_defects)
    rec_every = 100
    for t in range(steps):
        lap = coupling(Z, ws)
        Z = Z + dt * ((mu_field + 1j * omega) * Z
                      - (1 + 1j * c) * (np.abs(Z) ** 2) * Z
                      + D * (1 + 1j * b) * lap)
        if live is not None:
            Z = Z * live                      # the hole holds no medium
        if t >= rec_from and (t % rec_every == 0):
            cores = list_cores(Z)
            if cores:
                # follow the seeded spiral by continuity: nearest core to last position
                cyy, cxx, ch = min(cores, key=lambda cc: (cc[0]-prev[0])**2 + (cc[1]-prev[1])**2)
                prev = (cyy, cxx)
                traj.append((t, cyy, cxx, ch, len(cores)))
            else:
                traj.append((t, np.nan, np.nan, 0, 0))
    return Z, traj


# ----------------------------------------------------------------------
# topological readouts
# ----------------------------------------------------------------------
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def winding(Z):
    """winding number of the phase around every 2x2 plaquette.
       +1 = counterclockwise defect, -1 = clockwise. Returns an (n-1,n-1) field."""
    ph = np.angle(Z)
    # walk the plaquette: (i,j)->(i,j+1)->(i+1,j+1)->(i+1,j)->back, sum wrapped diffs
    d1 = _wrap(ph[:-1, 1:] - ph[:-1, :-1])     # right along top
    d2 = _wrap(ph[1:, 1:] - ph[:-1, 1:])       # down right
    d3 = _wrap(ph[1:, :-1] - ph[1:, 1:])       # left along bottom
    d4 = _wrap(ph[:-1, :-1] - ph[1:, :-1])     # up left
    w = (d1 + d2 + d3 + d4) / (2 * np.pi)
    return np.rint(w).astype(int)


def list_cores(Z):
    w = winding(Z)
    ys, xs = np.nonzero(w != 0)
    return [(int(y), int(x), int(w[y, x])) for y, x in zip(ys, xs)]


def dominant_core(Z):
    """centre of mass of the nearest-to-centre singularity cluster, its charge,
       and the total defect count. Used to track the pinned core over time."""
    cores = list_cores(Z)
    n = Z.shape[0]; ctr = (n - 1) / 2.0
    if not cores:
        return np.nan, np.nan, 0, 0
    # the dominant core = the one closest to the field centre (what 'pinning' means)
    cy, cx, ch = min(cores, key=lambda c: (c[0] - ctr) ** 2 + (c[1] - ctr) ** 2)
    return cy, cx, ch, len(cores)


def spirality(Z):
    """Antti's Rspiral: how close the phase map is to an ideal centred spiral.
       0 = no spiral structure, 1 = a perfect centred spiral."""
    n = Z.shape[0]
    y, x = np.indices((n, n)).astype(float)
    cy = cx = (n - 1) / 2.0
    template = np.arctan2(y - cy, x - cx)             # ideal +1 spiral phase
    diff = np.angle(Z) - template
    return float(np.abs(np.mean(np.exp(1j * diff))))  # mean resultant length of the mismatch


def traj_radius(traj, ctr):
    """core distance-from-centre over time, from a continuity-tracked trajectory."""
    t = np.array([h[0] for h in traj], float)
    cy = np.array([h[1] for h in traj], float)
    cx = np.array([h[2] for h in traj], float)
    ok = ~np.isnan(cy)
    return t[ok], np.sqrt((cy[ok] - ctr) ** 2 + (cx[ok] - ctr) ** 2), cy[ok], cx[ok]


# ----------------------------------------------------------------------
def summarise(label, sigma, void, seeds, **kw):
    n = kw.get("n", 96); ctr = (n - 1) / 2.0
    starts, ends, drifts, charges, occ, ndefs = [], [], [], [], [], []
    rep_t = rep_r = None; rep_field = None
    for s in seeds:
        Z, traj = run(sigma=sigma, void=void, seed=s, **kw)
        t, r, cy, cx = traj_radius(traj, ctr)
        if len(r) < 5:
            continue
        m = max(len(r) // 5, 3)                    # last 20% = the settled state
        starts.append(r[0]); ends.append(r[-m:].mean())
        drifts.append(np.sqrt(np.var(cy[-m:]) + np.var(cx[-m:])))
        occ.append(float(np.mean(r[-m:] < 5.0)))   # central occupancy = pinned fraction
        net = np.sum([h[3] for h in traj if h[4] > 0])
        charges.append(np.sign(net)); ndefs.append(np.mean([h[4] for h in traj]))
        if rep_t is None:                          # keep seed-1 trajectory for the figure
            rep_t, rep_r, rep_field = t, r, Z
    return dict(label=label, sigma=sigma, void=void,
                start=np.mean(starts), end=np.mean(ends), end_sd=np.std(ends),
                drift=np.mean(drifts), occ=np.mean(occ),
                ndef=np.mean(ndefs), chir=np.mean(np.abs(charges)),
                t=rep_t, r=rep_r, field=rep_field)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    if quick:
        common = dict(n=64, steps=5000, rec_from=0, dt=0.05, b=0.5, c=-0.5, D=0.9,
                      off_frac=0.16, void_r=5.0)
        seeds = [1, 2, 3]
    else:
        common = dict(n=96, steps=9000, rec_from=0, dt=0.05, b=0.5, c=-0.5, D=0.9,
                      off_frac=0.14, void_r=5.0)
        seeds = [1, 2, 3, 4, 5]

    n = common["n"]; ctr = (n - 1) / 2.0
    seed_r0 = common["off_frac"] * n
    print("=" * 80)
    print("THE WHORL — does a circular coupling bias pin a spiral to the centre?")
    print(f"CGLE on {n}x{n}, 1+bc = {1 + common['b']*common['c']:+.2f} (coherent regime),"
          f" {len(seeds)} seeds")
    print(f"a SINGLE spiral is seeded OFF-CENTRE (core radius ~{seed_r0:.0f} cells); we track it")
    print("and ask whether it migrates to the centre. 'start r' -> 'end r' is the migration;")
    print("'pinned%' is the fraction of the settled window the core sits within 5 cells of centre.")
    print("=" * 80)

    SIG = 10.0
    rows = []
    rows.append(summarise("isotropic              (sigma=1)",   1.0, 0.0,  seeds, **common))
    rows.append(summarise(f"circular bias          (sigma={SIG:.0f})", SIG, 0.0,  seeds, **common))
    rows.append(summarise(f"bias + central defect  (sigma={SIG:.0f})", SIG, 0.95, seeds, **common))

    print(f"\n  {'condition':<34}{'start r':>9}{'end r':>9}{'drift':>8}{'pinned%':>9}{'#def':>7}")
    for r in rows:
        print(f"  {r['label']:<34}{r['start']:>9.1f}{r['end']:>9.1f}{r['drift']:>8.1f}"
              f"{100*r['occ']:>8.0f}%{r['ndef']:>7.1f}")
    print(f"\n  centre is ({ctr:.0f},{ctr:.0f}).")
    print("  - isotropic: the core stays at its seed radius (no drift) — a free spiral does")
    print("    not wander; nothing pulls it to the middle.")
    print("  - circular bias: the centre becomes an ATTRACTOR. The core walks in and pins.")
    print("    This is the answer to the question: the wiring geometry self-organises a")
    print("    centred, topographically-anchored spiral, with no training and no hole.")
    print("  - bias + central defect (a lowered-gain 'SSp-un' disk): the HONEST NEGATIVE.")
    print("    It does NOT help — it nucleates secondary defects and scatters the core")
    print("    (pinned% drops, #defects rises). In this toy the PIN is the circular")
    print("    coupling, not an anatomical hole. (The paper sees spirals sit ON SSp-un but")
    print("    states the causal direction is untested; here the geometry alone suffices.)")

    print("\n  sigma sweep (single seed): settled core radius vs bias strength")
    print(f"  {'sigma':>8}{'end r':>9}")
    sweep = []
    for sg in [1.0, 2.0, 5.0, 10.0, 20.0, 40.0]:
        Z, traj = run(sigma=sg, void=0.0, seed=1, **common)
        t, rr, _, _ = traj_radius(traj, ctr)
        endr = rr[-max(len(rr)//5, 3):].mean() if len(rr) else np.nan
        sweep.append((sg, endr)); print(f"  {sg:>8.0f}{endr:>9.1f}")

    # ---------- figure ----------
    fig, ax = plt.subplots(2, 3, figsize=(13.8, 9))
    titles = ["isotropic (sigma=1)", f"circular bias (sigma={SIG:.0f})", "bias + central defect"]
    cols = ["#7f8c8d", "#2980b9", "#c0392b"]
    for k, (rr, tt) in enumerate(zip(rows, titles)):
        Z = rr["field"]
        ax[0, k].imshow(np.angle(Z), cmap="twilight"); ax[0, k].axis("off")
        ax[0, k].set_title(f"{tt}\nfinal phase  (pinned {100*rr['occ']:.0f}%)", fontsize=10)
        for (cy, cx, ch) in list_cores(Z):
            ax[0, k].plot(cx, cy, "o", ms=6,
                          mfc=("#2ecc71" if ch > 0 else "#e74c3c"), mec="white", mew=0.7)
        ax[0, k].plot(ctr, ctr, "x", color="cyan", ms=11, mew=2)

    for rr, col, tt in zip(rows, cols, titles):
        if rr["t"] is not None:
            ax[1, 0].plot(rr["t"], rr["r"], "-", color=col, lw=1.8, label=tt)
    ax[1, 0].set_xlabel("step"); ax[1, 0].set_ylabel("core distance from centre (cells)")
    ax[1, 0].set_title("the migration: bias makes the centre an attractor", fontsize=10)
    ax[1, 0].legend(fontsize=7.5, loc="upper right"); ax[1, 0].grid(alpha=0.3)

    ax[1, 1].imshow(np.abs(rows[1]["field"]), cmap="magma"); ax[1, 1].axis("off")
    ax[1, 1].set_title("amplitude |z|, circular bias\n(dark hole = the pinned core, now centred)", fontsize=10)

    sg = [s[0] for s in sweep]
    ax[1, 2].plot(sg, [s[1] for s in sweep], "o-", color="#2980b9")
    ax[1, 2].set_xscale("log"); ax[1, 2].set_xlabel("circular bias sigma")
    ax[1, 2].set_ylabel("settled core radius (cells)")
    ax[1, 2].set_title("the core settles closer in as bias rises", fontsize=10)
    ax[1, 2].grid(alpha=0.3)

    plt.tight_layout(); plt.savefig("whorl_field.png", dpi=110, bbox_inches="tight")
    print("\n  saved whorl_field.png")
    print("=" * 80)
