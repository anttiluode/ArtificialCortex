"""
whorl_hole.py — does the SSp-un "hole" pin the spiral, and if so, what KIND of hole?
====================================================================================
`the_whorl` showed the circular coupling bias alone pins a spiral at the centre
(88%), while a SOFT central defect (a lowered-gain disk) made it WORSE (18%) by
nucleating secondary defects. That left an honest open question, which is the one
Gemini posed: was that negative real, or an artifact of modelling the hole as a
soft active region?

This settles it by testing the OTHER kind of hole. Two physically distinct objects:
  - SOFT defect (void): a disk of reduced gain mu -- still an active medium, just
    weaker. The 'metabolic dip' reading of SSp-un.
  - TRUE no-flux hole (hole_r): the medium removed inside a disk, with zero flux
    across its rim (Neumann boundary). The 'structural wall' reading. This is the
    classic obstacle that PINS spiral waves in reaction-diffusion (cardiac scars,
    inexcitable obstacles; Davidenko, Pertsov, et al.).

Head to head, sigma=10, a single spiral seeded off-centre (~13 cells), tracked:
  - does the no-flux hole anchor the core (low radius, LOW DRIFT, no extra defects)?
  - or does it, like the soft defect, scatter it?

The verdict decides the architecture: if the no-flux wall anchors cleanly, SSp-un
as a STRUCTURAL boundary is a live hypothesis and the paper's pinning story holds
in mechanism; if even a clean wall fails to beat bias-alone, the circular wiring is
the sole pin and the hole is correlational tissue.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from whorl_field import run, list_cores, traj_radius


def measure(label, seeds, ctr, **kw):
    starts, ends, drifts, occ, ndefs = [], [], [], [], []
    rep_t = rep_r = rep_field = None
    for s in seeds:
        Z, traj = run(seed=s, **kw)
        t, r, cy, cx = traj_radius(traj, ctr)
        if len(r) < 5:
            continue
        m = max(len(r) // 5, 3)
        starts.append(r[0]); ends.append(r[-m:].mean())
        drifts.append(np.sqrt(np.var(cy[-m:]) + np.var(cx[-m:])))
        occ.append(float(np.mean(r[-m:] < 5.0)))
        ndefs.append(np.mean([h[4] for h in traj]))
        if rep_t is None:
            rep_t, rep_r, rep_field = t, r, Z
    return dict(label=label, start=np.mean(starts), end=np.mean(ends),
                drift=np.mean(drifts), occ=np.mean(occ), ndef=np.mean(ndefs),
                t=rep_t, r=rep_r, field=rep_field)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n = 64 if quick else 96
    steps = 5000 if quick else 9000
    seeds = [1, 2, 3] if quick else [1, 2, 3, 4, 5]
    base = dict(n=n, steps=steps, rec_from=0, dt=0.05, b=0.5, c=-0.5, D=0.9,
                off_frac=0.14, sigma=10.0)
    ctr = (n - 1) / 2.0

    print("=" * 80)
    print("THE WHORL / hole physics — soft gain-defect vs TRUE no-flux structural hole")
    print(f"CGLE {n}x{n}, sigma=10, single off-centre spiral (~{0.14*n:.0f} cells), {len(seeds)} seeds")
    print("the question: does a structural (no-flux) hole ANCHOR the core where the")
    print("soft gain-defect SCATTERED it? a good anchor = low end r, LOW drift, few defects.")
    print("=" * 80)

    rows = [
        measure("bias only            (no hole)",   seeds, ctr, void=0.0,  hole_r=0, **base),
        measure("bias + soft defect   (mu dip)",    seeds, ctr, void=0.95, hole_r=0, void_r=5, **base),
        measure("bias + no-flux hole  (r=5 wall)",  seeds, ctr, void=0.0,  hole_r=5, **base),
    ]
    print(f"\n  {'condition':<34}{'start r':>9}{'end r':>9}{'drift':>8}{'pinned%':>9}{'#def':>7}")
    for r in rows:
        print(f"  {r['label']:<34}{r['start']:>9.1f}{r['end']:>9.1f}{r['drift']:>8.1f}"
              f"{100*r['occ']:>8.0f}%{r['ndef']:>7.1f}")

    print("\n  no-flux hole radius sweep (single seed): does an optimal peg size exist?")
    print(f"  {'hole_r':>8}{'end r':>9}{'drift':>8}{'#def':>7}")
    sweep = []
    for hr in [2, 3, 4, 5, 6, 8, 10]:
        Z, traj = run(seed=1, void=0.0, hole_r=hr, **base)
        t, r, cy, cx = traj_radius(traj, ctr)
        m = max(len(r) // 5, 3)
        endr = r[-m:].mean(); dr = np.sqrt(np.var(cy[-m:]) + np.var(cx[-m:]))
        nd = np.mean([h[4] for h in traj])
        sweep.append((hr, endr, dr, nd)); print(f"  {hr:>8d}{endr:>9.1f}{dr:>8.1f}{nd:>7.1f}")

    # ---------- figure ----------
    fig, ax = plt.subplots(2, 3, figsize=(13.8, 9))
    cols = ["#2980b9", "#c0392b", "#27ae60"]
    for k, r in enumerate(rows):
        Z = r["field"]
        ax[0, k].imshow(np.angle(Z), cmap="twilight"); ax[0, k].axis("off")
        ax[0, k].set_title(f"{r['label'].split('(')[0].strip()}\nphase  (pinned {100*r['occ']:.0f}%,"
                           f" drift {r['drift']:.1f})", fontsize=9.5)
        for (cy, cx, ch) in list_cores(Z):
            ax[0, k].plot(cx, cy, "o", ms=6,
                          mfc=("#2ecc71" if ch > 0 else "#e74c3c"), mec="white", mew=0.7)
        ax[0, k].plot(ctr, ctr, "x", color="cyan", ms=11, mew=2)

    for r, col in zip(rows, cols):
        if r["t"] is not None:
            ax[1, 0].plot(r["t"], r["r"], "-", color=col, lw=1.8,
                          label=r["label"].split("(")[0].strip())
    ax[1, 0].set_xlabel("step"); ax[1, 0].set_ylabel("core distance from centre (cells)")
    ax[1, 0].set_title("the migration & what holds (or scatters) the core", fontsize=10)
    ax[1, 0].legend(fontsize=7.5, loc="upper right"); ax[1, 0].grid(alpha=0.3)

    ax[1, 1].imshow(np.abs(rows[2]["field"]), cmap="magma"); ax[1, 1].axis("off")
    ax[1, 1].set_title("amplitude |z|, no-flux hole\n(black disk = the structural wall)", fontsize=10)

    hr = [s[0] for s in sweep]
    ax[1, 2].plot(hr, [s[1] for s in sweep], "o-", color="#27ae60", label="end radius")
    ax[1, 2].plot(hr, [s[2] for s in sweep], "s--", color="#8e44ad", label="drift")
    ax[1, 2].set_xlabel("no-flux hole radius (cells)"); ax[1, 2].set_ylabel("cells")
    ax[1, 2].set_title("anchoring vs peg size", fontsize=10)
    ax[1, 2].legend(fontsize=8); ax[1, 2].grid(alpha=0.3)

    plt.tight_layout(); plt.savefig("whorl_hole.png", dpi=110, bbox_inches="tight")
    print("\n  saved whorl_hole.png")
    print("=" * 80)
