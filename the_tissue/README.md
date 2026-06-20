# The Tissue

### The computational substrate: a grid of skew-operator units coupled into a moving wave — the complex Ginzburg–Landau field, with the ephaptic coupling as its diffusion term

**PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.**

> Do not hype. Do not lie. Just show.

---

## The one idea

One rotation island of a unit's skew operator is a 2-D plane, and a coordinate in a rotating plane *is a complex number* `z`, evolving as `ż = iω z`. A unit holding one island is a complex oscillator at rate `ω`. Couple a lattice of them to their neighbours' held phase with a **complex-coefficient Laplacian**, add a saturating nonlinearity, and you have the **complex Ginzburg–Landau equation** — the universal field of coupled oscillators:

```
ż = (μ + iω) z  −  (1 + ic)|z|² z  +  D(1 + ib)·∇²z
                                       └──── the handshake ────┘
```

That coupling term is the **ephaptic field**: weak, local, always on, coupling each unit to the phase its neighbours hold. Subthreshold = the continuous `z` field; a spike = a threshold event riding on top. Traveling waves, spiral waves, and phase-locked spatial domains are its solutions.

**Two senses of "island," kept apart.** The SpectralIslands line's islands are *spectral* — eigenplanes of one unit's skew operator. The waves here make *spatial* islands — phase-locked domains across the sheet. The diffusion term is what converts a population of spectral-island-holders into spatial islands.

---

## What the code shows (one number decides the regime)

The **Benjamin–Feir–Newell** number `1 + bc` sets the regime. Measured on a 128×128 sheet:

| | ordered (1+bc=+0.75) | turbulent (1+bc=−1.0) |
|---|---|---|
| defect density (spiral cores) | 0.17% (~28) | **2.22% (~360)** |
| traveling-wave cross-corr | **−0.96** at lag 60 → speed 0.70 | 0.46 ragged at lag −477 |
| spatial correlation length | 6 cells | 6 cells — *did not separate* |

Defect density and the traveling-wave coherence separate the regimes cleanly; correlation length did **not** (at this coupling the structure scale is diffusion-set in both, ~6 cells — reported, not hidden). Flip the sign of `1+bc` and a coherent wave of few spiral cores becomes defect turbulence. The disorientation of the "where am I" agent is this coherence loss, made visible.

```bash
python spectral_field.py
```

---

## Ledger

**Established (used, not claimed):** complex Ginzburg–Landau / coupled oscillators (Aranson & Kramer 2002; Kuramoto); neural field theory (Wilson–Cowan 1973; Amari 1977; Bressloff); cortical traveling waves (Muller et al. 2018); spreading depression as a cortical wave (Leão 1944); weak, real ephaptic coupling (Jefferys 1995; Anastassiou et al. 2011).

**The line's own part:** the *identification* — held state = the skew lag-operator, spike = a coincidence, ephaptic field = the diffusion term — not the dynamics, which are textbook.

**Honest limits:** relative units, Euler integration, one seed per regime; "spike" reads as a phase crossing, not an action potential; the correlation-length measure did not discriminate here. **The bet (untouched):** that the wave is a *felt* thought.
