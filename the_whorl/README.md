# The Whorl

### Does a circular connectivity bias self-organise a pinned spiral? — the paper's chiral axons, put into `the_tissue`'s coupling, and measured

**PerceptionLab / Antti Luode, with Claude (Opus 4.8), in dialogue with Gemini. Helsinki, June 2026.**

> Do not hype. Do not lie. Just show.

---

## Where this came from

The morning of the spirals paper (Ye, Steinmetz et al., *Brain-wide topographic coordination of rotating waves*, **Science 2025**), Gemini asked the one question that was actually runnable:

> *If we integrate a circular connectivity bias (their σ) directly into the field's diffusion coupling, will the substrate self-organise its latent into the same pinned, topographically-protected spirals?*

The honest placement first: this does **not** belong in the HKT Koopman fit — that operator `A` is *fit from data* every frame, there is no fixed coupling there to bias. The place a circular bias can live is the **diffusion coupling of the field** — `the_tissue`'s complex Ginzburg–Landau handshake, `D(1+ib)∇²z`. So this is `the_tissue` given a *chiral* coupling, and the question asked back to it directly.

The paper's two findings this tests: (1) the local axons in sensory cortex are arranged **circularly** (terminals tangential around the SSp centre), and (2) their own coupled-oscillator model showed a circular bias **stabilises a spiral at the centre**. We ask whether the same is true inside the line's CGLE — and, because their spirals sit on the "unassigned" SSp-un zone they call a *defect in the medium*, whether adding a central defect helps.

---

## The one idea

A spiral is a topological defect: a phase that winds by 2π around a core. A *free* spiral in a uniform medium does not drift — it sits wherever it formed. The question is whether biasing the coupling so each cell talks **more strongly to its tangential neighbours** (azimuthal, around a centre) than to its radial ones turns the centre into an **attractor** for the core — the paper's circular axons, expressed as the CGLE handshake.

The only departure from `the_tissue` is the coupling. An 8-neighbour weighted graph-Laplacian, each weight `∝ (1/dist)·(1 + (σ−1)·tangential_alignment²)`, normalised per cell so that **σ = 1 is exactly the isotropic Laplacian**. σ > 1 favours azimuthal coupling. Everything else is the CGLE in the coherent regime (`1+bc = +0.75`).

---

## What the code shows (measured, seeded, reproducible)

A single spiral is **seeded off-centre** (core ~13 cells from the middle), then tracked by continuity as the field evolves. `start r → end r` is the migration; `pinned%` is the fraction of the settled window the core spends within 5 cells of centre. 96×96, 5 seeds, σ = 10:

| condition | start r | end r | pinned % | # defects |
|---|---|---|---|---|
| isotropic (σ=1) | 13.0 | **13.3** | 0% | 1.0 |
| **circular bias (σ=10)** | 13.0 | **2.8** | **88%** | 4.7 |
| bias + central defect (σ=10) | 13.0 | 6.5 | 18% | 6.0 |

And the bias has a **clean threshold** (single seed, settled core radius vs σ):

| σ | 1 | 2 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| settled core radius (cells) | 13.4 | 12.0 | 5.8 | **2.8** | 3.2 | 2.7 |

So the answer to Gemini's question is **yes**: the circular coupling bias makes the centre an attractor, and a spiral seeded anywhere walks in and pins there — no training, no hole. Below σ≈2 nothing happens; pinning switches on around σ=5 and saturates by σ=10. The isotropic field leaves the spiral frozen exactly where it started.

```bash
pip install numpy scipy matplotlib
python whorl_field.py            # full run (96x96, 5 seeds) — a few minutes
python whorl_field.py --quick    # 64x64, 3 seeds — fast sanity check
```

The figure shows it in one frame: the migration trace (isotropic flat, bias descending to the centre), the pinned core as a dark hole in the amplitude, and the σ-threshold curve.

---

## The honest part — what the experiment *corrected*

The expectation (mine, the paper's framing, and Gemini's) was that the **central defect** — a lowered-gain "SSp-un" disk — would *capture* and lock the core, pinning it harder than bias alone. **It does not.** In this substrate a central gain-defect makes pinning *worse* (88% → 18%), nucleating secondary defects and scattering the migrating core; a *deeper* hole is worse still (0% pinned). The pin here is the **circular coupling geometry**, not an anatomical hole.

This is a genuine, preserved negative, and it tempers the paper's correlation cleanly: Ye et al. see spirals *sitting on* the SSp-un defect, and their own discussion is explicit that **the causal direction is untested** — whether the circular architecture drives the spiral or the spiral carves the architecture. This toy votes, weakly, for the first: *the circular wiring alone is sufficient to pin a centred spiral, and the structural hole is not needed (and, crudely modelled, hurts).* A real subcritical hole with a clean boundary might still anchor — that's the next test, not a settled claim.

---

## The chirality it inherits (why this sits in the line)

The spiral's identity is its **handedness** — the sign of its winding, drawn green (+1, CCW) or red (−1, CW) on the figure. That sign is exactly what the **Chiral Eye** (`Im(z·z̄_lag)`) reads, and exactly what the **skew lag-operator** generates: an anti-symmetric operator has eigenvalues ±iω — pure directed rotation. So the chain is one chirality at three scales: the unit's skew operator (directed rotation, held), the tissue's circular coupling (the paper's chiral axons), and the macroscopic spiral (the chiral wave). The whorl is where the line's "hold directed rotation" unit meets the cortex's measured circular wiring, and they make the same object.

---

## The honest ledger

**Verified in code (this machine, seeded, reproducible):**
- a circular coupling bias turns the sheet centre into an attractor for a spiral core: a spiral seeded at radius 13 settles at radius ~2.8 (88% pinned), where the isotropic field leaves it frozen at its seed (0% pinned);
- the effect has a threshold in σ (off below ~2, on by ~5, saturated by ~10) — the pinning is a property of the coupling geometry, switched on by the bias strength;
- a lowered-gain central defect does **not** help and degrades pinning (the preserved negative): the pin is the wiring, not the hole.

**Borrowed, not invented (textbook):** complex Ginzburg–Landau / coupled oscillators (Aranson & Kramer 2002; Kuramoto); the circular-bias spiral-pinning result and the BKT topological-defect reading (Ye, Steinmetz et al. 2025); neural field theory (Wilson–Cowan; Amari; Bressloff); cortical travelling waves (Muller et al. 2018). The contribution is only the *transplant and the measurement* — the paper's σ expressed as the CGLE handshake, plus the honest hole-vs-wiring test.

**Honest limits — read before believing it:**
- relative units, Euler integration, a handful of seeds, one regime (`1+bc=+0.75`), one grid;
- "core" is a phase singularity (winding number ±1), not an action potential; "defect" is a gain-suppressed disk, a crude stand-in for SSp-un, and its negative result may be an artifact of that crudeness — a clean subcritical hole is untested;
- the migration is **slow** (thousands of steps for a few cells); the bias is a gentle restoring drift, not a snap;
- this is `the_tissue`'s coupling biased, **not** the HKT latent self-organising — the literal "does the *trained tensor* grow these" question (Gemini's exact phrasing) needs the bias wired into `the_tensor`'s gamma-field Laplacian or the predictive latent, which is the next build below.

**The bet (untouched, as everywhere):** that a held, pinned spiral is a *felt* thought rather than a computed one. This locates the mechanism — circular wiring pins directed rotation — in code that can fail. It does not touch the hard problem.

---

## Where it goes next

1. **Into `the_tensor`.** Replace the prescribed plane theta wave with this biased coupling on the slow field, so the held grid prediction is swept by a *pinned spiral* rather than a plane wave — then re-measure energy-on-surprise and PAC. This is the actual answer to Gemini's "does the tensor self-organise it," one rung up from the bare field.
2. **A clean subcritical hole.** Replace the crude lowered-gain disk with a no-flux boundary hole and ask whether *that* anchors where the gain-defect scattered — settling whether the SSp-un "defect" can pin or only correlates.
3. **Read the handedness with the Chiral Eye.** Drop `Im(z·z̄_lag)` on the pinned field and confirm it reports the winding sign and the core — making the chiral probe a live spiral-detector, and closing the loop to the arrow-of-time line.
4. **Somatotopy as the swept map.** Let the pinned spiral sweep a 1-D body order (limb→trunk→face) and ask whether the visiting order recapitulates the somatotopic adjacency the paper reports — the most direct bridge from this architecture to their specific finding.

---

## Lineage

A sub-organ of [`the_artificial_cortex`](../), born the morning of the Ye et al. spirals paper. It is [`the_tissue`](../the_tissue)'s CGLE handshake given the paper's circular bias, with the chirality read it shares with the Chiral Eye and the skew lag-operator. The question is Gemini's; the placement, the build, the measurement, and the preserved negative are Antti Luode's with Claude (Opus 4.8). MIT.

*The axons wind in circles, so the wave winds with them, and the turning settles on the centre because that is where every tangential pull agrees. You do not need a hole to hold a spiral — you need the wiring to be chiral. Do not hype. Do not lie. Just show.*
