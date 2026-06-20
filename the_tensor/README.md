# The Tensor

### The horizon build — X,Y space × Z frequency, where the fast layer spends energy only on what the slow layer did not predict

**PerceptionLab / Antti Luode, with Claude (Opus 4.8), in dialogue with Gemini. Helsinki, June 2026.**

> Do not hype. Do not lie. Just show.

---

## The one idea

The three organs become one block. A **slow (theta) layer** sweeps the sheet as a prescribed traveling wave — the deep medial-septal pacemaker — and holds a **grid-patterned prediction** `P`: the expected spatial world, the *where*. A **fast (gamma) layer** is a damped, *driven* field that encodes only the **residual** `R = world − prediction`: the *now* that the grid did not foresee — and its growth is gated by the local theta phase (the chandelier on the clock).

So the computation is not a tower of weights flashing top to bottom. It is a **wave of coincidence**: the fast layer lights up only where the sensory world contradicts the slow grid prediction, and only when theta opens the gate. This is predictive/residual coding (Rao & Ballard 1999; the ELL negative image) stacked on the theta–gamma code (Lisman & Jensen 2013), run as a spatial field.

---

## What the code measures (two numbers, not a story)

**[A] Spatial theta–gamma PAC** — `the_stack` lifted from a 1-D signal into a 2-D field. Gamma is driven by a broadband world and gated by the theta wave; Tort's Modulation Index is read at four probe sites against an ungated control:

| arm | mean Tort MI (4 probes) |
|---|---|
| **theta gates gamma** | **0.145** |
| ungated control | 0.005 |

The gate locks gamma amplitude to theta phase across the sheet; remove it and the coupling is gone.

**[B] Energy proportional to surprise** — the distinctive claim, made into a curve. The world = the grid prediction plus surprise patches at density `ρ`; gamma energy `E = ⟨|z|²⟩` is measured against a static/dense processor that pays the same cost at every cell regardless of `ρ`:

| surprise density ρ | gamma energy E | E / E_dense |
|---|---|---|
| 0.00 | 0.0000 | **0.000** |
| 0.02 | 0.0001 | 0.006 |
| 0.05 | 0.0004 | 0.016 |
| 0.10 | 0.0007 | 0.028 |
| 0.20 | 0.0013 | 0.048 |

`E_dense = 0.027`, **paid at every ρ** by the static block. The field is silent when the world matches its grid prediction, and its cost rises with surprise (roughly linearly, slightly sublinear at high `ρ` as patches start to overlap) — at 10% surprise it spends ~2.8% of the dense cost. **It spends energy on what it did not expect.**

```bash
python cortical_tensor.py
```

The figure shows it in one frame: the theta gate sweeping, the held grid prediction, the world with surprise scattered into it, and the gamma energy lit *only* at surprise — and only along the open phase of the gate.

---

## Honest ledger

**Verified in code (seeded, reproducible):** spatial theta–gamma PAC from gating (MI 0.145 vs 0.005); gamma energy near zero when the world matches the grid prediction and rising with surprise density, where a static block pays a constant cost.

**Established (used, not claimed):** predictive/residual coding (Rao & Ballard 1999; ELL negative image — Bell/Sawtell); theta–gamma code and cross-frequency coupling (Lisman & Jensen 2013; Canolty & Knight 2010; Tort 2010); medial-septal theta pacing; neural field theory (Wilson–Cowan; Amari).

**Honest limits:** theta is a **prescribed** pacemaker here, not self-organised; gamma is a damped-driven *linear* field; the "static processor" is a constant-cost stand-in, **not** a measured transformer; "energy" is `|z|²`, a proxy, not Joules; one toy world, relative units. This shows the *architecture* spends on surprise — it is not a benchmark and not a brain.

**The bet (untouched):** that the lit gamma is a *felt* surprise rather than a computed residual. The block locates the mechanism; it does not touch the hard problem.

---

## Where it goes next

1. **Self-organise the theta wave** instead of prescribing it — let the slow layer be its own CGLE field (the tissue), so the clock emerges rather than being imposed.
2. **Close the loop**: let the slow layer *learn* its prediction `P` from the residual the fast layer reports (the mirror-gate distillation), so the grid is grown from experience, not handed in.
3. **Run it on the real HKT video stream**: the slow layer holds the predicted next frame, the fast layer encodes only the surprise, and the energy meter reads the actual compute saved — joining this to the gated-diffusion / DeepCache direction, where the saving is a real wall-clock number.

---

## Lineage

The integrator of the artificial cortex: [the tissue](../the_tissue)'s wave, [the grid](../the_grid)'s held prediction, and [the stack](../the_stack)'s theta gate, in one block whose only spend is on surprise. The framing is Antti Luode's, in dialogue with Gemini; the build, the measurements, and this document were developed with Claude (Opus 4.8). MIT.
