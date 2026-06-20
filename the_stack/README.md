# The Stack

### The cortical stack's vertical wiring — the slow band gates the fast band, and theta–gamma coupling is the fingerprint

**PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.**

> Do not hype. Do not lie. Just show.

---

## The one idea

The cortex is not one sheet; it is a **stack of frequency bands** (the Z-axis: give each unit its full skew spectrum, several `ω`, and each band is its own wave). The question is how a slow band talks to a fast one. The answer the line keeps arriving at — the chandelier on a theta clock, from the HKT work — is that the **slow theta wave is the coincidence-gate for the fast gamma field**. Where the theta peak washes over a neighbourhood it disinhibits it, and the gamma coincidences are allowed to fire there.

The measurable fingerprint of that arrangement is **theta–gamma phase–amplitude coupling (PAC)**: gamma amplitude peaking at a preferred theta phase. PAC is the single most robust EEG signature of human memory and cognition (Canolty & Knight 2010; Tort et al. 2010; Lisman & Jensen 2013, the theta–gamma neural code).

---

## What the code shows

The gated arrangement is built and the coupling **measured** with Tort's Modulation Index, against a control with the gate removed:

| arm | Tort modulation index |
|---|---|
| **gated** (theta opens the gate) | **0.096** |
| control (no gate / constant) | 0.000 |

The gated stack shows a strong PAC where the ungated control shows none; gamma fires where the theta wave lets it. The figure shows gamma amplitude peaking near the theta peak, and gamma bursts riding the slow wave.

```bash
python theta_gamma.py
```

---

## Honest scope

This **reproduces** the PAC signature from a gating architecture; it does **not** prove cortex generates PAC this way (there are several candidate generators). The gate here is a clean cosine of theta phase; real gating is messier and noisier. One synthetic recording, relative units. What is claimed is exactly the table: a slow-gates-fast architecture produces the brain's most robust cognition signature, and removing the gate removes it.

**The bet (untouched):** that the coupling is *experienced* as a bound moment rather than computed as one.

---

## Lineage

The vertical wiring of the artificial cortex — the elevator between [the tissue](../the_tissue)'s bands. It is the HKT theta-gated chandelier, measured as PAC. MIT.
