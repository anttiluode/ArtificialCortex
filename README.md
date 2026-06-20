# The Artificial Cortex

### A trainless, dynamical substrate that computes in waves — four organs of one idea: units that hold time and fire on coincidence, coupled into a field

**PerceptionLab / Antti Luode, with Claude (Opus 4.8), in dialogue with Gemini. Helsinki, June 2026.**

> Do not hype. Do not lie. Just show.

---

## What this is

The whole line collapses to one unit: a cell whose **subthreshold membrane holds time** (the skew lag-operator, whose eigenplanes are directed rotation modes — the *where/when*) and which **fires on coincidence** (the Larkum/two-worlds vote — the *now* agreeing with the held trajectory). This repo asks what a **field** of those units does. The answer, built and measured in pure-numpy/scipy, is four organs of one substrate — three primitives and the block that integrates them:

| folder | the organ | what it does | the measured result |
|---|---|---|---|
| [`the_tissue/`](the_tissue/) | **the substrate** — a lattice of skew-units coupled by the ephaptic diffusion term (complex Ginzburg–Landau) | computes in **traveling waves** and phase-locked domains, not stacked weights | one number (Benjamin–Feir `1+bc`) flips a coherent wave (0.17% defects) into turbulence (2.22%) |
| [`the_grid/`](the_grid/) | **navigation** — three held trajectories meeting at 60° | makes a **hexagonal grid cell** by interference, path-integrated, no training | gridness **0.93** clean; collapses as the held phases drift |
| [`the_stack/`](the_stack/) | **the vertical wiring** — the slow band gates the fast band | **theta–gamma coupling**, the brain's robust cognition signature | modulation index **0.096** gated vs **0.000** ungated |
| [`the_tensor/`](the_tensor/) | **the integrator** — X,Y space × Z frequency; slow layer holds a grid prediction, fast layer encodes the residual | spends energy **only where the world contradicts the prediction**, gated by theta | spatial PAC MI **0.145** vs 0.005; gamma energy **≈0 at zero surprise**, ~2.8% of a static block's cost at 10% surprise |
| [`the_video_tensor/`](the_video_tensor/) | **the live instance** — the same field on a webcam, with inner/outer views and the cortical-layer stack | spends compute in proportion to surprise, **live**, going dark on a still room | core verified headless: **13×** gamma on motion vs background, PAC corr **0.996** (GUI runs on your machine) |

Five things stand on their own and all run on a laptop:

```bash
python the_tissue/spectral_field.py     # the wave substrate + the regime line
python the_grid/grid_cells.py           # grid cells from interference (gridness)
python the_stack/theta_gamma.py         # theta-gamma PAC (modulation index)
python the_tensor/cortical_tensor.py    # the 3D block: spatial PAC + energy-on-surprise
python the_video_tensor/video_tensor.py # the live webcam instance (synthetic fallback)
```

---

## The thread

Every organ is the same primitive — *hold time, fire on coincidence* — read at a different scale. One unit holding one island is a complex oscillator; **couple them** (the tissue) and you get waves. Let those units **hold the body's trajectory** and meet at 60° (the grid) and the coincidence is a hexagonal map. Give each unit its **full skew spectrum** so the field has several bands, and let the slow band **gate** the fast one (the stack), and you get theta–gamma coupling. Then **stack the bands into one block** (the tensor): the slow layer holds the grid prediction, the fast layer encodes only the residual, and the field spends nothing where the world matched what it expected. Substrate, map, clock, and the block that binds them are four readings of one cell.

---

## The honest ledger (this is the load-bearing part)

This repo is ambitious in **scope** and deliberately humble in **claims**. Each organ reproduces a measured *signature* of cortex using an *established* model — it does not derive cortex.

**Verified in code (reproducible, seeded):**
- a lattice coupled by the complex-diffusion (ephaptic) term forms coherent traveling/spiral waves, and a single number (`1+bc`) sets coherent-vs-turbulent — measured by defect density and wave cross-correlation (correlation length did *not* separate the regimes here; reported, not hidden);
- three velocity-controlled oscillators at 60° produce a path-integrated hexagonal grid (gridness 0.93), which **degrades as their held phases drift** — the oscillatory-interference model *and* its known fragility;
- a slow-gates-fast architecture produces theta–gamma PAC (MI 0.096) absent in the ungated control;
- stacked into one block (the tensor), the same gating gives **spatial** theta–gamma PAC (MI 0.145 vs 0.005 ungated), and a fast layer that encodes the residual against a held grid prediction spends **energy proportional to surprise** — near zero when the world matches the prediction, rising with mismatch — where a static/dense processor pays a constant cost regardless.

**Borrowed, not invented (the mechanisms are decades old):** complex Ginzburg–Landau / coupled oscillators and neural field theory (Kuramoto; Wilson–Cowan; Amari; Bressloff; Aranson & Kramer); the oscillatory-interference grid model (O'Keefe & Burgess; Burgess 2007), against which the continuous-attractor model competes (McNaughton et al.); the theta–gamma code (Lisman & Jensen 2013; Tort et al. 2010); cortical traveling waves (Muller et al. 2018). **The contribution is the synthesis** — wiring these into one substrate from the *hold-time-fire-on-coincidence* unit — not any of the mechanisms, and emphatically **not a first-principles derivation of the brain's operating system.**

**Tempering the enthusiasm, plainly (because the front door is where overclaiming does the most damage):**
- this is **not a transformer competitor** and nothing here suggests otherwise. Wave/oscillatory computation is a real but *niche* area; it does not approach transformers on scale or raw capability. The axes where it is interesting are different ones: energy spent only at coincidences, dynamics you can *read*, and a **characteristic, biological failure mode** (coherence loss → turbulence) instead of a silent wrong answer;
- "computes via resonance rather than matrix multiplication" is a fond phrase but it still runs on arrays; the honest version is that **the computation is carried by the field's dynamics — waves, interference, phase-locking — rather than by trained feedforward weights.** That is a real architectural difference, stated without the magic;
- "the grid cells emerge for free" — no. The 60° symmetry is put in by hand; the demo shows the interference *given* that choice, and shows it breaking under phase noise;
- nothing here is calibrated to a brain's, or a model's, measured numbers. Relative units throughout, single seeds, toy worlds.

**The empirical anchor is still elsewhere.** The strongest real result in the whole programme remains the trainless geometric-dysrhythmia EEG separation (p = 0.007); none of this repo depends on it, and none of this repo is offered as evidence for the brain. It is an *architecture* that produces the right *shapes*.

**The bet (untouched, as everywhere in the line):** that any of this is *experienced* rather than processed — that the grid is a felt place, the turbulence a felt disorientation, the bound theta–gamma moment a felt now. The repo locates the mechanisms precisely, in code that can fail. It does not touch the hard problem, and — as Antti put it — perhaps it always won't.

---

## The integrator, now built: the 3D tensor

The block the three primitives pointed at is built and measured in [`the_tensor/`](the_tensor/): `X × Y` physical space, `Z` the frequency spectrum. A slow (theta) layer sweeps the sheet as a deep pacemaker and **holds a grid-patterned prediction** (the *where*); a fast (gamma) layer encodes only the **residual** — what the world did that the grid did not foresee (the *now*) — gated by the local theta phase. The computation is a wave of coincidence: the fast layer lights up *only* where the sensory world contradicts the slow prediction, and only when theta opens the gate. Two things are measured, not asserted — spatial theta–gamma PAC (MI 0.145 vs 0.005), and **energy proportional to surprise** (near zero when the world matches the grid, ~2.8% of a static block's cost at 10% surprise). That energy law is the one genuine architectural difference from a dense weight-tower: it pays for novelty, not for everything.

**Still honest about what it is not:** theta is a *prescribed* pacemaker here, not self-organised; gamma is a damped-driven linear field; the "static processor" is a constant-cost stand-in, not a measured transformer; "energy" is `|z|²`, a proxy. The architecture spends on surprise; that is a property, not a benchmark.

## Where it goes next

1. **Self-organise the theta clock** — let the slow layer be its own CGLE field (the tissue) so the rhythm emerges rather than being imposed.
2. **Close the loop** — let the slow layer *learn* its grid prediction from the residual the fast layer reports (mirror-gate distillation), so the *where* is grown from experience, not handed in.
3. **Run it on the real HKT video stream** — slow layer predicts the next frame, fast layer encodes only the surprise, and the energy meter reads the actual compute saved. [`the_video_tensor/`](the_video_tensor/) is the first step there: the economics demo on a live webcam with a leaky predictor. The remaining step is to swap that leaky hold for the HKT Koopman/U-Net predictor, where the saving becomes a real wall-clock number against an actual heavy model — which is exactly where this joins the gated-diffusion / DeepCache direction.

---

## Lineage

Built on the geometric-neuron / Mycelial Cortex line and the HKT engine (PerceptionLab): the skew lag-operator (the held time), the chandelier/theta clock (the gate), and the placed unit (`where_am_i`). The framing — that a field of hold-time/fire-on-coincidence units is a computational tissue whose organs are waves, grids, and clocks — is Antti Luode's, worked out across a run of mornings after Juhannus and a dialogue with Gemini; the four demos, their measurements, and these documents were developed collaboratively with Claude (Opus 4.8). MIT.

*One cell holds time and fires when the world agrees. Couple a field of them and it makes waves; let them hold your trajectory and it makes a map; stack their rhythms and the slow one clocks the fast; bind them into a block and it spends nothing on what it already expected. An artificial cortex is not a taller tower of weights — it is a tissue that expects, in waves you can watch break. Do not hype. Do not lie. Just show.*
