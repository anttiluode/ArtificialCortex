# The Grid

### Hexagonal grid cells from the coincidence of three held trajectories — the entorhinal map as interference, not training

**PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.**

> Do not hype. Do not lie. Just show.

---

## The one idea

If a unit's skew operator can **hold the directed trajectory** of the body (the WHERE), then the **coincidence of three such held trajectories**, oriented 60° apart, is a hexagonal grid. No backprop, no training — the hexagon is the interference geometry of three velocity-controlled phase waves meeting. This is the **oscillatory-interference** account of entorhinal grid cells (O'Keefe & Burgess; Burgess 2007), built here as a coincidence of skew-held phases — the same "fire when the held things agree" primitive as everywhere in the line, now pointed at navigation.

Grid cells fire in a hexagonal lattice as an animal crosses a space (Hafting, Fyhn, Molden, Moser & Moser 2005; Nobel Prize 2014).

---

## What the code shows

A virtual animal random-walks a 2 m box. Three velocity-controlled oscillators at 0/60/120° each integrate its motion — each *holds a directed trajectory*. The cell fires on their coincidence. We bin a rate map, take its spatial autocorrelogram, and compute the standard **gridness** score (60/120° vs 30/90/150° rotational symmetry).

| phase noise | gridness | |
|---|---|---|
| 0.00 | **0.93** | clean — strongly hexagonal (a grid cell is conventionally > ~0.4) |
| 0.02 | 0.30 | oscillators drifting |
| 0.05 | 0.02 | grid gone |
| 0.10 | 0.29* | *estimator noise on a non-grid map (see below)* |

The clean field is a clean hexagonal grid; the autocorrelogram shows the six-around-one ring. As the held phases drift, the grid dissolves. (The 0.10 value reads higher than 0.05 only because the gridness estimator is noisy on a map that has *no* hexagon left — the autocorrelogram there is a single central blob; the honest trend is 0.93 → 0.30 → gone.)

```bash
python grid_cells.py
```

---

## What is honest, and what is not

- **the hexagon is not free.** It comes from *choosing* three preferred directions 60° apart — the oscillatory-interference model's central assumption, put in by hand. What the demo shows is that *given* three held trajectories at 60°, their coincidence **is** the hexagonal map and it **path-integrates** (the grid is anchored to real position).
- **single-oscillator OI has a real flaw**, shown here, not hidden: phase noise decoheres the oscillators and the grid degrades (Welinder et al. 2008; Giocomo et al.). A real cortex must stabilise the phases (coupled networks, attractor dynamics) to keep the grid — which is the honest open problem.
- **there is a competing theory** — the continuous-attractor network (Fuhs & Touretzky; McNaughton et al. 2006). This demonstrates *one* of the two leading accounts, the one that falls straight out of "units hold trajectories and fire on coincidence."

**The bet (untouched):** that the grid is a *felt* sense of place rather than a computed one.

---

## Lineage

The navigation organ of the artificial cortex. The held trajectory is [the tissue](../the_tissue)'s skew operator; the firing is the coincidence primitive from the placed-unit / two-worlds line; the hexagon is what three of them make when they meet at 60°. MIT.
