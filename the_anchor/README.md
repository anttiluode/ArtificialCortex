# The Anchor

### Making AI video boring on purpose — a slow held prediction that freezes what the model kept re-rolling, so only the genuinely-moving part is allowed to change

**PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.**

> Do not hype. Do not lie. Just show.

---

## What this is

`the_video_tensor` measured surprise on a webcam. **The Anchor** turns that meter into an actuator: it *holds* the regions a video model could predict, and only lets the surprising regions through. Pointed at the "temporal boiling" of AI video — the fever-dream morphing where a brick wall shimmers into cobblestone frame to frame — it freezes the wall and spends only on the hand that actually moved.

The primitive is the same one as everywhere in the line: a **slow layer holds time** (a leaky, optionally motion-compensated prediction of the scene — the buffer's `K`, the *anchor*) and a **fast layer fires on coincidence** (the residual `surprise = |world − prediction|`). The gate is one line:

```
out = g · frame + (1 − g) · prediction       g = soft_gate(surprise)
```

`g ≈ 0` where the world matched the hold → the pixel is locked to its prediction and cannot boil. `g ≈ 1` where the world broke the hold → the new frame passes untouched.

**Say it plainly, because this is where overclaiming would do damage:** this enforces *temporal consistency*, not *truth*. It will hold a wrong-but-stable region exactly as happily as a right one — a six-fingered hand becomes a **stable** six-fingered hand. "Boring" means consistent, not correct. It does not cure hallucination; it stops the *flicker*.

---

## Two ways to run it

**[1] `run_filter.py` — de-boil any video you already have. CPU, no torch, no model.**
The honest workhorse. Point it at a boiling clip (Sora/Runway/SVD/AnimateDiff output) and it stabilises the decoded frames directly — model-agnostic, instant, runs on a laptop.

```bash
pip install numpy scipy opencv-python
python run_filter.py boiling_clip.mp4 -o anchored.mp4 --side-by-side
```

**[2] `run_svd.py` — the anchor inside the denoising loop of Stable Video Diffusion. Needs a GPU.**
The generation-time version: at the end of each denoising step it locks low-cross-frame-variation latent tokens toward their temporal mean (static → held) and leaves high-variation tokens alone (moving → free). Written against the diffusers `callback_on_step_end` API; **this one is yours to run** — I cannot execute torch here, so its shape logic is checked but it is not run end-to-end on my machine.

```bash
pip install torch diffusers transformers accelerate pillow
python run_svd.py --image first_frame.png -o out.mp4 --strength 0.6 --baseline
```

**`test_anchor.py` — the proof, headless, no dependencies beyond numpy/scipy.**
Builds a synthetic clip with the disease and measures the cure.

```bash
python test_anchor.py
```

---

## The honest ledger

**Verified in code on this machine (synthetic boiling clip, `test_anchor.py`, seeded):**
- background temporal std (the boil) **0.0547 → 0.0145**, a **74% suppression** in the static region, against a clean floor of 0.0020;
- the moving subject is **not degraded**: its region error is 0.0403 anchored vs 0.0423 boiled — the gate passes it through;
- the gate let through **~6% of the frame** on average; a dense per-frame model pays 100% on every frame. That gap is the compute a surprise-gated front end would skip.
- `run_filter.py` runs end-to-end on a real `.mp4` (cv2 I/O verified), reporting the same ~6% moving fraction.

`thr ≈ 0.09` is the measured sweet spot here — maximum freeze *before* a slow subject starts to trail; past it, the subject error rises. **The surprise threshold is the one knob that matters; it must be tuned per clip** because it is set relative to the boil amplitude.

**Not run here, stated plainly:** `run_svd.py` was **not executed** (no GPU/weights in this environment). The latent gate is written to the confirmed callback API and checked for shape logic; the wall-clock behaviour on a real SVD render is for you to confirm.

**Borrowed, not invented (the known neighbourhood — same as RatSLAM, we walked into it):** this is the temporal-consistency family — TokenFlow, Rerender-A-Video, FRESCO, Text2Video-Zero's cross-frame attention, and the latent-caching line (DeepCache). The contribution is only the *framing*: a surprise-gated hold, the line's one primitive, pointed at the denoiser. Predictive/residual coding (Rao & Ballard 1999; the ELL negative image) is the lineage of the gate itself.

**Honest limits, not hidden:**
- consistency ≠ correctness (the load-bearing caveat, repeated because it matters);
- lock too hard and a region that *starts* to move trails its held value (ghosting) until surprise clears the gate; the `--strength` / `--thr` knobs are exactly this tradeoff;
- the leaky-`K` hold is a cheap stand-in for the HKT Koopman/U-Net predictor; optical flow (`--flow`) is the next rung and helps **only when the camera moves** — on a locked-off shot it adds its own warp noise and does slightly *worse* (measured: 66% vs 74%), so it is off by default;
- "energy %" is the active-pixel fraction, a proxy for what a sparse system would recompute, not Joules;
- single synthetic seed for the headline numbers; real footage will need its threshold tuned.

**The bet (untouched, as everywhere):** that the held world is a *felt* steadiness rather than a computed one. The Anchor locates the mechanism; it does not touch the hard problem.

---

## Lineage

The live de-boiling instance of [`the_video_tensor`](../the_video_tensor) — the slow buffer of the HKT line used as an anchor (Antti's word: "the Slow Layer holds the trajectory; Elvis is Elvis through time"), the surprise meter turned into a gate. Framing by Antti Luode; built with Claude (Opus 4.8). MIT.

*Hold what the model already decided, and stop paying to re-decide it. Spend only on the part of the frame that genuinely moved. The wall stops boiling — not because the model learned the wall, but because the math refused to re-roll it. Do not hype. Do not lie. Just show.*
