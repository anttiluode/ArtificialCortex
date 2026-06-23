# Foveated agreement-layer loop — offline verifier + live version

Two persistent layers (gist/theta, retina/gamma) and one transient layer
(agreement), linked by a coherence gate. Sharpness is not stored; it is the
standing agreement between prediction and incoming detail at the fovea,
re-fetched each fixation. Surprise (gate mismatch) drives the next saccade.

`gate: gain = 0.5*(1 + cos d_phi), surprise = 1 - gain`, with `d_phi` the local
Gabor phase difference `angle(z_world * conj(z_gist))`, energy-weighted over
orientations — the same complex carry used since janus.

## Files
- `saccade_agreement_demo.py` — offline, CPU, no model/GPU/webcam. **Verified: I ran it.**
- `live_agreement_cortex.py` — same math, on a webcam. Runs now on a blur-prior;
  optional `--model` adapter for your SplatVAE (unrun by me — see caveat below).
- `saccade_agreement.png`, `saccade_novelty.png` — the measured figures.

## What the offline run measured (not asserted)

EXP 1 — core mechanism, against a blurry prior:
- held percept (transient): **+1.08 dB** over the gist globally — barely moves: not stored.
- accumulating stitch: **+2.59 dB** — climbs toward a stored sharp photo.
- at fixated patches the percept is locally sharp: **+2.00 dB** over the gist.
- surprise inside the changed region falls **0.479 → 0.275** as the slow prior
  update resolves it: the loop settles.

EXP 2 — surprise targeting, against a *settled* prior (the honest test):
- relative to a blurry prior every edge is surprise, so a change is not special.
  Only once the prior is in agreement with the scene does a new change stand out.
- change-box enrichment: intensity residual **4.82×**, phase gate **7.71×** —
  the phase gate concentrates harder here (I did not tune it to win; the first
  blurry-prior test had it *losing*, and the reason was the wrong regime).
- first 6 fixations: **4/6 land inside the change**.

## Honest ceiling
A hand-built mechanism behaving as designed on one contrived static scene.
Not evidence about brains, not qualia, not better than any existing model.
The "is the agreement experienced" question stays in the drawer.

## Live caveat
The live loop's math is the verified offline numpy. I could not run the loop
itself (no webcam/GPU/model in the sandbox). It runs today with a blur prior and
zero model. The `--model` path uses a small adapter (`ModelGist`) that I could
not check against your real SplatVAE API this session — confirm the three marked
lines match `splat_generator.py` before trusting it. Start with `--smoke`, then
`--image face.jpg`, then webcam, then `--model`.
