# Temporal cortex — predicting the next frame (the honest version)

Answering "do we need a new model maker + live code?": **the live code, yes — and
it doesn't even need a new model. The model maker is optional and a bet.** Here's
why, measured rather than asserted.

## What the probe found (koopman_predict_probe.py — run, measured)
On a moving face video, predicting the next frame:
- free linear Koopman operator `z_{t+1}=K z_t` : **8.6 dB**
- persistence (copy the current frame)         : **22.5 dB**  ← the baseline to beat
- best identity-anchored operator the search found : **= pure identity (persistence)**

So Gemini's core move — a free linear K that "aligns the prior" — predicts the
next frame *worse* than doing nothing, and no linear operator beat persistence.
Image motion is not linear in a latent of pixels, so a linear operator can't
represent it. **The strongest next-frame predictor is simply the previous frame.**

The real payoff of going temporal is different and real: when the prior is a
prediction from the past, **surprise becomes motion** — sparse and localized —
instead of the static gate's "every edge is surprise" (the failure in your
screenshot). That fix needs no trained operator at all.

## Files
- `koopman_predict_probe.py` + `koopman_predict.png` — the measured result above.
- `live_temporal_cortex.py` — **runs today, no training, no model.** Prior = the
  previous frame (persistence). Surprise = motion. The eye chases what moved; the
  percept stays sharp where the world is static. This is the fix to the static
  gate, available now. Optional `--model` / `--koopman` to use the VAE prior.
- `splat_predictor.py` — the trainer you asked for, built honestly: the Koopman
  operator is INITIALISED TO IDENTITY (starts as persistence) and the log prints
  the persistence baseline every epoch. If `pred` never drops below `persistence`,
  the operator isn't earning its keep — which the probe says is the likely outcome.
  Reuses your real `splat_generator` Encoder/Decoder/GaborRenderer.

## Run order
1. `python live_temporal_cortex.py`  — see surprise become motion, eye chase it. No model.
2. Only if you want a learned predictor:
   - record ~1 min of webcam motion to `me.mp4`
   - `python splat_predictor.py --smoke`  (CPU shape check first)
   - `python splat_predictor.py --video me.mp4 --image_size 128 --num_packets 512 --amp`
   - watch `pred` vs `persistence` in the log.

## Honest ceiling
The probe is a software measurement on a moving sequence, not a brain claim.
A model trained on one short video is a per-video proof, not a world predictor.
Persistence is a strong baseline; the learned operator is a bet the probe says
to temper. The Koopman *modes* (eigenvalues of K) are real and are where your
HKT intuition lives — but a linear operator only finds them if the latent is a
coordinate system where motion is linear, which a pixel-VAE latent is not. That
"find the right observables" step is the hard, unsolved part — not free.
