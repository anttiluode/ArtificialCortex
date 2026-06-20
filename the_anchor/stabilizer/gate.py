"""
gate.py — the two-layer temporal anchor (slow hold + surprise gate), pure numpy
================================================================================
This is the load-bearing primitive of the whole app, and it runs on a CPU with no
torch. It is `the_video_tensor`'s core, repurposed from "measure surprise" to
"suppress the predictable so only surprise survives."

PER FRAME:
  SLOW layer  — a held prediction of the scene:  pred <- (1-K)*pred + K*frame
                (optionally motion-compensated by optical flow first, so the hold
                 survives camera/subject motion instead of smearing — this is the
                 cheap stand-in for the HKT Koopman predictor).
  FAST layer  — the residual:  surprise = |frame - pred|  (blurred into regions,
                 not speckle).
  GATE        — g in [0,1], soft-thresholded on surprise:
                  g ~ 0 where the world matched the hold  -> LOCK to prediction
                  g ~ 1 where the world broke the hold     -> PASS the new frame
  OUTPUT      — out = g*frame + (1-g)*pred
                so static texture is frozen to its held value (no boil) and moving
                content passes through untouched.

WHAT THIS HONESTLY DOES: it removes temporal variation in regions the hold could
predict, and leaves regions it could not. On a finished AI clip that is exactly
the boil (zero-mean high-frequency flicker in regions that should be static); the
leaky hold averages it out, the gate keeps the moving subject. It enforces
*temporal consistency*, not *truth* — it will hold a wrong-but-stable region just
as happily as a right one. "Boring", not "correct".

FAILURE MODES (exposed as knobs, not hidden): lock too hard and a region that
*starts* to move trails its held value (ghosting) until surprise clears the gate;
without flow, a moving camera makes everything surprising and the gate opens
everywhere (no saving); the gate blur trades seam artifacts against responsiveness.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def _gray(x):
    """HxWx3 float[0,1] -> HxW luminance."""
    if x.ndim == 2:
        return x
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114


def _warp(img, flow):
    """warp img by dense optical flow (cv2 remap). flow: HxWx2 (dx,dy)."""
    h, w = img.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (gx + flow[..., 0]).astype(np.float32)
    map_y = (gy + flow[..., 1]).astype(np.float32)
    return cv2.remap(img.astype(np.float32), map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


class TemporalAnchor:
    """Slow held prediction + surprise gate. Call .step(frame) per frame."""

    def __init__(self, K=0.03, surprise_thr=0.08, gate_softness=0.02,
                 surprise_blur=3.0, gate_blur=5.0, use_flow=True,
                 lock_strength=1.0):
        self.K = float(K)                       # slow-hold rate; lower = more boil killed, more lag
        self.surprise_thr = float(surprise_thr) # gate midpoint on |frame-pred|
        self.gate_softness = float(gate_softness)
        self.surprise_blur = float(surprise_blur)
        self.gate_blur = float(gate_blur)
        self.use_flow = bool(use_flow) and _HAS_CV2
        self.lock_strength = float(lock_strength)  # 1.0 = fully freeze locked regions
        self.pred = None
        self.prev_gray = None

    def reset(self):
        self.pred = None
        self.prev_gray = None

    def step(self, frame):
        """frame: HxWx3 float in [0,1]. Returns (out, gate, energy_pct)."""
        frame = frame.astype(np.float32)
        cur_gray = _gray(frame)

        # first frame: nothing held yet, pass through and seed the hold
        if self.pred is None or self.pred.shape != frame.shape:
            self.pred = frame.copy()
            self.prev_gray = cur_gray
            return frame.copy(), np.ones(frame.shape[:2], np.float32), 100.0

        # motion-compensate the held prediction so it survives real motion
        pred = self.pred
        if self.use_flow:
            flow = cv2.calcOpticalFlowFarneback(
                (self.prev_gray * 255).astype(np.uint8),
                (cur_gray * 255).astype(np.uint8),
                None, 0.5, 3, 21, 3, 5, 1.2, 0)
            pred = _warp(self.pred, flow)

        # surprise = residual the hold did not foresee, regionised
        surprise = np.abs(cur_gray - _gray(pred))
        if self.surprise_blur > 0:
            surprise = gaussian_filter(surprise, self.surprise_blur)

        # soft gate: 0 -> lock to prediction, 1 -> pass the new frame
        g = 1.0 / (1.0 + np.exp(-(surprise - self.surprise_thr) / self.gate_softness))
        if self.gate_blur > 0:
            g = gaussian_filter(g, self.gate_blur)
        g = np.clip(g, 0.0, 1.0)
        g3 = g[..., None]

        # locked regions take the held prediction; surprise regions take the frame
        held = self.lock_strength * pred + (1.0 - self.lock_strength) * frame
        out = g3 * frame + (1.0 - g3) * held

        # leaky-integrate the hold toward the input (low K averages out the boil)
        self.pred = (1.0 - self.K) * pred + self.K * frame
        self.prev_gray = cur_gray

        # honest "energy": fraction of pixels the gate let through (what a sparse
        # system would actually recompute) vs a dense net's 100%
        energy_pct = 100.0 * float((g > 0.5).mean())
        return np.clip(out, 0.0, 1.0), g, energy_pct
