"""
live_temporal_cortex.py
=======================
The temporal agreement loop -- the honest version, after the probe
(koopman_predict_probe.py) showed:
  * a free linear Koopman operator predicts the next frame WORSE than persistence;
  * the strongest next-frame predictor is simply the PREVIOUS frame;
  * the real payoff of going temporal is that SURPRISE becomes MOTION
    (sparse, localized) instead of "every edge" (the static gate's failure).

So the prior here is a PREDICTION FROM THE PAST, default = the previous frame
(persistence). Because that prior is sharp (not a blurry VAE gist), the percept
stays sharp where the world is static, and the fovea is drawn to where the
prediction failed = where things moved. This runs TODAY, no training, no model.

  prediction (prior) : previous frame  (persistence -- verified strongest baseline)
  retina             : current frame
  gate               : gain = 0.5*(1+cos d_phi)  -> agree where static, surprise where moved
  saccades           : eye -> argmax(temporal surprise) = motion, inhibition of return
  percept            : the held prediction, refreshed foveally where it was wrong

Optional --model uses your trained SplatVAE as the prior instead of persistence
(semantic completion, at the cost of identity-collapse -- see README). The
optional --koopman flag applies a trained operator; the probe says don't expect
it to beat persistence.

Run:
  python live_temporal_cortex.py                 # webcam, persistence prior, runs now
  python live_temporal_cortex.py --image_seq d/  # folder of ordered frames, no webcam
  python live_temporal_cortex.py --smoke
  python live_temporal_cortex.py --model runs/splat/model.pt --num_packets 512 --image_size 128
Keys: q quit.  Knobs: --fixations 4 --decay 0.6 --fovea 16
"""
import argparse, sys, glob, os
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

S = 256

def gabor_kernel(theta, lam=8., sigma=6., ks=25):
    r = ks // 2; yy, xx = np.mgrid[-r:r+1, -r:r+1]
    xr = xx*np.cos(theta) + yy*np.sin(theta); yr = -xx*np.sin(theta) + yy*np.cos(theta)
    k = np.exp(-(xr**2 + yr**2)/(2*sigma**2)) * np.exp(1j*2*np.pi*xr/lam)
    return k - k.mean()
_K = [gabor_kernel(np.pi*o/4) for o in range(4)]
def analytic(im):
    return np.stack([fftconvolve(im, k.real, "same") + 1j*fftconvolve(im, k.imag, "same") for k in _K], 0)
def gate_and_surprise(prior, frame):
    zp, zf = analytic(prior), analytic(frame)
    ph = zf*np.conj(zp); amp = np.abs(zf)*np.abs(zp)
    cos = np.real(ph)/(amp+1e-9); w = amp/(amp.sum(0, keepdims=True)+1e-9)
    gain = np.clip((w*0.5*(1+cos)).sum(0), 0, 1)
    return gain, 1.0 - gain
def aperture(cy, cx, s):
    yy, xx = np.mgrid[0:S, 0:S]; return np.exp(-(((yy-cy)**2 + (xx-cx)**2)/(2*s**2)))


class PersistencePrior:
    """The verified-strongest predictor: the previous frame."""
    def __init__(self): self.prev = None
    def __call__(self, frame):
        p = self.prev if self.prev is not None else gaussian_filter(frame, 5)
        self.prev = frame
        return p


class ModelPrior:
    """Optional: SplatVAE prediction. Wired to the real splat_generator API.
    --koopman applies a trained operator; probe says it won't beat persistence."""
    def __init__(self, path, num_packets, image_size, koopman=None):
        import torch, importlib.util, cv2
        self.torch, self.cv2 = torch, cv2
        spec = importlib.util.spec_from_file_location("splat_generator", "splat_generator.py")
        sg = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.sz = image_size
        self.model = sg.SplatVAE(image_size, 128, num_packets).to(self.dev)
        self.model.load_state_dict(torch.load(path, map_location=self.dev), strict=False); self.model.eval()
        self.koop = None
        if koopman and os.path.exists(koopman):
            self.koop = torch.load(koopman, map_location=self.dev)  # (latent,latent)
    def __call__(self, frame):
        t, cv2 = self.torch, self.cv2
        x = cv2.resize(frame, (self.sz, self.sz))
        xt = t.from_numpy(x).float()[None, None].repeat(1, 3, 1, 1).to(self.dev)
        with t.no_grad():
            mu, lv = self.model.enc(xt)
            z = self.koop @ mu[0] if self.koop is not None else mu[0]
            g = self.model.ren(self.model.dec(z[None]))[0].mean(0).clamp(0, 1).cpu().numpy()
        return cv2.resize(g, (frame.shape[1], frame.shape[0]))


class TemporalLoop:
    def __init__(self, prior_fn, fixations=4, decay=0.6, fovea=16.0, ior_s=26.0):
        self.prior_fn, self.fix = prior_fn, fixations
        self.decay, self.fovea, self.ior_s = decay, fovea, ior_s
        self.agree = np.zeros((S, S)); self.ior = np.zeros((S, S))
        self.path = []
    def step(self, frame):
        prior = self.prior_fn(frame)                 # prediction from the past
        gain, surp = gate_and_surprise(prior, frame) # surprise = where prediction failed = motion
        residual = frame - prior
        self.path = []; yy, xx = np.mgrid[0:S, 0:S]; self.ior *= 0.9
        for _ in range(self.fix):
            sal = gaussian_filter(surp, 4) * (1 - np.clip(self.ior, 0, 1))
            sal[:18] = sal[-18:] = 0; sal[:, :18] = sal[:, -18:] = 0
            cy, cx = np.unravel_index(np.argmax(sal), sal.shape); self.path.append((cy, cx))
            ap = aperture(cy, cx, self.fovea)
            self.agree = self.decay*self.agree + ap*residual   # admit the CHANGE at the fovea
            self.ior += np.exp(-(((yy-cy)**2 + (xx-cx)**2)/(2*self.ior_s**2)))
        percept = np.clip(prior + self.agree, 0, 1)            # held prediction, refreshed where it moved
        return prior, percept, surp


def panel(frame, prior, percept, surp, path):
    import cv2
    u8 = lambda a: (np.clip(a, 0, 1)*255).astype(np.uint8)
    sm = cv2.applyColorMap(u8(surp/(surp.max()+1e-9)), cv2.COLORMAP_MAGMA)
    p = cv2.cvtColor(u8(percept), cv2.COLOR_GRAY2BGR)
    for cy, cx in path: cv2.circle(p, (cx, cy), 4, (235, 183, 133), 1)
    row = np.hstack([cv2.cvtColor(u8(frame), cv2.COLOR_GRAY2BGR),
                     cv2.cvtColor(u8(prior), cv2.COLOR_GRAY2BGR), p, sm])
    for i, tt in enumerate(["retina (now)", "prior (prediction)", "percept", "surprise = motion"]):
        cv2.putText(row, tt, (i*S+6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 230, 160), 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--koopman")
    ap.add_argument("--num_packets", type=int, default=512); ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--image_seq"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--fixations", type=int, default=4); ap.add_argument("--decay", type=float, default=0.6)
    ap.add_argument("--fovea", type=float, default=16.0)
    a = ap.parse_args()

    prior_fn = PersistencePrior()
    if a.model:
        print("[model] SplatVAE prior (semantic, with identity-collapse caveat)")
        prior_fn = ModelPrior(a.model, a.num_packets, a.image_size, a.koopman)
    loop = TemporalLoop(prior_fn, a.fixations, a.decay, a.fovea)

    if a.smoke:
        g0 = np.zeros((S, S)); g0[80:160, 60:140] = 0.8; g0 = gaussian_filter(g0, 2)
        loop.step(g0)                                  # establish prev
        g1 = np.roll(g0, 12, axis=1)                   # something moved 12px
        prior, percept, surp = loop.step(g1)
        print(f"[smoke] ok: prior{prior.shape} percept{percept.shape} surp{surp.shape}")
        print(f"[smoke] surprise concentrated where it moved: max at {np.unravel_index(np.argmax(gaussian_filter(surp,4)), surp.shape)}")
        print(f"[smoke] fixations: {loop.path}"); return

    import cv2
    if a.image_seq:
        paths = sorted(glob.glob(os.path.join(a.image_seq, "*")))
        if not paths: print("no frames in", a.image_seq, file=sys.stderr); return
        for k, pth in enumerate(paths):
            f = cv2.resize(cv2.cvtColor(cv2.imread(pth), cv2.COLOR_BGR2GRAY).astype(np.float64)/255., (S, S))
            prior, percept, surp = loop.step(f)
            cv2.imwrite(f"temporal_{k:03d}.png", panel(f, prior, percept, surp, loop.path))
        print(f"wrote temporal_*.png ({len(paths)} frames)"); return

    cap = cv2.VideoCapture(a.cam)
    if not cap.isOpened(): print("no webcam; try --image_seq DIR or --smoke", file=sys.stderr); return
    print("running -- press q to quit")
    while True:
        ok, frame = cap.read()
        if not ok: break
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)/255., (S, S))
        prior, percept, surp = loop.step(g)
        cv2.imshow("temporal cortex", panel(g, prior, percept, surp, loop.path))
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
