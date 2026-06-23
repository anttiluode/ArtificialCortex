"""
live_agreement_cortex.py
========================

LIVE version of the foveated agreement-layer loop, same math as the verified
offline demo (saccade_agreement_demo.py). Runs on a webcam.

  prior / gist (theta)  : a blurry held prediction of the frame
                          - default: Gaussian blur of the frame (runs TODAY, no model)
                          - optional: your trained SplatVAE decode (--model ...)
  retina (gamma)        : the live frame, sharp
  coherence gate        : gain = 0.5*(1+cos d_phi) from local Gabor phase
  agreement (transient) : gated foveal detail in a DECAYING buffer (not stored)
  saccades              : eye -> argmax(surprise) in a window, inhibition of return

Honest status: the MATH here is the same numpy that was measured offline. The
loop itself I could not run in the sandbox (no webcam, no GPU, no model), so:
  - it runs live on a blur-prior with ZERO model (verify the mechanism first), and
  - the model hook is a small, clearly-marked adapter you confirm against your
    own SplatVAE class -- I did not have your exact model API in this session,
    so do NOT trust the --model path until the adapter below matches your code.

Run:
  python live_agreement_cortex.py                 # webcam, blur prior, runs now
  python live_agreement_cortex.py --image face.jpg  # still image, no webcam
  python live_agreement_cortex.py --smoke         # synthetic, no webcam/model
  python live_agreement_cortex.py --model runs/splat/model.pt --num_packets 512
Keys: q quit.  Knobs: --fixations 4 --decay 0.5 --fovea 16
"""
import argparse, sys
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

S = 256  # working resolution for the gate (keeps it real-time on CPU)


# ----- gabor / gate : IDENTICAL math to the offline verifier -----
def gabor_kernel(theta, lam, sigma, ksize=25):
    r = ksize // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)
    k = np.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2)) * np.exp(1j * 2 * np.pi * xr / lam)
    return k - k.mean()


_KERNELS = [gabor_kernel(np.pi * o / 4, 8.0, 6.0) for o in range(4)]


def analytic(img):
    return np.stack([fftconvolve(img, k.real, "same") + 1j * fftconvolve(img, k.imag, "same")
                     for k in _KERNELS], 0)


def gate_and_surprise(gist, zw):
    zg = analytic(gist)
    ph = zw * np.conj(zg)
    amp = np.abs(zw) * np.abs(zg)
    cos_dphi = np.real(ph) / (amp + 1e-9)
    w = amp / (amp.sum(0, keepdims=True) + 1e-9)
    gain = np.clip((w * 0.5 * (1 + cos_dphi)).sum(0), 0, 1)
    return gain, 1.0 - gain


def aperture(cy, cx, s):
    yy, xx = np.mgrid[0:S, 0:S]
    return np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2)))


# ----- prior sources -----
def blur_gist(gray):
    return gaussian_filter(gray, 5.0)


class ModelGist:
    """Adapter to your trained SplatVAE. CHECK these calls against splat_generator.py
    before trusting --model. I did not have your exact class API in this session."""
    def __init__(self, path, num_packets):
        import torch, importlib.util, os
        self.torch = torch
        spec = importlib.util.spec_from_file_location("splat_generator", "splat_generator.py")
        sg = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        # --- ADAPTER: match the next 3 lines to your SplatVAE constructor + forward ---
        self.model = sg.SplatVAE(num_packets=num_packets).to(self.dev)   # <-- ctor name/args?
        self.model.load_state_dict(torch.load(path, map_location=self.dev))
        self.model.eval()

    def __call__(self, gray):
        t = self.torch
        x = t.from_numpy(gray).float()[None, None].repeat(1, 3, 1, 1).to(self.dev)
        with t.no_grad():
            out = self.model(x)                       # <-- returns recon? (recon, mu, logvar)?
            recon = out[0] if isinstance(out, (tuple, list)) else out
        g = recon[0].mean(0).clamp(0, 1).cpu().numpy()  # to gray
        return g


# ----- the loop, run once per frame, state persists across frames -----
class AgreementLoop:
    def __init__(self, gist_fn, fixations=4, decay=0.5, fovea=16.0, ior_s=26.0):
        self.gist_fn = gist_fn
        self.fix, self.decay, self.fovea, self.ior_s = fixations, decay, fovea, ior_s
        self.agree = np.zeros((S, S)); self.ior = np.zeros((S, S))
        self.path = []

    def step(self, gray):
        gist = self.gist_fn(gray)
        zw = analytic(gray)
        gain, surp = gate_and_surprise(gist, zw)
        residual = gray - gist
        self.path = []
        yy, xx = np.mgrid[0:S, 0:S]
        self.ior *= 0.92
        for _ in range(self.fix):
            sal = gaussian_filter(surp, 4) * (1 - np.clip(self.ior, 0, 1))
            sal[:18] = sal[-18:] = 0; sal[:, :18] = sal[:, -18:] = 0   # mask FFT borders
            cy, cx = np.unravel_index(np.argmax(sal), sal.shape)
            self.path.append((cy, cx))
            ap = aperture(cy, cx, self.fovea)
            self.agree = self.decay * self.agree + ap * gain * residual
            self.ior += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * self.ior_s ** 2)))
        percept = np.clip(gist + self.agree, 0, 1)
        return gist, percept, surp


def to_gray256(frame_bgr):
    import cv2
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    return cv2.resize(g, (S, S))


def panel(gray, gist, percept, surp, path):
    import cv2
    def u8(a): return (np.clip(a, 0, 1) * 255).astype(np.uint8)
    sm = surp / (surp.max() + 1e-9)
    sm = cv2.applyColorMap(u8(sm), cv2.COLORMAP_MAGMA)
    p = cv2.cvtColor(u8(percept), cv2.COLOR_GRAY2BGR)
    for (cy, cx) in path:
        cv2.circle(p, (cx, cy), 4, (235, 183, 133), 1)
    row = np.hstack([cv2.cvtColor(u8(gray), cv2.COLOR_GRAY2BGR),
                     cv2.cvtColor(u8(gist), cv2.COLOR_GRAY2BGR), p, sm])
    labels = ["retina (gamma)", "gist (theta)", "percept = gist + agreement", "surprise -> saccade"]
    for i, t in enumerate(labels):
        cv2.putText(row, t, (i * S + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 230, 160), 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--num_packets", type=int, default=512)
    ap.add_argument("--image"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--fixations", type=int, default=4)
    ap.add_argument("--decay", type=float, default=0.5); ap.add_argument("--fovea", type=float, default=16.0)
    a = ap.parse_args()

    gist_fn = blur_gist
    if a.model:
        print("[model] loading via adapter -- verify ModelGist matches your SplatVAE")
        gist_fn = ModelGist(a.model, a.num_packets)

    loop = AgreementLoop(gist_fn, a.fixations, a.decay, a.fovea)

    if a.smoke:
        g = np.zeros((S, S)); g[60:180, 60:180] = 0.8
        g = gaussian_filter(g, 2) + 0.05 * np.random.randn(S, S)
        gist, percept, surp = loop.step(np.clip(g, 0, 1))
        print(f"[smoke] shapes ok: gist{gist.shape} percept{percept.shape} surp{surp.shape}")
        print(f"[smoke] fixations: {loop.path}"); return

    import cv2
    if a.image:
        g = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        g = cv2.resize(g, (S, S))
        gist, percept, surp = loop.step(g)
        cv2.imwrite("live_agreement_still.png", panel(g, gist, percept, surp, loop.path))
        print("wrote live_agreement_still.png"); return

    cap = cv2.VideoCapture(a.cam)
    if not cap.isOpened():
        print("no webcam; try --image PATH or --smoke", file=sys.stderr); return
    print("running -- press q to quit")
    while True:
        ok, frame = cap.read()
        if not ok: break
        g = to_gray256(frame)
        gist, percept, surp = loop.step(g)
        cv2.imshow("agreement cortex", panel(g, gist, percept, surp, loop.path))
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
