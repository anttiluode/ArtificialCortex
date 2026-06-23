"""
live_saccade_cortex.py — a foveated, saccade-driven perceiver
================================================================================
live_cortex_perception.py fit ALL packets to the WHOLE frame every step — uniform
resolution, and the floater storm, because every packet is free to chase any pixel.
That is not how you see. You hold a blurry gist everywhere and sharpen only a tiny
foveal window, then SACCADE the fovea to wherever the prediction is worst, and
STITCH a stable percept across fixations.

This builds that loop on your trained SplatVAE:

  1. PRIOR     encoder -> decoder -> packets -> a blurry gist over the whole frame
               (held; this is the periphery).
  2. SACCADE   find where |frame - percept| is largest (the surprise), put the fovea
               there, and refine ONLY the packets inside that foveal window against
               the live frame for a few gradient steps. Commit them into the belief.
  3. REPEAT    a handful of fixations per frame, with inhibition-of-return so the eye
               does not re-fixate the same spot. The belief accumulates sharp detail
               where the fovea visited; the rest stays gist.

So the periphery is the cheap held prior, the fovea is expensive and sharp, and the
percept is stitched across saccades — foveal vision, not a whole-frame re-fit. As a
bonus this tames the floaters: only foveal packets ever move, so they cannot orphan
themselves all over the frame.

VERIFIED vs NOT: the MECHANISM (gist held, error-driven fovea, foveal refinement,
inhibition-of-return, stitching) is verified offline on a real image in
`saccade_demo.py` — run that to see it work with no model and no GPU (+1.6 dB at
fixated regions, periphery held flat). THIS live file is written against the
SplatVAE interface but was NOT run here (no GPU/webcam/model in the sandbox). It is
heavier than the old loop (several fixations x several steps per frame), so start
with few fixations and raise them until your fps is tolerable.

This is foveated perception. It is NOT 3D, NOT recognition, NOT a deepfake — it is a
held gist sharpened foveally by reality, stitched across saccades. The eye that
moves, not the fake that fools.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import argparse, math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from splat_generator import SplatVAE


def to_cv2(t):
    img = t[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="runs/splat/model.pt")
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--num_packets", type=int, default=512)  # MATCH your .pt
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--fixations", type=int, default=5, help="saccades per frame")
    ap.add_argument("--steps", type=int, default=8, help="refine steps per fixation")
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--fovea", type=float, default=0.18, help="foveal radius (fraction of frame)")
    ap.add_argument("--cam", type=int, default=1)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading cortex on {dev} ...")
    model = SplatVAE(args.image_size, args.latent, args.num_packets, chunk=64).to(dev)
    model.load_state_dict(torch.load(args.model_path, map_location=dev))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    H = args.image_size
    ys, xs = torch.meshgrid(torch.linspace(0, 1, H, device=dev),
                            torch.linspace(0, 1, H, device=dev), indexing="ij")
    anchor = model.ren.anchor_logit  # (N,2)
    blur = torch.ones(1, 1, 9, 9, device=dev) / 81.0   # box blur for the error map
    r2 = args.fovea ** 2

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"cannot open camera {args.cam} (try --cam 0)"); return
    print("saccade cortex online. press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape; s = min(h, w)
        crop = frame[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
        rgb = cv2.cvtColor(cv2.resize(crop, (H, H)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).float().permute(2, 0, 1)[None].to(dev) / 255.0

        # 1. PRIOR: the held gist
        with torch.no_grad():
            mu, _ = model.enc(x)
            belief = model.dec(mu).detach()           # (1,N,K)
            prior_img = model.ren(belief)

        ior = torch.zeros(H, H, device=dev)
        fixations = []

        # 2. SACCADES
        for _ in range(args.fixations):
            with torch.no_grad():
                percept = model.ren(belief)
                resid = (x - percept).abs().mean(1, keepdim=True)          # (1,1,H,W)
                resid = F.conv2d(resid, blur, padding=4)[0, 0]
                err = resid * (1 - ior)
                fy, fx = np.unravel_index(int(err.argmax().item()), (H, H))
            fyn, fxn = fy / (H - 1), fx / (H - 1)
            fixations.append((fx, fy))

            win = torch.exp(-((xs - fxn) ** 2 + (ys - fyn) ** 2) / (2 * r2))   # (H,W)
            ior = torch.clamp(ior + torch.exp(-((xs - fxn) ** 2 + (ys - fyn) ** 2)
                                              / (2 * (0.9 * args.fovea) ** 2)), 0, 1)

            # packets inside the fovea (positions = activated anchors + offsets)
            with torch.no_grad():
                px = torch.sigmoid(anchor[:, 0] + belief[0, :, 0])
                py = torch.sigmoid(anchor[:, 1] + belief[0, :, 1])
                fov_idx = torch.nonzero(((px - fxn) ** 2 + (py - fyn) ** 2)
                                        < (1.5 * args.fovea) ** 2, as_tuple=False).flatten()
            if fov_idx.numel() == 0:
                continue

            held = belief.detach()
            fov_params = belief[0, fov_idx].clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([fov_params], lr=args.lr)
            Wm = win[None, None]
            for _ in range(args.steps):
                opt.zero_grad()
                full = held.index_copy(1, fov_idx, fov_params[None])   # functional, autograd-safe
                img2 = model.ren(full)
                loss = (Wm * (img2 - x) ** 2).sum() / (Wm.sum() * 3 + 1e-6)
                loss.backward(); opt.step()
            belief = held.index_copy(1, fov_idx, fov_params.detach()[None])

        with torch.no_grad():
            final = model.ren(belief)

        cv_real = to_cv2(x); cv_prior = to_cv2(prior_img); cv_fov = to_cv2(final)
        for (fx, fy) in fixations:                      # draw the scanpath on panel 3
            cv2.circle(cv_fov, (int(fx), int(fy)), 3, (0, 255, 255), -1)
        if len(fixations) > 1:
            pts = np.array(fixations, np.int32)
            cv2.polylines(cv_fov, [pts], False, (0, 200, 200), 1)
        for im, lab in ((cv_real, "1. retina"), (cv_prior, "2. prior (gist)"),
                        (cv_fov, "3. foveated (saccades)")):
            cv2.putText(im, lab, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        panel = cv2.resize(np.hstack([cv_real, cv_prior, cv_fov]),
                           (H * 6, H * 2), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Artificial Cortex - foveated saccades", panel)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
