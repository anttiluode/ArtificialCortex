"""
run_filter.py — de-boil ANY existing video (no torch, no GPU, no model weights)
===============================================================================
This is the one you can run today, on a clip you already have. Point it at a
boiling AI-generated video (Sora/Runway/SVD/AnimateDiff output) and it applies the
TemporalAnchor: it holds the regions the model kept re-rolling and only lets the
genuinely-moving parts through, killing the background boil.

It does NOT need the generator. It is a post-hoc temporal stabiliser — the slow
hold + surprise gate run on the decoded RGB frames. That makes it model-agnostic
and instant, and it is the honest proof that the economics work on real pixels.

Usage:
    python run_filter.py input.mp4 -o output.mp4 --side-by-side
    python run_filter.py input.mp4 --thr 0.10 --K 0.03 --flow

Knobs that matter:
    --thr   surprise threshold: THE knob. Raise it to freeze more (more stable,
            but a slowly-moving subject can start to trail). Lower to be safe.
    --K     slow-hold rate: lower = more boil averaged out, more lag on reveals.
    --flow  motion-compensate the hold (use when the CAMERA moves; off for a
            locked-off shot, where it only adds warp noise).

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Do not hype. Do not lie. Just show.
"""
import argparse
import numpy as np

try:
    import cv2
except Exception:
    raise SystemExit("This needs opencv-python:  pip install opencv-python")

from stabilizer.gate import TemporalAnchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--K", type=float, default=0.03)
    ap.add_argument("--thr", type=float, default=0.08)
    ap.add_argument("--softness", type=float, default=0.02)
    ap.add_argument("--surprise-blur", type=float, default=3.0)
    ap.add_argument("--gate-blur", type=float, default=5.0)
    ap.add_argument("--flow", action="store_true", help="motion-compensate the hold")
    ap.add_argument("--side-by-side", action="store_true",
                    help="write input | output | gate triptych")
    args = ap.parse_args()

    out_path = args.output or args.input.rsplit(".", 1)[0] + "_anchored.mp4"

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = W * 3 if args.side_by_side else W

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, H))

    anchor = TemporalAnchor(K=args.K, surprise_thr=args.thr,
                            gate_softness=args.softness,
                            surprise_blur=args.surprise_blur,
                            gate_blur=args.gate_blur, use_flow=args.flow)

    energies = []
    n = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out, gate, energy = anchor.step(rgb)
        energies.append(energy)
        out_bgr = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        if args.side_by_side:
            gate_vis = cv2.applyColorMap((gate * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            triptych = np.concatenate([frame_bgr, out_bgr, gate_vis], axis=1)
            writer.write(triptych)
        else:
            writer.write(out_bgr)
        n += 1
        if n % 30 == 0:
            print(f"  frame {n:5d}   mean energy so far {np.mean(energies[1:]):.1f}%")

    cap.release()
    writer.release()
    meane = float(np.mean(energies[1:])) if len(energies) > 1 else 100.0
    print("=" * 64)
    print(f"wrote {out_path}  ({n} frames)")
    print(f"mean energy let through (the moving fraction): {meane:.1f}%")
    print(f"a dense per-frame model would pay 100% on every one of {n} frames.")
    print("the gap is the compute a surprise-gated front-end would skip.")
    print("=" * 64)


if __name__ == "__main__":
    main()
