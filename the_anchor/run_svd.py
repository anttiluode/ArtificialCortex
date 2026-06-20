"""
run_svd.py — the anchor INSIDE the denoising loop of Stable Video Diffusion
===========================================================================
This is the generation-time version of the idea, and the one you must run (it
needs torch + a GPU + the SVD weights; I cannot execute torch here, so this is
written against the confirmed diffusers callback API and verified only for shape
logic, not run end-to-end on this machine — that part is yours).

WHAT IT DOES, HONESTLY. Stable Video Diffusion generates all frames jointly; its
latent during sampling has shape [B, F, C, h, w] (F = the frame axis). "Boiling"
is the model re-rolling texture independently across F in regions that should be
static. At the end of each denoising step we:
   1. measure, per spatial token, how much it VARIES across the frame axis;
   2. where that variation is LOW (a token that should be static), pull every
      frame's value toward the token's temporal mean (lock it — it stops boiling);
   3. where it is HIGH (a token that is genuinely moving), leave it alone.
The pull strength ramps up over the denoising schedule (weak while structure is
still forming, firm once it has), so we consolidate consistency rather than
freezing noise early.

This is the latent-space twin of `run_filter.py`'s pixel gate: same slow-holds /
fast-spends primitive, applied to the model's own internal state mid-generation.

WHAT IT IS NOT. It enforces cross-frame *consistency*, not *correctness*: a stable
wrong texture stays stably wrong. It is in the same family as TokenFlow,
Rerender-A-Video, FRESCO, and Text2Video-Zero's cross-frame attention — the known
neighbourhood of temporal-consistency methods (you'll recognise the pattern). The
contribution here is just the framing: a surprise-gated hold, the same primitive
as everywhere in the line, pointed at the denoiser.

Run (img2vid — SVD anchors to a real first frame, which makes the hold honest):
    pip install -r requirements.txt
    python run_svd.py --image first_frame.png -o out.mp4 --strength 0.6

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Do not hype. Do not lie. Just show.
"""
import argparse


def make_anchor_callback(total_steps, strength=0.6, var_thr=0.6,
                         softness=0.25, start_ratio=0.25):
    """Returns a diffusers callback_on_step_end that locks low-variation tokens
    across the frame axis toward their temporal mean. Pure torch; shapes assume
    SVD latents [B, F, C, h, w]."""
    import torch

    def callback(pipe, step, timestep, callback_kwargs):
        lat = callback_kwargs["latents"]            # [B, F, C, h, w]
        if lat.dim() != 5 or lat.shape[1] < 2:
            return callback_kwargs                  # not a frame-stacked latent; do nothing

        progress = step / max(total_steps - 1, 1)
        if progress < start_ratio:
            return callback_kwargs                  # let structure form before consolidating

        # ramp the hold from 0 at start_ratio to `strength` at the end
        ramp = (progress - start_ratio) / max(1.0 - start_ratio, 1e-6)
        s = strength * float(min(max(ramp, 0.0), 1.0))

        tmean = lat.mean(dim=1, keepdim=True)        # [B,1,C,h,w] temporal mean per token
        # per-token variation across frames, pooled over channels -> [B,1,1,h,w]
        var = ((lat - tmean) ** 2).mean(dim=(1, 2), keepdim=True).sqrt()
        # normalise variation to a 0..1 scale by its own spatial median (scale-free)
        med = var.median()
        vnorm = var / (med + 1e-6)

        # gate: 1 where moving (leave), 0 where static (lock toward temporal mean)
        g = torch.sigmoid((vnorm - var_thr) / softness)   # [B,1,1,h,w]
        locked = (1.0 - s) * lat + s * tmean              # broadcast mean over F
        lat = g * lat + (1.0 - g) * locked

        callback_kwargs["latents"] = lat
        return callback_kwargs

    return callback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="first/conditioning frame (img2vid)")
    ap.add_argument("-o", "--output", default="svd_anchored.mp4")
    ap.add_argument("--model", default="stabilityai/stable-video-diffusion-img2vid-xt")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--fps", type=int, default=7)
    ap.add_argument("--strength", type=float, default=0.6,
                    help="how hard to lock static tokens (0=off, ~0.8=very stable)")
    ap.add_argument("--var-thr", type=float, default=0.6,
                    help="below this normalised cross-frame variation, a token is locked")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", action="store_true",
                    help="also render with the anchor OFF, for an A/B")
    args = ap.parse_args()

    import torch
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import load_image, export_to_video

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    image = load_image(args.image).resize((1024, 576))

    def render(use_anchor):
        gen = torch.manual_seed(args.seed)
        cb = make_anchor_callback(args.steps, strength=args.strength,
                                  var_thr=args.var_thr) if use_anchor else None
        result = pipe(
            image, num_frames=args.frames, num_inference_steps=args.steps,
            fps=args.fps, generator=gen,
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=["latents"] if use_anchor else None,
        )
        return result.frames[0]

    if args.baseline:
        export_to_video(render(False), args.output.replace(".mp4", "_baseline.mp4"), fps=args.fps)
        print("wrote baseline (anchor OFF)")
    export_to_video(render(True), args.output, fps=args.fps)
    print(f"wrote {args.output} (anchor ON, strength={args.strength})")
    print("compare the two: the anchored background should stop boiling while the "
          "moving subject is untouched.")


if __name__ == "__main__":
    main()
