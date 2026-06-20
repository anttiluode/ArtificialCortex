"""
test_anchor.py — verify the de-boiler headlessly on synthetic "boiling" video
=============================================================================
No torch, no webcam, no model weights. Builds a synthetic clip that has exactly
the disease: a STATIC textured background + a MOVING bright blob, with per-frame
high-frequency noise ("the boil") sprayed over the WHOLE frame. Then runs the
TemporalAnchor and measures two things, honestly:

  [1] BOIL KILLED. Per-pixel temporal std in the static background, averaged,
      BEFORE (boiled input) vs AFTER (anchored output). The anchor should crush it.
  [2] SUBJECT PRESERVED. Fidelity of the moving blob region in the AFTER clip
      against a CLEAN (un-boiled) reference. The anchor should keep it.

If [1] drops a lot while [2] stays high, the thesis holds: it removes the
predictable variation and spends only on the part that genuinely moved.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Do not hype. Do not lie. Just show.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stabilizer.gate import TemporalAnchor

H, W, T = 144, 192, 90
rng = np.random.default_rng(0)


def make_clips():
    # static textured background (fixed for all frames)
    yy, xx = np.mgrid[0:H, 0:W]
    bg = (0.45
          + 0.12*np.sin(xx/9.0) * np.cos(yy/13.0)
          + 0.06*np.sin((xx+yy)/5.0))
    bg = np.clip(bg, 0, 1).astype(np.float32)
    bg = np.repeat(bg[..., None], 3, axis=2)

    clean = np.empty((T, H, W, 3), np.float32)
    blob_mask = np.zeros((T, H, W), bool)
    for t in range(T):
        frame = bg.copy()
        cx = W*(0.5 + 0.34*np.sin(t*0.10))      # moving subject
        cy = H*(0.5 + 0.30*np.cos(t*0.08))
        d2 = (xx-cx)**2 + (yy-cy)**2
        blob = 0.75*np.exp(-d2/(2*11.0**2))
        frame = np.clip(frame + blob[..., None]*np.array([1.0, 0.4, 0.2]), 0, 1)
        clean[t] = frame
        blob_mask[t] = blob > 0.10

    # the boil: per-frame high-frequency zero-mean flicker over the whole frame
    boil = 0.055*rng.standard_normal((T, H, W, 1)).astype(np.float32)
    boiled = np.clip(clean + boil, 0, 1)

    static_mask = ~blob_mask.any(axis=0)        # pixels the blob never touches
    return clean, boiled, blob_mask, static_mask


def temporal_std(clip, mask2d):
    s = clip.std(axis=0).mean(axis=2)           # per-pixel temporal std, mean over RGB
    return float(s[mask2d].mean())


def region_fidelity(a, b, mask3d):
    """mean abs error inside a per-frame moving mask (lower = better preserved)."""
    errs = []
    for t in range(len(a)):
        m = mask3d[t]
        if m.sum() == 0:
            continue
        errs.append(np.abs(a[t][m] - b[t][m]).mean())
    return float(np.mean(errs))


if __name__ == "__main__":
    print("="*74)
    print("THE ANCHOR — making AI video boring: de-boil verification (synthetic)")
    print("="*74)
    clean, boiled, blob_mask, static_mask = make_clips()
    print(f"clip {T} frames @ {W}x{H}; boil = 0.055 zero-mean noise / frame, whole frame")
    print(f"static background = {100*static_mask.mean():.0f}% of pixels\n")

    for use_flow in (False, True):
        anchor = TemporalAnchor(use_flow=use_flow)  # shipped defaults
        out = np.empty_like(boiled)
        energies = []
        for t in range(T):
            o, g, e = anchor.step(boiled[t])
            out[t] = o
            energies.append(e)

        boil_in = temporal_std(boiled, static_mask)
        boil_out = temporal_std(out, static_mask)
        # also the untouched clean's residual std in static region = the floor
        floor = temporal_std(clean, static_mask)
        subj_in = region_fidelity(boiled, clean, blob_mask)   # boil's error on subject
        subj_out = region_fidelity(out, clean, blob_mask)     # anchor's error on subject

        tag = "WITH optical-flow hold" if use_flow else "leaky hold only (no flow)"
        print(f"[{tag}]")
        print(f"   background temporal std   boiled {boil_in:.4f} -> anchored {boil_out:.4f}"
              f"   (clean floor {floor:.4f})")
        print(f"   -> boil suppressed by {100*(1-boil_out/boil_in):.0f}% in the static background")
        print(f"   subject region mean-abs-err   boiled {subj_in:.4f} -> anchored {subj_out:.4f}"
              f"   (lower=better; subject preserved)")
        print(f"   mean energy let through (gate>0.5): {np.mean(energies[1:]):.1f}%"
              f"   (a dense net pays 100% every frame)\n")
        if use_flow:
            out_flow = out

    # money shot: one static-region pixel's value over time, boiled vs anchored
    py, px = np.argwhere(static_mask)[len(np.argwhere(static_mask))//2]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(boiled[:, py, px, 0], color="#c0392b", lw=1.0, label="boiled (input)")
    ax[0].plot(out_flow[:, py, px, 0], color="#2ecc71", lw=1.6, label="anchored (output)")
    ax[0].plot(clean[:, py, px, 0], color="#2c3e50", lw=1.0, ls="--", label="clean (truth)")
    ax[0].set_title(f"one static-background pixel over time  (px {px},{py})", fontsize=11)
    ax[0].set_xlabel("frame"); ax[0].set_ylabel("red value"); ax[0].legend(fontsize=9)

    smap = boiled.std(axis=0).mean(axis=2)
    omap = out_flow.std(axis=0).mean(axis=2)
    vmax = smap.max()
    ax[1].imshow(np.concatenate([smap, omap], axis=1), cmap="inferno", vmax=vmax)
    ax[1].set_title("per-pixel temporal std:  boiled  |  anchored\n"
                    "(bright = flickering; the background goes dark, the subject's path stays)",
                    fontsize=10)
    ax[1].axis("off")
    plt.tight_layout(); plt.savefig("anchor_verification.png", dpi=110, bbox_inches="tight")
    print("saved anchor_verification.png")
    print("="*74)
