"""
video_tensor.py — the cortical tensor pointed at a live webcam, with the inner/outer
                  views and the cortical-layer stack made visible
=====================================================================================
This runs the energy-on-surprise + theta-gating math of `the_tensor/cortical_tensor.py`
on a live video feed. The field math is a pure-numpy CORE (VideoTensorCore), verified
headlessly on synthetic video; the GUI is a thin wrapper around it.

THE FIELD (per frame):
  SLOW layer  — a leaky predictor of the room: pred <- (1-K)*pred + K*world
                (the WHERE — it expects things to stay where they are; this is the
                 buffer's K, here as a spatial Kalman-ish hold, not full Koopman).
  FAST layer  — the residual: surprise = |world - pred|  (the NOW it didn't foresee).
  THETA clock — a deep pacemaker; its phase gates the broadcast.
  GAMMA       — surprise * theta_gate: the energy actually spent, and it BREATHES.
  STACK (Z)   — the same surprise read through several band clocks (the cortical
                layers): slow bands pulse slowly, faster ones flicker.
  INNER view  — the model's internal state: the held prediction with the gated
                surprise flaring on top (what the cortex is "running"); the OUTER
                view is the raw world. (Calling the inner view 'qualia' is Antti's
                word; whether it is FELT is the bet, and it stays in the drawer.)

ENERGY METRIC (honest): a dense network processes every pixel every frame (100%).
This spends only on pixels whose surprise clears a threshold — the fraction it would
actually need to recompute. Sit still and that fraction collapses toward sensor
noise; move, and it rises only locally, where the prediction broke.

Run:  pip install opencv-python numpy pillow ; python video_tensor.py
(no webcam? it falls back to a synthetic moving-blob world so you still see it work.)
PerceptionLab / Antti Luode, with Claude (Opus 4.8); first video build by Gemini.
Helsinki, June 2026. Do not hype. Do not lie. Just show.
"""
import numpy as np


# =====================================================================
# THE CORE — pure numpy, no cv2 / tk / PIL, so it can be unit-tested headless
# =====================================================================
class VideoTensorCore:
    def __init__(self, K=0.08, theta_hz=6.0, band_hz=(3.0, 5.0, 8.0, 11.0),
                 sharp=3.0, surprise_thr=0.06):
        self.K = float(K)
        self.theta_hz = float(theta_hz)
        self.band_hz = np.array(band_hz, float)
        self.sharp = float(sharp)
        self.surprise_thr = float(surprise_thr)
        self.pred = None
        self.theta_phase = 0.0
        self.band_phase = np.zeros(len(self.band_hz))

    def reset(self):
        self.pred = None

    def step(self, gray, dt):
        """gray: HxW float in [0,1]; dt: seconds. Returns the full field state."""
        if self.pred is None or self.pred.shape != gray.shape:
            self.pred = gray.copy()
        # SLOW layer: leaky spatial prediction (the held room)
        self.pred = (1.0 - self.K) * self.pred + self.K * gray
        # FAST layer: residual
        surprise = np.abs(gray - self.pred)
        # THETA clock + gamma broadcast
        self.theta_phase = (self.theta_phase + 2*np.pi*self.theta_hz*dt) % (2*np.pi)
        gate = (0.5*(1 + np.cos(self.theta_phase)))**self.sharp
        gamma = surprise * gate
        # the cortical STACK (Z-axis): same surprise through several band clocks
        self.band_phase = (self.band_phase + 2*np.pi*self.band_hz*dt) % (2*np.pi)
        band_gates = (0.5*(1 + np.cos(self.band_phase)))**self.sharp
        bands = [surprise * g for g in band_gates]
        # honest energy: fraction of pixels above the surprise threshold (what a
        # sparse predictive system would actually recompute) vs dense = 100%
        active = float(np.mean(surprise > self.surprise_thr))
        return dict(prediction=self.pred, surprise=surprise, gamma=gamma,
                    gate=float(gate), bands=bands, band_gates=band_gates,
                    active=active, energy_pct=100.0*active,
                    mean_gamma=float(gamma.mean()))


# =====================================================================
# THE GUI — imports cv2/PIL/tk lazily so the core stays headless-testable
# =====================================================================
def run_gui():
    import cv2, time, threading
    import tkinter as tk
    from PIL import Image, ImageTk

    PROC_W, PROC_H = 213, 160          # field resolution (fast, clean dynamics)
    HERO = (430, 323)                   # big inner/outer panels
    MED  = (260, 195)                   # prediction / surprise / gamma
    BAND = (150, 112)                   # cortical-stack tiles

    def colorize(field, cmap, gain=3.0):
        u = (np.clip(field*gain, 0, 1)*255).astype(np.uint8)
        return cv2.applyColorMap(u, cmap)

    def inner_view(pred, gamma, gain=3.0):
        base = cv2.cvtColor((pred*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        base = (base*0.55).astype(np.uint8)
        hot = colorize(gamma, cv2.COLORMAP_INFERNO, gain)
        return cv2.addWeighted(base, 1.0, hot, 1.0, 0)

    class GUI:
        def __init__(self, master):
            self.m = master
            master.title("The Video Tensor — inner/outer + cortical layers")
            master.configure(bg="#141414")
            self.core = VideoTensorCore()
            self.running = True
            self.fps = 0.0
            self._imgs = {}
            self._build()
            self.cap = cv2.VideoCapture(0)
            if self.cap is not None and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                self.synthetic = False
            else:
                self.synthetic = True
            self.t0 = time.time()
            threading.Thread(target=self.loop, daemon=True).start()

        def _panel(self, parent, title, r, c, color="#ecf0f1"):
            f = tk.Frame(parent, bg="#141414")
            f.grid(row=r, column=c, padx=8, pady=(2, 8))
            tk.Label(f, text=title, fg=color, bg="#141414",
                     font=("Helvetica", 10, "bold")).pack()
            lbl = tk.Label(f, bg="black"); lbl.pack()
            return lbl

        def _build(self):
            # controls
            ctl = tk.Frame(self.m, bg="#1f2d3d", pady=8); ctl.pack(fill="x")
            def slider(label, lo, hi, res, init):
                tk.Label(ctl, text=label, fg="white", bg="#1f2d3d").pack(side="left", padx=(12, 2))
                s = tk.Scale(ctl, from_=lo, to=hi, resolution=res, orient="horizontal",
                             bg="#1f2d3d", fg="white", highlightthickness=0, length=130)
                s.set(init); s.pack(side="left"); return s
            self.k_s = slider("Slow hold K", 0.01, 0.5, 0.01, 0.08)
            self.th_s = slider("Theta Hz", 1.0, 12.0, 0.5, 6.0)
            self.thr_s = slider("Surprise thr", 0.02, 0.25, 0.01, 0.06)
            tk.Button(ctl, text="Reset prediction (amnesia)", command=self.core.reset,
                      bg="#c0392b", fg="white", font=("Helvetica", 9, "bold")).pack(side="right", padx=12)

            # heroes: outer | inner
            heroes = tk.Frame(self.m, bg="#141414"); heroes.pack()
            self.l_outer = self._panel(heroes, "OUTER — the world arriving", 0, 0, "#5dade2")
            self.l_inner = self._panel(heroes, "INNER — held field + what surprises it", 0, 1, "#e67e22")

            # mediums: prediction | surprise | gamma
            mids = tk.Frame(self.m, bg="#141414"); mids.pack()
            self.l_pred = self._panel(mids, "slow layer: held prediction (WHERE)", 0, 0)
            self.l_surp = self._panel(mids, "surprise = |world - prediction| (NOW)", 0, 1)
            self.l_gamma = self._panel(mids, "gamma energy: gated surprise (the spend)", 0, 2)

            # cortical stack strip
            tk.Label(self.m, text="CORTICAL STACK (Z) — the same surprise read through band clocks",
                     fg="#bdc3c7", bg="#141414", font=("Helvetica", 10, "italic")).pack(pady=(6, 0))
            strip = tk.Frame(self.m, bg="#141414"); strip.pack()
            self.l_bands = []
            for i, hz in enumerate(self.core.band_hz):
                self.l_bands.append(self._panel(strip, f"{hz:.0f} Hz", 0, i, "#95a5a6"))

            # energy meter
            bot = tk.Frame(self.m, bg="#141414", pady=8); bot.pack(fill="x", side="bottom")
            self.meter_txt = tk.Label(bot, text="", fg="white", bg="#141414",
                                      font=("Courier", 11, "bold")); self.meter_txt.pack()
            self.canvas = tk.Canvas(bot, width=860, height=26, bg="#000000",
                                    highlightthickness=0); self.canvas.pack(pady=4)
            self.bar = self.canvas.create_rectangle(0, 0, 0, 26, fill="#2ecc71", width=0)
            self.bar_txt = self.canvas.create_text(430, 13, text="", fill="white",
                                                   font=("Courier", 11, "bold"))

        def synth_frame(self, t):
            yy, xx = np.mgrid[0:PROC_H, 0:PROC_W]
            bg = (0.35 + 0.12*np.sin(xx/30.0)).astype(np.float32)
            cx = PROC_W*(0.5 + 0.32*np.sin(t*1.3)); cy = PROC_H*(0.5 + 0.3*np.cos(t*0.9))
            blob = 0.6*np.exp(-(((xx-cx)**2 + (yy-cy)**2)/(2*14.0**2)))
            g = np.clip(bg + blob, 0, 1).astype(np.float32)
            return cv2.cvtColor((g*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        def loop(self):
            last = time.time()
            while self.running:
                if self.synthetic:
                    frame = self.synth_frame(time.time() - self.t0)
                else:
                    ok, frame = self.cap.read()
                    if not ok: time.sleep(0.03); continue
                frame = cv2.resize(frame, (PROC_W, PROC_H))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0

                now = time.time(); dt = max(now - last, 1e-3); last = now
                self.fps = 0.9*self.fps + 0.1*(1.0/dt)
                self.core.K = self.k_s.get(); self.core.theta_hz = self.th_s.get()
                self.core.surprise_thr = self.thr_s.get()
                st = self.core.step(gray, min(dt, 0.1))

                outer = frame
                pred = cv2.cvtColor((st["prediction"]*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                surp = colorize(st["surprise"], cv2.COLORMAP_VIRIDIS)
                gamma = colorize(st["gamma"], cv2.COLORMAP_INFERNO)
                inner = inner_view(st["prediction"], st["gamma"])
                bands = [colorize(b, cv2.COLORMAP_INFERNO) for b in st["bands"]]

                self.m.after(0, self.render, outer, inner, pred, surp, gamma, bands,
                             st["energy_pct"], st["gate"])
                time.sleep(max(0.0, 1/30.0 - (time.time()-now)))

        def _tk(self, bgr, size, key):
            img = cv2.resize(bgr, size)
            ph = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            self._imgs[key] = ph
            return ph

        def render(self, outer, inner, pred, surp, gamma, bands, energy, gate):
            self.l_outer.config(image=self._tk(outer, HERO, "o"))
            self.l_inner.config(image=self._tk(inner, HERO, "i"))
            self.l_pred.config(image=self._tk(pred, MED, "p"))
            self.l_surp.config(image=self._tk(surp, MED, "s"))
            self.l_gamma.config(image=self._tk(gamma, MED, "g"))
            for i, b in enumerate(bands):
                self.l_bands[i].config(image=self._tk(b, BAND, f"b{i}"))
            saved = 100.0 - energy
            w = (energy/100.0)*860
            self.canvas.coords(self.bar, 0, 0, w, 26)
            self.canvas.itemconfig(self.bar, fill="#e74c3c" if energy > 25 else "#2ecc71")
            self.canvas.itemconfig(self.bar_txt, text=f"{energy:4.1f}% spent")
            mode = "SYNTHETIC (no webcam)" if self.synthetic else "webcam"
            self.meter_txt.config(
                text=f"Energy spent vs static dense net:  {energy:5.1f}%   "
                     f"|  compute saved: {saved:5.1f}%   |  theta gate: {gate:.2f}   "
                     f"|  {self.fps:4.1f} fps  [{mode}]")

        def close(self):
            self.running = False
            try:
                if not self.synthetic and self.cap is not None: self.cap.release()
            except Exception: pass
            self.m.destroy()

    root = tk.Tk()
    app = GUI(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
