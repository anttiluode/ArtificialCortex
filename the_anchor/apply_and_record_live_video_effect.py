"""
integrated_app.py — Unified Native GUI for the Cortical Tensor
==============================================================
A single-file interface that imports the libraries natively. 

Features:
- LIVE WEBCAM CASCADE: Stack X temporal anchors to recursively punish time.
- LIVE RECORDER: Capture the filtered webcam output to .mp4 in real-time.
- SVD ANCHOR: Native diffusers generation with GUI progress logging and 
  12GB VRAM safety optimizations.

PerceptionLab / Antti Luode. Helsinki, 2026. Do not hype. Just show.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import numpy as np
import sys
import os

# Webcam and Image libraries
try:
    import cv2
    from PIL import Image, ImageTk
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Import the core primitive
try:
    from stabilizer.gate import TemporalAnchor
    HAS_ANCHOR = True
except ImportError:
    HAS_ANCHOR = False


class IntegratedWorkspaceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PerceptionLab — Native Workspace (Cascade & Record)")
        self.root.geometry("850x750")
        self.root.configure(bg="#141414")
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Threads and state
        self.webcam_running = False
        self.wants_to_record = False
        self.webcam_cap = None
        self.anchors = [] 
        
        self.build_webcam_tab()
        self.build_svd_tab()
        self.build_console()

        if not HAS_CV2 or not HAS_ANCHOR:
            self.log("WARNING: cv2, PIL, or stabilizer.gate missing. Webcam won't run.")

    def log(self, text):
        self.console.configure(state='normal')
        self.console.insert(tk.END, text + "\n")
        self.console.see(tk.END)
        self.console.configure(state='disabled')

    # ==========================================
    # TAB 1: LIVE WEBCAM FILTER (CASCADE & RECORD)
    # ==========================================
    def build_webcam_tab(self):
        tab = tk.Frame(self.notebook, bg="#1a1a1a")
        self.notebook.add(tab, text="Live Webcam (Cascade & Record)")
        
        self.lbl_video = tk.Label(tab, bg="black", text="Camera Offline", fg="#555555")
        self.lbl_video.pack(pady=10, expand=True)

        ctrl_frame = tk.Frame(tab, bg="#1a1a1a")
        ctrl_frame.pack(fill=tk.X, pady=10, padx=20)
        
        tk.Label(ctrl_frame, text="Slow Hold K:", fg="white", bg="#1a1a1a").grid(row=0, column=0, sticky=tk.W)
        self.k_var = tk.DoubleVar(value=0.08)
        tk.Scale(ctrl_frame, variable=self.k_var, from_=0.01, to=0.5, resolution=0.01, orient=tk.HORIZONTAL, bg="#1a1a1a", fg="white", length=150).grid(row=0, column=1, padx=10)
        
        tk.Label(ctrl_frame, text="Surprise Thr:", fg="white", bg="#1a1a1a").grid(row=1, column=0, sticky=tk.W)
        self.thr_var = tk.DoubleVar(value=0.06)
        tk.Scale(ctrl_frame, variable=self.thr_var, from_=0.02, to=0.25, resolution=0.01, orient=tk.HORIZONTAL, bg="#1a1a1a", fg="white", length=150).grid(row=1, column=1, padx=10)

        tk.Label(ctrl_frame, text="Layer Depth (X):", fg="#e74c3c", bg="#1a1a1a", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.depth_var = tk.IntVar(value=1)
        tk.Scale(ctrl_frame, variable=self.depth_var, from_=1, to=20, resolution=1, orient=tk.HORIZONTAL, bg="#1a1a1a", fg="#e74c3c", length=150).grid(row=2, column=1, padx=10, pady=5)

        # Action Buttons
        self.btn_webcam = tk.Button(ctrl_frame, text="Start Webcam", command=self.toggle_webcam, bg="#2c3e50", fg="white", font=("Arial", 10, "bold"))
        self.btn_webcam.grid(row=0, column=2, rowspan=2, padx=20, ipadx=10, ipady=5)

        self.btn_record = tk.Button(ctrl_frame, text="Start Recording", command=self.toggle_record, bg="#8e44ad", fg="white", font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.btn_record.grid(row=2, column=2, padx=20, ipadx=10, ipady=5)

    def toggle_webcam(self):
        if not self.webcam_running:
            self.webcam_running = True
            self.btn_webcam.config(text="Stop Webcam", bg="#c0392b")
            self.btn_record.config(state=tk.NORMAL)
            threading.Thread(target=self._webcam_thread, daemon=True).start()
        else:
            self.webcam_running = False
            self.wants_to_record = False
            self.btn_webcam.config(text="Start Webcam", bg="#2c3e50")
            self.btn_record.config(text="Start Recording", bg="#8e44ad", state=tk.DISABLED)

    def toggle_record(self):
        if not self.wants_to_record:
            self.wants_to_record = True
            self.btn_record.config(text="Stop Recording", bg="#e74c3c")
        else:
            self.wants_to_record = False
            self.btn_record.config(text="Start Recording", bg="#8e44ad")

    def _webcam_thread(self):
        if not HAS_CV2: return
        self.webcam_cap = cv2.VideoCapture(0)
        
        video_writer = None
        frame_width, frame_height = 480, 360 # Fixed processing resolution
        
        while self.webcam_running:
            ret, frame = self.webcam_cap.read()
            if not ret: 
                time.sleep(0.03)
                continue
            
            frame = cv2.resize(frame, (frame_width, frame_height))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # Cascade Depth Management
            target_depth = self.depth_var.get()
            if len(self.anchors) != target_depth:
                if len(self.anchors) < target_depth:
                    for _ in range(target_depth - len(self.anchors)):
                        self.anchors.append(TemporalAnchor(K=self.k_var.get(), surprise_thr=self.thr_var.get()))
                else:
                    self.anchors = self.anchors[:target_depth]
            
            # Push through cascade
            out = rgb
            for layer in self.anchors:
                layer.K = self.k_var.get()
                layer.surprise_thr = self.thr_var.get()
                out, _, _ = layer.step(out)
            
            out_uint8 = (out * 255).astype(np.uint8)

            # Thread-safe recording logic
            if self.wants_to_record and video_writer is None:
                filename = f"tensor_record_{int(time.time())}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (frame_width, frame_height))
                self.root.after(0, lambda f=filename: self.log(f"[RECORDING] Started saving to: {f}"))
            
            elif not self.wants_to_record and video_writer is not None:
                video_writer.release()
                video_writer = None
                self.root.after(0, lambda: self.log("[RECORDING] File saved successfully."))
            
            if video_writer is not None:
                # VideoWriter expects BGR, not RGB
                bgr_out = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_out)
            
            # Render to Tkinter
            img = ImageTk.PhotoImage(Image.fromarray(out_uint8))
            self.root.after(0, self._update_webcam_lbl, img)
            
            time.sleep(1/60.0)
            
        if video_writer is not None:
            video_writer.release()

        self.webcam_cap.release()
        self.root.after(0, lambda: self.lbl_video.config(image='', text="Camera Offline"))

    def _update_webcam_lbl(self, img):
        self.lbl_video.config(image=img)
        self.lbl_video.image = img

    # ==========================================
    # TAB 2: SVD GENERATOR (NO PROMPT)
    # ==========================================
    def build_svd_tab(self):
        tab = tk.Frame(self.notebook, bg="#1a1a1a")
        self.notebook.add(tab, text="SVD Generator")
        
        tk.Label(tab, text="Stable Video Diffusion natively generates video from an image (no prompt).", fg="#bdc3c7", bg="#1a1a1a").pack(pady=10)

        frame_inputs = tk.Frame(tab, bg="#1a1a1a")
        frame_inputs.pack(fill=tk.X, padx=20, pady=10)
        
        self.svd_img = tk.StringVar()
        tk.Label(frame_inputs, text="Source Image:", fg="white", bg="#1a1a1a").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(frame_inputs, textvariable=self.svd_img, width=40).grid(row=0, column=1, padx=10)
        tk.Button(frame_inputs, text="Browse", command=lambda: self.svd_img.set(filedialog.askopenfilename())).grid(row=0, column=2)

        self.svd_out = tk.StringVar(value="svd_anchored.mp4")
        tk.Label(frame_inputs, text="Output Target:", fg="white", bg="#1a1a1a").grid(row=1, column=0, sticky=tk.W, pady=10)
        tk.Entry(frame_inputs, textvariable=self.svd_out, width=40).grid(row=1, column=1, padx=10)
        
        param_frame = tk.Frame(tab, bg="#1a1a1a")
        param_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(param_frame, text="Anchor Strength:", fg="white", bg="#1a1a1a").grid(row=0, column=0)
        self.svd_str = tk.DoubleVar(value=0.6)
        tk.Entry(param_frame, textvariable=self.svd_str, width=6).grid(row=0, column=1, padx=5)

        tk.Label(param_frame, text="Variance Thr:", fg="white", bg="#1a1a1a").grid(row=0, column=2, padx=15)
        self.svd_var = tk.DoubleVar(value=0.6)
        tk.Entry(param_frame, textvariable=self.svd_var, width=6).grid(row=0, column=3, padx=5)

        tk.Button(tab, text="Generate Video (12GB VRAM Safe)", command=self.launch_svd, bg="#27ae60", fg="white", font=("Arial", 12, "bold")).pack(pady=20, ipadx=10, ipady=5)

    def launch_svd(self):
        if not self.svd_img.get():
            messagebox.showerror("Error", "Select a source image first.")
            return
        threading.Thread(target=self._svd_thread, daemon=True).start()

    def _svd_thread(self):
        self.log("Importing PyTorch and Diffusers (this takes a moment)...")
        try:
            import torch
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import load_image, export_to_video
        except ImportError:
            self.log("ERROR: Could not import torch or diffusers.")
            return

        def make_anchor_callback(total_steps, strength, var_thr):
            def callback(pipe, step, timestep, callback_kwargs):
                lat = callback_kwargs["latents"]
                if lat.dim() != 5 or lat.shape[1] < 2: return callback_kwargs
                
                progress = step / max(total_steps - 1, 1)
                if progress < 0.25: return callback_kwargs
                
                ramp = (progress - 0.25) / 0.75
                s = strength * float(min(max(ramp, 0.0), 1.0))
                
                lat_32 = lat.to(torch.float32)
                tmean = lat_32.mean(dim=1, keepdim=True)
                var = ((lat_32 - tmean) ** 2).mean(dim=(1, 2), keepdim=True).sqrt()
                med = var.median()
                vnorm = var / (med + 1e-6)
                
                g = torch.sigmoid((vnorm - var_thr) / 0.25)
                locked = (1.0 - s) * lat_32 + s * tmean
                lat_32 = g * lat_32 + (1.0 - g) * locked
                
                callback_kwargs["latents"] = lat_32.to(lat.dtype)
                
                self.root.after(0, lambda: self.log(f"Rendering step {step + 1}/{total_steps}..."))
                return callback_kwargs
            return callback

        try:
            self.log("Loading SVD xt variant into GPU...")
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt", 
                torch_dtype=torch.float16, variant="fp16"
            )
            pipe.to("cuda")
            
            pipe.enable_model_cpu_offload()

            self.log(f"Preparing image: {self.svd_img.get()}")
            image = load_image(self.svd_img.get()).resize((576, 320)) 
            
            cb = make_anchor_callback(25, self.svd_str.get(), self.svd_var.get())
            
            self.log("Generating... (Watch console for progress)")
            gen = torch.manual_seed(0)
            result = pipe(
                image, num_frames=25, num_inference_steps=25,
                fps=7, generator=gen,
                decode_chunk_size=2,
                callback_on_step_end=cb,
                callback_on_step_end_tensor_inputs=["latents"]
            )
            
            self.log("Decoding complete! Saving video...")
            out_path = self.svd_out.get()
            export_to_video(result.frames[0], out_path, fps=7)
            self.log(f"\n[SUCCESS] Saved anchored video to: {out_path}")
            
        except Exception as e:
            self.log(f"[ERROR] Generation failed: {str(e)}")

    # ==========================================
    # CONSOLE
    # ==========================================
    def build_console(self):
        console_frame = tk.Frame(self.root, bg="#1a1a1a")
        console_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        tk.Label(console_frame, text="Execution Log", fg="#7f8c8d", bg="#1a1a1a").pack(anchor=tk.W)
        self.console = tk.Text(console_frame, height=8, bg="#0d0d0d", fg="#2ecc71", font=("Consolas", 9), state='disabled')
        self.console.pack(fill=tk.BOTH, expand=True)

if __name__ == '__main__':
    root = tk.Tk()
    app = IntegratedWorkspaceGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: sys.exit(0))
    root.mainloop()
