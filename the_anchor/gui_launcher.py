"""
gui_launcher.py — Graphical User Interface for the Cortical Tensor Workspace
===========================================================================
A unified, zero-CLI control panel to launch the live webcam tensor, de-boil 
existing video files, or run the anchored Stable Video Diffusion generation loop.

Runs long-running tasks in background threads to maintain UI responsiveness.

PerceptionLab / Antti Luode. Helsinki, 2026. Do not hype. Just show.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import subprocess
import sys
import os

class TensorWorkspaceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PerceptionLab — Cortical Tensor Workspace")
        self.root.geometry("680x620")
        self.root.resizable(True, True)
        
        # Configure overall layout style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Notebook for Tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Build Tabs
        self.build_webcam_tab()
        self.build_filter_tab()
        self.build_svd_tab()
        
        # Global Log / Output Console
        self.build_console()

    def log(self, text):
        """Append text safely to the console window from any thread."""
        self.console.configure(state='normal')
        self.console.insert(tk.END, text + "\n")
        self.console.see(tk.END)
        self.console.configure(state='disabled')

    def run_command_async(self, cmd, success_message="Task completed successfully."):
        """Execute a CLI command in a background thread to prevent UI freezing."""
        def worker():
            self.log(f"Running: {' '.join(cmd)}")
            try:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                    text=True, bufsize=1, universal_newlines=True
                )
                if process.stdout:
                    for line in process.stdout:
                        self.log(line.strip())
                process.wait()
                if process.returncode == 0:
                    self.log(f"\n[SUCCESS] {success_message}")
                    messagebox.showinfo("Success", success_message)
                else:
                    self.log(f"\n[ERROR] Command exited with code {process.returncode}")
                    messagebox.showerror("Error", f"Task failed with exit code {process.returncode}")
            except Exception as e:
                self.log(f"\n[EXCEPTION] {str(e)}")
                messagebox.showerror("Exception", f"Failed to execute command:\n{str(e)}")
        
        threading.Thread(target=worker, daemon=True).start()

    # ==========================================
    # TAB 1: LIVE WEBCAM TENSOR
    # ==========================================
    def build_webcam_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Live Webcam Tensor")
        
        lbl_info = ttk.Label(
            tab, text="The Cortical Tensor pointed at a live webcam feed.\n"
                       "Measures surprise, applies the spatial hold (K), and drives the theta pacemaker.",
            justify=tk.LEFT, padding=15
        )
        lbl_info.pack(anchor=tk.W)
        
        btn_launch = ttk.Button(tab, text="Launch Live Webcam Feed", command=self.launch_webcam)
        btn_launch.pack(padx=20, pady=20, ipadx=10, ipady=5)

    def launch_webcam(self):
        if not os.path.exists("video_tensor.py"):
            messagebox.showerror("Error", "Could not find video_tensor.py in the working directory.")
            return
        cmd = [sys.executable, "video_tensor.py"]
        self.run_command_async(cmd, "Webcam session closed.")

    # ==========================================
    # TAB 2: VIDEO FILE DE-BOILER
    # ==========================================
    def build_filter_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Video De-Boiler (Filter)")
        
        # File Inputs
        frame_files = ttk.LabelFrame(tab, text=" File Selections ", padding=10)
        frame_files.pack(fill=tk.X, padx=10, pady=5)
        
        self.input_video = tk.StringVar()
        ttk.Label(frame_files, text="Input Video:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame_files, textvariable=self.input_video, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(frame_files, text="Browse...", command=lambda: self.browse_file(self.input_video, [("Video Files", "*.mp4 *.avi *.mov")])).grid(row=0, column=2)

        self.output_video = tk.StringVar(value="deboiled_output.mp4")
        ttk.Label(frame_files, text="Output Video:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame_files, textvariable=self.output_video, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(frame_files, text="Save As...", command=lambda: self.save_file(self.output_video, [("Video Files", "*.mp4")])).grid(row=1, column=2)

        # Knobs / Parameters
        frame_knobs = ttk.LabelFrame(tab, text=" Hyperparameters ", padding=10)
        frame_knobs.pack(fill=tk.X, padx=10, pady=5)
        
        # Threshold
        ttk.Label(frame_knobs, text="Surprise Threshold (--thr):").grid(row=0, column=0, sticky=tk.W)
        self.val_thr = tk.DoubleVar(value=0.10)
        scale_thr = ttk.Scale(frame_knobs, from_=0.01, to=0.50, variable=self.val_thr, orient=tk.HORIZONTAL, length=300)
        scale_thr.grid(row=0, column=1, padx=10, pady=5)
        lbl_thr = ttk.Label(frame_knobs, text="0.10")
        lbl_thr.grid(row=0, column=2)
        self.val_thr.trace_add("write", lambda *args: lbl_thr.config(text=f"{self.val_thr.get():.2f}"))

        # K Slow Hold Rate
        ttk.Label(frame_knobs, text="Slow-Hold Rate (--K):").grid(row=1, column=0, sticky=tk.W)
        self.val_k = tk.DoubleVar(value=0.03)
        scale_k = ttk.Scale(frame_knobs, from_=0.001, to=0.20, variable=self.val_k, orient=tk.HORIZONTAL, length=300)
        scale_k.grid(row=1, column=1, padx=10, pady=5)
        lbl_k = ttk.Label(frame_knobs, text="0.03")
        lbl_k.grid(row=1, column=2)
        self.val_k.trace_add("write", lambda *args: lbl_k.config(text=f"{self.val_k.get():.3f}"))

        # Flags
        self.flag_flow = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_knobs, text="Motion-Compensate Hold (--flow) [Use if camera moves]", variable=self.flag_flow).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.flag_side = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame_knobs, text="Side-by-Side Visualization Render (--side-by-side)", variable=self.flag_side).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Action Button
        btn_run = ttk.Button(tab, text="Run Post-Hoc De-Boiler Filter", command=self.run_filter)
        btn_run.pack(pady=10, ipadx=10, ipady=3)

    def run_filter(self):
        if not self.input_video.get():
            messagebox.showerror("Error", "Please pick an input video file first.")
            return
        if not os.path.exists("run_filter.py"):
            messagebox.showerror("Error", "Could not find run_filter.py in the working directory.")
            return
            
        cmd = [
            sys.executable, "run_filter.py", self.input_video.get(),
            "-o", self.output_video.get(),
            "--thr", f"{self.val_thr.get():.2f}",
            "--K", f"{self.val_k.get():.3f}"
        ]
        if self.flag_flow.get():
            cmd.append("--flow")
        if self.flag_side.get():
            cmd.append("--side-by-side")
            
        self.run_command_async(cmd, f"Video de-boiling complete!\nSaved to: {self.output_video.get()}")

    # ==========================================
    # TAB 3: GENERATIVE SVD ANCHOR
    # ==========================================
    def build_svd_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Generative SVD Anchor")
        
        frame_svd = ttk.LabelFrame(tab, text=" Stable Video Diffusion Config ", padding=10)
        frame_svd.pack(fill=tk.X, padx=10, pady=5)
        
        # Model selection
        self.svd_model = tk.StringVar(value="stabilityai/stable-video-diffusion-img2vid-xt")
        ttk.Label(frame_svd, text="HuggingFace Model Path:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame_svd, textvariable=self.svd_model, width=45).grid(row=0, column=1, padx=5, sticky=tk.W)
        
        # Input Image
        self.svd_image = tk.StringVar()
        ttk.Label(frame_svd, text="Source Image:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame_svd, textvariable=self.svd_image, width=45).grid(row=1, column=1, padx=5, sticky=tk.W)
        ttk.Button(frame_svd, text="Browse...", command=lambda: self.browse_file(self.svd_image, [("Images", "*.png *.jpg *.jpeg")])).grid(row=1, column=2, sticky=tk.W)

        # Output Target
        self.svd_output = tk.StringVar(value="svd_anchored.mp4")
        ttk.Label(frame_svd, text="Output Target:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame_svd, textvariable=self.svd_output, width=45).grid(row=2, column=1, padx=5, sticky=tk.W)
        ttk.Button(frame_svd, text="Save As...", command=lambda: self.save_file(self.svd_output, [("Video Files", "*.mp4")])).grid(row=2, column=2, sticky=tk.W)

        # Numerical Adjustments
        frame_params = ttk.Frame(frame_svd)
        frame_params.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Steps
        ttk.Label(frame_params, text="Steps:").grid(row=0, column=0, sticky=tk.W)
        self.svd_steps = tk.IntVar(value=25)
        ttk.Entry(frame_params, textvariable=self.svd_steps, width=6).grid(row=0, column=1, padx=5, pady=2)

        # Frames
        ttk.Label(frame_params, text="Frames:").grid(row=0, column=2, sticky=tk.W, padx=10)
        self.svd_frames = tk.IntVar(value=14)
        ttk.Entry(frame_params, textvariable=self.svd_frames, width=6).grid(row=0, column=3, padx=5, pady=2)

        # Pull Strength
        ttk.Label(frame_params, text="Anchor Strength:").grid(row=1, column=0, sticky=tk.W)
        self.svd_strength = tk.DoubleVar(value=0.15)
        ttk.Entry(frame_params, textvariable=self.svd_strength, width=6).grid(row=1, column=1, padx=5, pady=2)

        # Variance Threshold
        ttk.Label(frame_params, text="Variance Thr:").grid(row=1, column=2, sticky=tk.W, padx=10)
        self.svd_var = tk.DoubleVar(value=0.05)
        ttk.Entry(frame_params, textvariable=self.svd_var, width=6).grid(row=1, column=3, padx=5, pady=2)

        # Baseline
        self.flag_baseline = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_svd, text="Render Comparative Baseline Copy (Anchor OFF)", variable=self.flag_baseline).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)

        btn_gen = ttk.Button(tab, text="Generate Anchored Video via Pipeline", command=self.run_svd)
        btn_gen.pack(pady=10, ipadx=10, ipady=3)

    def run_svd(self):
        if not self.svd_image.get():
            messagebox.showerror("Error", "Please select a source initiation image first.")
            return
        if not os.path.exists("run_svd.py"):
            messagebox.showerror("Error", "Could not find run_svd.py script.")
            return
            
        cmd = [
            sys.executable, "run_svd.py",
            "--model", self.svd_model.get(),
            "--image", self.svd_image.get(),
            "--output", self.svd_output.get(),
            "--steps", str(self.svd_steps.get()),
            "--frames", str(self.svd_frames.get()),
            "--strength", f"{self.svd_strength.get():.4f}",
            "--var_thr", f"{self.svd_var.get():.4f}"
        ]
        if self.flag_baseline:
            cmd.append("--baseline")
            
        self.run_command_async(cmd, f"SVD generation complete!\nOutput video saved.")

    # ==========================================
    # UTILITIES & CONSOLE
    # ==========================================
    def build_console(self):
        frame_console = ttk.LabelFrame(self.root, text=" Live Execution Console / Standard Output ")
        frame_console.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.console = tk.Text(frame_console, height=10, state='disabled', font=("Consolas", 9), wrap=tk.WORD)
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scroll = ttk.Scrollbar(frame_console, command=self.console.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.console.configure(yscrollcommand=scroll.set)

    def browse_file(self, var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def save_file(self, var, filetypes):
        path = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=filetypes[0][1].split()[-1])
        if path:
            var.set(path)

if __name__ == '__main__':
    root = tk.Tk()
    app = TensorWorkspaceGUI(root)
    root.mainloop()