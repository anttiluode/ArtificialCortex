"""
splat_predictor.py -- temporal upgrade of SplatVAE: predict the NEXT frame.

HONEST FRAMING (from koopman_predict_probe.py, which was run and measured):
  * a FREE linear Koopman operator predicts the next frame WORSE than persistence;
  * the best identity-anchored operator the search found WAS pure identity --
    no linear operator beat "copy the previous frame";
  * so the operator here is INITIALISED TO IDENTITY (it starts as
    persistence-through-the-VAE) and earns deviations only if they actually help.
    Do NOT expect it to beat persistence on PSNR. The value of training on video
    is the learned representation + the next-frame objective, not a magic
    predictor. Persistence is strong; the log prints it every epoch so you can see.

Reuses your real splat_generator classes (Encoder/Decoder/GaborRenderer), so the
renderer math is identical to the model you already trained.

  enc: frame_t -> z_t ;  koop: z_t -> z_pred (init = identity) ;  dec+ren: z -> image
  loss = recon(frame_t) + w_pred*predict(frame_t+1) + w_lin*(z_pred ~ enc(frame_t+1)) + beta*KL

DATA = a VIDEO of motion (consecutive-frame pairs). A folder of unordered CelebA
stills will NOT work -- the model must see movement. Record ~1 min of webcam.

Run:
  python splat_predictor.py --smoke
  python splat_predictor.py --video me.mp4    --image_size 128 --num_packets 512 --amp
  python splat_predictor.py --frames_dir fr/  --image_size 128 --num_packets 512 --amp
"""
import os, time, argparse, importlib.util, glob
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

def load_sg():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("splat_generator", os.path.join(here, "splat_generator.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
sg = load_sg()


class SplatPredictor(nn.Module):
    def __init__(self, image_size=128, latent=128, num_packets=512, chunk=64):
        super().__init__()
        self.enc = sg.Encoder(image_size, latent)
        self.dec = sg.Decoder(latent, num_packets)
        self.ren = sg.GaborRenderer(image_size, num_packets, chunk)
        self.koop = nn.Linear(latent, latent, bias=False)
        with torch.no_grad():
            self.koop.weight.copy_(torch.eye(latent))     # start = persistence
        self.latent = latent
    def forward(self, xt, xtp1):
        mu_t, lv_t = self.enc(xt)
        z_t = mu_t + torch.randn_like(mu_t) * torch.exp(0.5 * lv_t)
        recon_t = self.ren(self.dec(z_t))
        pred = self.ren(self.dec(self.koop(z_t)))
        mu_tp1, _ = self.enc(xtp1)
        return recon_t, pred, mu_t, lv_t, self.koop(mu_t), mu_tp1.detach()


class VideoPairs(Dataset):
    def __init__(self, path, image_size, stride=1, maxframes=4000):
        import cv2
        cap = cv2.VideoCapture(path); self.fr = []; i = 0
        while len(self.fr) < maxframes:
            ok, f = cap.read()
            if not ok: break
            if i % stride == 0:
                self.fr.append(cv2.cvtColor(cv2.resize(f, (image_size, image_size)), cv2.COLOR_BGR2RGB).astype("uint8"))
            i += 1
        cap.release()
        if len(self.fr) < 2: raise RuntimeError("need >=2 frames from the video")
    def __len__(self): return len(self.fr) - 1
    def __getitem__(self, i):
        g = lambda a: torch.from_numpy(a).permute(2, 0, 1).float() / 255.
        return g(self.fr[i]), g(self.fr[i + 1])


class FramesPairs(Dataset):
    def __init__(self, d, image_size):
        from PIL import Image; from torchvision import transforms as T
        self.paths = sorted(glob.glob(os.path.join(d, "*"))); self.Image = Image
        self.tf = T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()])
        if len(self.paths) < 2: raise RuntimeError("need >=2 ordered frames")
    def __len__(self): return len(self.paths) - 1
    def __getitem__(self, i):
        return (self.tf(self.Image.open(self.paths[i]).convert("RGB")),
                self.tf(self.Image.open(self.paths[i + 1]).convert("RGB")))


class SmokeVideo(Dataset):
    def __init__(self, n=8, size=32): self.n, self.s = n, size
    def __len__(self): return self.n
    def __getitem__(self, i):
        s = self.s; a = torch.zeros(3, s, s); b = torch.zeros(3, s, s)
        x0 = 4 + i % 8; a[:, 8:20, x0:x0+10] = 0.8; b[:, 8:20, x0+2:x0+12] = 0.8   # moved 2px
        return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--frames_dir"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="./runs/temporal")
    ap.add_argument("--image_size", type=int, default=128); ap.add_argument("--num_packets", type=int, default=512)
    ap.add_argument("--latent", type=int, default=128); ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--w_pred", type=float, default=2.0); ap.add_argument("--w_lin", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=1); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.image_size, a.num_packets, a.batch, a.epochs, a.amp = 32, 16, 4, 1, False
        dev = torch.device("cpu"); ds = SmokeVideo(8, 32)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if a.video: ds = VideoPairs(a.video, a.image_size, a.stride)
        elif a.frames_dir: ds = FramesPairs(a.frames_dir, a.image_size)
        else: raise SystemExit("give --video FILE or --frames_dir DIR (or --smoke)")
    os.makedirs(a.out, exist_ok=True)
    print(f"device={dev} pairs={len(ds)} image_size={a.image_size} packets={a.num_packets}")
    model = SplatPredictor(a.image_size, a.latent, a.num_packets, a.chunk).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=(0 if a.smoke else a.workers), drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)
    step = 0
    for ep in range(a.epochs):
        model.train(); beta = a.beta * min(1., (ep + 1) / 10.); t0 = time.time(); rr = rp = 0.
        last = None
        for xt, xtp1 in dl:
            xt, xtp1 = xt.to(dev), xtp1.to(dev); last = (xt, xtp1)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=a.amp):
                recon, pred, mu, lv, zp, mut = model(xt, xtp1)
                rec = F.mse_loss(recon, xt); pl = F.mse_loss(pred, xtp1)
                lin = F.mse_loss(zp, mut); kld = sg.kl(mu, lv)
                loss = rec + a.w_pred * pl + a.w_lin * lin + beta * kld
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.); scaler.step(opt); scaler.update()
            rr += rec.item(); rp += pl.item(); step += 1
            if a.smoke and step >= 2:
                drift = (model.koop.weight.detach() - torch.eye(a.latent)).abs().mean().item()
                print("smoke OK: fwd+bwd ran. rec=%.4f pred=%.4f  koop drift from identity=%.2e"
                      % (rec.item(), pl.item(), drift)); return
        nb = len(dl)
        with torch.no_grad():
            pers = F.mse_loss(last[0], last[1]).item()       # persistence baseline (copy frame_t)
        print(f"ep {ep+1:3d}/{a.epochs}  rec {rr/nb:.4f}  pred {rp/nb:.4f}  "
              f"(persistence {pers:.4f}; pred<persistence means the operator helps)  {time.time()-t0:4.1f}s")
        model.eval()
        with torch.no_grad():
            torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
            torch.save(model.koop.weight.detach().cpu(), os.path.join(a.out, "koopman.pt"))
            from torchvision.utils import save_image
            recon, pred, *_ = model(last[0][:8], last[1][:8])
            save_image(torch.cat([last[0][:8], recon.clamp(0, 1), pred.clamp(0, 1), last[1][:8]], 0),
                       os.path.join(a.out, f"pred_{ep+1:03d}.png"), nrow=8)
    print("done. pred_*.png rows: input_t / recon_t / PREDICTED_t+1 / true_t+1")


if __name__ == "__main__":
    main()
