# Foveated saccadic stitching, shown offline (autograd, CPU, no trained model).
# A blurry low-frequency GIST is laid down once (the held prior). Then the "eye"
# saccades to wherever the prediction is WORST (error-driven), spawns a few
# high-frequency Gabor packets INSIDE a foveal window, fits them to the local
# residual, and COMMITS them. The percept is held across fixations and sharpens
# only where the fovea has visited — peripheral stays gist. Inhibition-of-return
# stops it re-fixating the same spot. This is the mechanism; the live webcam
# version wires the same loop onto the trained SplatVAE.
import time, numpy as np
import autograd.numpy as anp
from autograd import grad
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.cbook as cb
from scipy.ndimage import zoom, gaussian_filter

img = plt.imread(cb.get_sample_data("grace_hopper.jpg")).astype(float)
if img.ndim == 3: img = img[..., :3].mean(2)
H = 84; img = zoom(img, (H/img.shape[0], H/img.shape[1]), order=1)
img = (img - img.min())/(img.max()-img.min())
gy, gx = np.meshgrid(np.linspace(0,1,H), np.linspace(0,1,H), indexing="ij")
GX, GY, TGT = anp.array(gx), anp.array(gy), anp.array(img)

def render(px, py, sig, th, f, a, b):
    px=px[:,None,None]; py=py[:,None,None]; s=sig[:,None,None]; th=th[:,None,None]
    f=f[:,None,None]; a=a[:,None,None]; b=b[:,None,None]
    dx=GX[None]-px; dy=GY[None]-py; xr=dx*anp.cos(th)+dy*anp.sin(th)
    env=anp.exp(-(dx*dx+dy*dy)/(2*s*s))
    return anp.sum(env*(a*anp.cos(2*np.pi*f*xr)-b*anp.sin(2*np.pi*f*xr)), axis=0)

def adam(v, gl, it, lr):
    m=np.zeros_like(v); s=np.zeros_like(v); b1,b2=0.9,0.999
    for t in range(1,it+1):
        g=gl(v); m=b1*m+(1-b1)*g; s=b2*s+(1-b2)*g*g
        v=v-lr*(m/(1-b1**t))/(np.sqrt(s/(1-b2**t))+1e-8)
    return v

# ---- 1. the GIST: a few low-frequency packets fit globally (the held prior) ----
rng=np.random.default_rng(0); Ng=70
side=int(np.ceil(np.sqrt(Ng))); g=np.linspace(0.1,0.9,side)
pos=np.array([(a,b) for a in g for b in g])[:Ng]
gist_fixed=dict(px=pos[:,0], py=pos[:,1], sig=np.full(Ng,0.12),
                th=rng.uniform(0,np.pi,Ng), f=np.full(Ng,2.5))
def gist_loss(v):
    a,b,bias=v[:Ng],v[Ng:2*Ng],v[2*Ng]
    r=render(anp.array(gist_fixed['px']),anp.array(gist_fixed['py']),anp.array(gist_fixed['sig']),
             anp.array(gist_fixed['th']),anp.array(gist_fixed['f']),a,b)+bias
    return anp.mean((r-TGT)**2)
vg=adam(np.concatenate([rng.normal(0,.05,Ng),rng.normal(0,.05,Ng),[float(img.mean())]]),
        grad(gist_loss), 120, 0.03)
ga,gb,bias=vg[:Ng],vg[Ng:2*Ng],vg[2*Ng]
committed = np.array(render(gist_fixed['px'],gist_fixed['py'],gist_fixed['sig'],
                            gist_fixed['th'],gist_fixed['f'],ga,gb))+bias
prior = committed.copy()

# ---- 2. SACCADES: look where error is worst, spawn foveal high-freq detail ----
K=10; rad=0.16; per=26
ior=np.zeros((H,H)); fixations=[]
for k in range(K):
    resid=np.abs(committed-img)
    err=gaussian_filter(resid,2.0)*(1-ior)
    fy,fx=np.unravel_index(np.argmax(err),err.shape)
    fxn,fyn=fx/(H-1),fy/(H-1); fixations.append((fxn,fyn))
    # foveal window + inhibition of return
    win=np.exp(-((gx-fxn)**2+(gy-fyn)**2)/(2*rad**2))
    ior=np.clip(ior+np.exp(-((gx-fxn)**2+(gy-fyn)**2)/(2*(rad*0.9)**2)),0,1)
    # spawn high-freq packets inside the fovea, fit to the LOCAL residual
    ang=rng.uniform(0,2*np.pi,per); rr=rad*np.sqrt(rng.uniform(0,1,per))
    spx=np.clip(fxn+rr*np.cos(ang),0,1); spy=np.clip(fyn+rr*np.sin(ang),0,1)
    sth=rng.uniform(0,np.pi,per); sf=np.exp(np.linspace(np.log(7),np.log(26),per))[rng.permutation(per)]
    W=anp.array(win); HELD=anp.array(committed)
    def loss(v):
        a,b=v[:per],v[per:2*per]
        r=HELD+render(anp.array(spx),anp.array(spy),anp.full(per,0.045),
                      anp.array(sth),anp.array(sf),a,b)
        return anp.sum(W*(r-TGT)**2)/(anp.sum(W)+1e-6)
    vv=adam(rng.normal(0,.02,2*per), grad(loss), 50, 0.05)
    committed = committed + np.array(render(spx,spy,np.full(per,0.045),sth,sf,vv[:per],vv[per:2*per]))

def psnr(a,b,m=None):
    if m is None: e=np.mean((a-b)**2)
    else: e=np.sum(m*(a-b)**2)/np.sum(m)
    return 10*np.log10(1/max(e,1e-9))
foveated=np.zeros((H,H))
for fxn,fyn in fixations: foveated+=np.exp(-((gx-fxn)**2+(gy-fyn)**2)/(2*rad**2))
foveated=np.clip(foveated,0,1); periph=1-foveated
print(f"prior (gist) PSNR          : {psnr(prior,img):.2f} dB")
print(f"after {K} saccades, fixated : {psnr(committed,img,foveated):.2f} dB")
print(f"after {K} saccades, periphery: {psnr(committed,img,periph):.2f} dB  (stays gist)")

fig,ax=plt.subplots(1,4,figsize=(13,3.4))
ax[0].imshow(img,cmap="gray"); ax[0].set_title("retina (target)",fontsize=10)
ax[1].imshow(np.clip(prior,0,1),cmap="gray"); ax[1].set_title("prior: the gist\n(low-freq, held)",fontsize=10)
ax[2].imshow(np.clip(committed,0,1),cmap="gray")
fxs=[f[0]*(H-1) for f in fixations]; fys=[f[1]*(H-1) for f in fixations]
ax[2].plot(fxs,fys,'-o',color="#27e0c0",ms=5,lw=1.2,mec='k',mew=.4)
ax[2].plot(fxs[0],fys[0],'o',color="#27e0c0",ms=9,mec='k')
ax[2].set_xlim(0,H-1); ax[2].set_ylim(H-1,0)
ax[2].set_title(f"after {K} saccades\n(sharp where it looked)",fontsize=10)
ax[3].imshow(foveated,cmap="magma"); ax[3].set_title("where the fovea visited\n(the rest stays gist)",fontsize=10)
for a in ax: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig("saccade_demo.png",dpi=110,bbox_inches="tight")
print("saved saccade_demo.png")
