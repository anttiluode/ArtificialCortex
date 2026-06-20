"""
spectral_field.py — coupling a grid of skew-operator units into a moving wave
=============================================================================
THE HANDSHAKE: one rotation island per unit is a complex amplitude z (z_dot=i*w*z).
Couple a lattice with a complex-coefficient Laplacian (the ephaptic field) plus a
saturating nonlinearity -> the COMPLEX GINZBURG-LANDAU field:

    z_dot = (mu + i*w) z  -  (1 + i c)|z|^2 z  +  D(1 + i b) Lap(z)
                                                   ^^^^^^^^^^^^^^^^^ the handshake

Regime set by Benjamin-Feir-Newell number 1+bc: >0 -> coherent waves+islands;
<0 -> turbulence. Discriminators here: spatial correlation length (long=islands)
and defect density (phase singularities; many=turbulence). Probe cross-correlation
gives the traveling-wave speed.

GROUNDING (established, used not claimed): complex Ginzburg-Landau (Aranson &
Kramer 2002; Chate-Manneville); neural field theory (Wilson-Cowan; Amari;
Bressloff); cortical traveling waves (Muller et al. 2018); spreading depression
(Leao 1944); weak real ephaptic coupling (Jefferys 1995; Anastassiou 2011). The
coupling math is TEXTBOOK; the line's own part is the identification (held state =
skew lag-operator, spike = coincidence, ephaptic field = the diffusion term).
HONEST LIMITS: relative units, Euler, one seed/regime; "spike" reads as a phase
crossing, not an action potential. The bet (wave = FELT thought) stays in the drawer.
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def lap(Z):
    return (np.roll(Z,1,0)+np.roll(Z,-1,0)+np.roll(Z,1,1)+np.roll(Z,-1,1)-4*Z)

def run(n=128, steps=8000, dt=0.03, b=0.5, c=-0.5, D=1.0, mu=1.0, omega=0.0,
        seed=1, probes=None, rec_from=5000):
    rng = np.random.default_rng(seed)
    Z = 0.05*(rng.standard_normal((n,n))+1j*rng.standard_normal((n,n)))
    p1=[]; p2=[]
    for t in range(steps):
        Z = Z + dt*((mu+1j*omega)*Z - (1+1j*c)*(np.abs(Z)**2)*Z + D*(1+1j*b)*lap(Z))
        if probes is not None and t>=rec_from:
            a,bp = probes; p1.append(Z[a].real); p2.append(Z[bp].real)
    return Z, np.array(p1), np.array(p2)

def corr_length(Z):
    """radial spatial correlation length of the unit phasor field (cells to 1/e)."""
    n=Z.shape[0]; g=Z/(np.abs(Z)+1e-12)
    F=np.fft.fft2(g); ac=np.fft.ifft2(F*np.conj(F)).real/g.size
    ac=np.fft.fftshift(ac); ac/=ac.max()
    cy,cx=n//2,n//2; y,x=np.indices((n,n))
    r=np.sqrt((x-cx)**2+(y-cy)**2).astype(int)
    prof=np.bincount(r.ravel(),ac.ravel())/np.bincount(r.ravel())
    below=np.where(prof<np.exp(-1))[0]
    return float(below[0]) if len(below) else float(len(prof))

def defect_density(Z):
    """fraction of cells that are phase singularities (|z| near zero = spiral cores)."""
    a=np.abs(Z); return float((a < 0.25*np.median(a)).mean())

def wave_lag(p1,p2):
    x=p1-p1.mean(); y=p2-p2.mean(); x/=(x.std()+1e-12); y/=(y.std()+1e-12)
    n=len(x); cc=np.correlate(x,y,mode='full')/n; lags=np.arange(-n+1,n)
    k=int(np.argmax(np.abs(cc))); return int(lags[k]), float(cc[k])

if __name__=="__main__":
    n=128; d=n//3
    probes=((n//2, n//3),(n//2, 2*n//3))
    print("="*74)
    print("SPECTRAL FIELD — a grid of skew-units coupled into a moving wave")
    print("handshake = D(1+ib) Lap(z)   [the ephaptic diffusion term]")
    print("="*74)
    rows=[]
    for name,b,c in [("ORDERED",   0.5, -0.5),     # 1+bc = +0.75
                     ("TURBULENT", 2.0, -1.0)]:    # 1+bc = -1.00
        Z,p1,p2 = run(n=n,b=b,c=c,seed=1,probes=probes)
        bf=1+b*c; xi=corr_length(Z); dd=defect_density(Z); lag,corr=wave_lag(p1,p2)
        speed=d/abs(lag) if lag!=0 else float('nan')
        rows.append((name,bf,xi,dd,lag,corr,speed,Z))
        print(f"\n[{name}]  b={b}, c={c},  Benjamin-Feir 1+bc = {bf:+.2f}")
        print(f"   spatial correlation length (cells to 1/e): {xi:6.1f}")
        print(f"   defect density (phase singularities)     : {100*dd:6.2f}%")
        print(f"   probe cross-corr peak                    : {corr:+.2f} at lag {lag}")
        print(f"   implied traveling-wave speed (d/|lag|)   : {speed:6.3f} cells/step")
    print("\n"+"-"*74)
    o,t = rows[0],rows[1]
    print(f"correlation length  ORDERED {o[2]:.0f}  vs  TURBULENT {t[2]:.0f} cells")
    print(f"defect density      ORDERED {100*o[3]:.1f}% vs  TURBULENT {100*t[3]:.1f}%")
    print("="*74)

    fig,ax=plt.subplots(2,2,figsize=(9,9))
    for col,(name,bf,xi,dd,lag,corr,speed,Z) in enumerate(rows):
        ax[0,col].imshow(np.angle(Z),cmap="twilight"); ax[0,col].axis("off")
        ax[0,col].set_title(f"{name}  phase   (1+bc={bf:+.1f})",fontsize=11)
        ax[1,col].imshow(np.abs(Z),cmap="magma"); ax[1,col].axis("off")
        ax[1,col].set_title("amplitude |z|  (dark = spike cores / defects)",fontsize=10)
    plt.tight_layout(); plt.savefig("spectral_field.png",dpi=110,bbox_inches="tight")
    print("saved spectral_field.png")
