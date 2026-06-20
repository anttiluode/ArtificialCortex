"""
theta_gamma.py — the cortical stack's vertical wiring: slow gates fast (PAC)
============================================================================
THE STACK has a Z-axis of frequency bands. The question is how a slow band talks
to a fast one. The answer the line keeps arriving at (the chandelier on a theta
clock, from the HKT work): the SLOW theta wave is the coincidence-gate for the
FAST gamma field. Where the theta peak washes over a neighbourhood it disinhibits
it, and the fast gamma coincidences are allowed to fire there. The measurable
fingerprint of that arrangement is THETA-GAMMA PHASE-AMPLITUDE COUPLING (PAC) —
gamma amplitude peaking at a preferred theta phase.

PAC is the single most robust EEG signature of human memory and cognition
(Canolty & Knight 2010; Tort et al. 2010; Lisman & Jensen 2013, the theta-gamma
neural code). This script builds the gated arrangement and MEASURES the coupling
with Tort's Modulation Index, against a control with the gate removed.

HONEST SCOPE: this REPRODUCES the PAC signature from a gating architecture; it
does not prove cortex uses this mechanism (PAC has several candidate generators).
The gate here is a clean cosine of theta phase; real gating is messier. Relative
units, one synthetic recording. The MI separating gated from ungated is the claim.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

fs = 1000.0
def analytic(x):
    """analytic signal via FFT (Hilbert), no scipy needed."""
    X = np.fft.fft(x); N = len(x); h = np.zeros(N)
    h[0] = 1
    if N % 2 == 0: h[N//2] = 1; h[1:N//2] = 2
    else: h[1:(N+1)//2] = 2
    return np.fft.ifft(X*h)

def bandpass(x, lo, hi):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1/fs)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n=len(x))

def tort_MI(phase, amp, nbin=18):
    bins = np.linspace(-np.pi, np.pi, nbin+1)
    m = np.array([amp[(phase >= bins[i]) & (phase < bins[i+1])].mean()
                  for i in range(nbin)])
    P = m / m.sum()
    H = -np.sum(P*np.log(P + 1e-12))
    return (np.log(nbin) - H)/np.log(nbin), P, bins

def make_signal(gate=True, f_theta=6.0, f_gamma=50.0, T=80.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(T*fs))/fs
    theta = np.cos(2*np.pi*f_theta*t)
    if gate:
        env = (0.5*(1 + np.cos(2*np.pi*f_theta*t - 0.6)))**3   # gamma bursts near theta peak
    else:
        env = np.full_like(t, 0.4)                              # constant — no coupling
    gamma = env * np.cos(2*np.pi*f_gamma*t)
    return theta + 0.8*gamma + 0.3*rng.standard_normal(len(t))

if __name__ == "__main__":
    print("="*74)
    print("THE STACK — slow gates fast: theta-gamma phase-amplitude coupling")
    print("="*74)
    print("the slow theta wave is the coincidence-gate for the fast gamma field;")
    print("its fingerprint is gamma amplitude locked to theta phase (PAC).\n")

    out = {}
    for label, gate in [("gated  (theta opens the gate)", True),
                        ("control(no gate / constant)", False)]:
        sig = make_signal(gate=gate)
        ph = np.angle(analytic(bandpass(sig, 4, 8)))
        am = np.abs(analytic(bandpass(sig, 30, 80)))
        MI, P, bins = tort_MI(ph, am)
        out[gate] = (P, bins, MI)
        print(f"  {label:<32} Tort modulation index = {MI:.4f}")
    ratio = out[True][2] / (out[False][2] + 1e-9)
    print(f"\n  -> the gated stack shows ~{ratio:.0f}x the coupling of the ungated control.")
    print("     Gamma fires where the theta wave lets it — the slow band gating the")
    print("     fast one produces the brain's most robust cognition signature, here")
    print("     from architecture alone. Relative units; PAC reproduced, not proven.")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for gate, name, col in [(True, "gated", "#c0392b"), (False, "control", "#7f8c8d")]:
        P, bins, MI = out[gate]
        ctr = 0.5*(bins[:-1] + bins[1:])
        ax[0].bar(np.degrees(ctr) + (0 if gate else 0), P, width=16,
                  alpha=0.65 if gate else 0.45, color=col, label=f"{name} (MI={MI:.3f})")
    ax[0].set_xlabel("theta phase (deg)"); ax[0].set_ylabel("norm. gamma amplitude")
    ax[0].set_title("gamma amplitude vs theta phase"); ax[0].legend(fontsize=9)
    sig = make_signal(gate=True)
    t = np.arange(2000)/fs
    ax[1].plot(t, sig[:2000], color="#2c3e50", lw=0.7, label="raw (theta+gamma)")
    ax[1].plot(t, bandpass(sig, 4, 8)[:2000], color="#2980b9", lw=1.6, label="theta")
    ax[1].plot(t, np.abs(analytic(bandpass(sig, 30, 80)))[:2000], color="#c0392b",
               lw=1.4, label="gamma envelope")
    ax[1].set_xlabel("time (s)"); ax[1].set_title("gamma bursts ride the theta wave")
    ax[1].legend(fontsize=8, loc="upper right")
    plt.tight_layout(); plt.savefig("theta_gamma.png", dpi=110, bbox_inches="tight")
    print("\n  saved theta_gamma.png"); print("="*74)
