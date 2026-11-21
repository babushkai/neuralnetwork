#%% md
# 2026 — Mamba-3: Trapezoidal Discretization, Complex States, MIMO

**Paper:** Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu, *"Mamba-3: Improved Sequence Modeling using State Space Principles"* (arXiv 2603.15569, March 2026; ICLR 2026 oral).

Mamba (notebook 39) and Mamba-2 update a hidden state with a real, decaying recurrence $h_t = e^{\Delta_t A_t} h_{t-1} + \Delta_t B_t x_t$ (the "exponential-Euler" discretization of a continuous ODE). Mamba-3 changes three things, all derived from classical state-space theory:

1. **Exponential-trapezoidal discretization.** The input contribution averages the current *and previous* step:
$$h_t = \alpha_t h_{t-1} + \beta_t B_{t-1}x_{t-1} + \gamma_t B_t x_t,\qquad \alpha_t = e^{\Delta_t A_t},\ \ \beta_t = (1-\lambda_t)\Delta_t e^{\Delta_t A_t},\ \ \gamma_t = \lambda_t\Delta_t$$
with a data-dependent $\lambda_t\in[0,1]$. $\lambda=1$ gives back Mamba-2; $\lambda=\tfrac12$ is the classical trapezoid rule. It acts like a width-2 convolution *inside* the recurrence, and replaces the separate short convolution Mamba-2 needed.
2. **Complex-valued state = data-dependent RoPE.** A complex eigenvalue $A + i\theta$ makes the state *rotate* as well as decay. Rotations are exactly what state tracking needs (parity is "rotate by $\pi$ on every 1"), and real decays in $[0,1]$ cannot do it. The paper's **RoPE trick** (Prop. 3–4) pushes the cumulative rotation into $B$ and $C$: with $\phi_t = \sum_{s\le t}\Delta_s\theta_s$, use $\bar B_t = R(-\phi_t)B_t$ and $\bar C_t = R(-\phi_t)C_t$, then run an ordinary real recurrence. It is RoPE (notebook 53), with angles the model chooses from its input.
3. **MIMO.** Instead of a rank-1 update $B_t x_t^\top$, use rank $R$: $H_t = \alpha_t H_{t-1} + B_t X_t^\top$ with $B_t\in\mathbb R^{N\times R}$, $X_t\in\mathbb R^{P\times R}$. More FLOPs per byte of state, which decoding has spare (it is memory-bound), so quality improves at about the same latency.

Also: RMSNorm on $B$ and $C$ ("BCNorm", like QK-norm) plus learnable channel-wise biases, and **no short convolution**.

**Paper results (100B tokens):** at 1.5B, perplexity 10.47 (Mamba-2) → 10.35 (Mamba-3 SISO) → 10.24 (MIMO, R=4); average accuracy +0.6 points over Gated DeltaNet for SISO and +1.8 for MIMO. On parity, Mamba-3 scores 100% and Mamba-2 0.9% (scaled accuracy, where 0 = chance).

**Our choices where the text we read is not explicit:** how $\theta_t$ and $\lambda_t$ are projected (linear maps; $\lambda=\sigma(\cdot)$ as the paper's default), the shapes of the B/C biases, and how MIMO's $R$ outputs are combined (a learned per-channel weighting).
#%%
{{SETUP}}
#%% md
## 1. The recurrence, and an exact parallel form
For training we use the "attention-like" (SSD) form. With $G_t = \sum_{s\le t}\log\alpha_s$, the trapezoidal recurrence unrolls to
$$y_t = \sum_{s\le t} W_{ts}\,(\bar C_t^\top \bar B_s)\,x_s,\qquad W_{ts} = \gamma_s e^{G_t-G_s} + [s<t]\,\beta_{s+1}e^{G_t-G_{s+1}}$$
i.e. the paper's Eq. 7: a decay mask times a 2-band matrix. Real kernels compute this chunk by chunk; for short toy sequences the full $T\times T$ form is fine. First check that it matches the token-by-token recurrence, including rotations and MIMO.
#%%
def rotate(x, phi):
    """Rotate consecutive channel pairs of x (..., N, R) by angles phi (..., N/2)."""
    x1, x2 = x[..., 0::2, :], x[..., 1::2, :]
    c, s = phi.cos()[..., None], phi.sin()[..., None]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -2).flatten(-3, -2)

def ssm_recurrent(X, Bm, Cm, log_a, beta, gamma, phi):
    """X: (B,T,H,P,R)  Bm,Cm: (B,T,H,N,R)  log_a,beta,gamma: (B,T,H)  phi: (B,T,H,N/2). Returns Y: (B,T,H,P,R)."""
    Bb, Cb = rotate(Bm, -phi), rotate(Cm, -phi)                 # the RoPE trick
    Bsz, T, H, P, R = X.shape; N = Bm.shape[3]
    h = X.new_zeros(Bsz, H, N, P); prev = X.new_zeros(Bsz, H, N, P); ys = []
    for t in range(T):
        u = torch.einsum("bhnr,bhpr->bhnp", Bb[:, t], X[:, t])       # B_t X_t^T  (rank-R update)
        h = torch.exp(log_a[:, t])[..., None, None] * h + beta[:, t][..., None, None] * prev + gamma[:, t][..., None, None] * u
        prev = u
        ys.append(torch.einsum("bhnr,bhnp->bhpr", Cb[:, t], h))     # Y_t = H_t^T C_t
    return torch.stack(ys, 1)

def ssm_parallel(X, Bm, Cm, log_a, beta, gamma, phi):
    Bb, Cb = rotate(Bm, -phi), rotate(Cm, -phi)
    T = X.shape[1]
    G = torch.cumsum(log_a, 1).transpose(1, 2)                     # (B,H,T)
    diff = G[..., :, None] - G[..., None, :]                         # G_t - G_s
    causal = torch.ones(T, T, dtype=torch.bool, device=X.device).tril()
    E = torch.exp(diff.masked_fill(~causal, -float("inf")))          # e^{G_t-G_s} for s<=t
    g, b = gamma.transpose(1, 2), beta.transpose(1, 2)
    E_next = torch.cat([E[..., 1:], torch.zeros_like(E[..., :1])], -1)                     # e^{G_t-G_{s+1}} (column s+1)
    b_next = torch.cat([b[..., 1:], torch.zeros_like(b[..., :1])], -1)
    strict = torch.ones(T, T, dtype=torch.bool, device=X.device).tril(-1)
    W = E * g[..., None, :] + (E_next * b_next[..., None, :]).masked_fill(~strict, 0)      # (B,H,T,S)
    Bsz, _, H, N, R = Bb.shape; P = X.shape[3]
    Ct = Cb.permute(0, 2, 1, 4, 3).reshape(Bsz * H, T * R, N)        # rows (t, r)
    Bs_ = Bb.permute(0, 2, 1, 4, 3).reshape(Bsz * H, T * R, N)       # rows (s, q)
    M = (Ct @ Bs_.transpose(1, 2)).view(Bsz, H, T, R, T, R)          # C̄_t^T B̄_s  (R x R per pair)
    A = M * W[:, :, :, None, :, None]                                # weight by the decay/trapezoid mask
    Xs = X.permute(0, 2, 1, 4, 3).reshape(Bsz * H, T * R, P)         # rows (s, q)
    Y = A.view(Bsz * H, T * R, T * R) @ Xs                           # one batched matmul
    return Y.view(Bsz, H, T, R, P).permute(0, 2, 1, 4, 3)            # (B,T,H,P,R)

Bs, T, H, P, N, R = 2, 40, 3, 4, 8, 2
X = torch.randn(Bs, T, H, P, R, device=device); Bm, Cm = torch.randn(2, Bs, T, H, N, R, device=device)
log_a = -torch.rand(Bs, T, H, device=device) * 0.5; dt = torch.rand(Bs, T, H, device=device); lam = torch.rand(Bs, T, H, device=device)
beta, gamma = (1 - lam) * dt * torch.exp(log_a), lam * dt
phi = torch.cumsum(torch.randn(Bs, T, H, N // 2, device=device), 1)
print(f"max |parallel - recurrent| = {(ssm_parallel(X, Bm, Cm, log_a, beta, gamma, phi) - ssm_recurrent(X, Bm, Cm, log_a, beta, gamma, phi)).abs().max().item():.2e}")
#%% md
## 2. The Mamba-3 layer
One class with switches, so we can ablate: `trap` (trapezoid vs Euler), `rope` (complex/rotating state vs real), `R` (MIMO rank), and `conv` (Mamba-2's short convolution). Mamba-2-style = `trap=False, rope=False, R=1, conv=True`.
#%%
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6): super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x): return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class Mamba3(nn.Module):
    def __init__(self, d, heads=4, N=16, R=1, trap=True, rope=True, conv=False):
        super().__init__()
        self.H, self.P, self.N, self.R, self.trap, self.rope, self.conv = heads, d // heads, N, R, trap, rope, conv
        self.in_x, self.in_z = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.in_B, self.in_C = nn.Linear(d, N * R, bias=False), nn.Linear(d, N * R, bias=False)   # shared across heads
        self.in_dt, self.in_A, self.in_lam = nn.Linear(d, heads), nn.Linear(d, heads), nn.Linear(d, heads)
        self.in_theta = nn.Linear(d, heads * N // 2)
        self.bnorm, self.cnorm = RMSNorm(N), RMSNorm(N)                                            # BCNorm
        self.b_bias, self.c_bias = nn.Parameter(torch.zeros(heads, N, R)), nn.Parameter(torch.zeros(heads, N, R))
        self.wx, self.wy = nn.Parameter(torch.ones(self.P, R)), nn.Parameter(torch.ones(self.P, R) / R)  # MIMO expand / combine
        self.D = nn.Parameter(torch.ones(heads))
        if conv: self.cv = nn.Conv1d(d, d, 4, groups=d, padding=3)
        with torch.no_grad():
            self.in_dt.bias.copy_(torch.log(torch.expm1(torch.empty(heads).uniform_(1e-3, 0.1))))
            self.in_A.bias.fill_(0.5)
        self.onorm, self.out = RMSNorm(d), nn.Linear(d, d, bias=False)

    def coeffs(self, u):
        Bsz, T, _ = u.shape
        dt = F.softplus(self.in_dt(u))                                   # (B,T,H)
        A = -F.softplus(self.in_A(u))                                    # data-dependent, negative
        log_a = dt * A
        lam = torch.sigmoid(self.in_lam(u)) if self.trap else torch.ones_like(dt)
        beta, gamma = (1 - lam) * dt * torch.exp(log_a), lam * dt
        if self.rope:
            theta = self.in_theta(u).view(Bsz, T, self.H, self.N // 2)
            phi = torch.cumsum(dt[..., None] * theta, 1)                 # cumulative angle per head and channel pair
        else:
            phi = u.new_zeros(Bsz, T, self.H, self.N // 2)
        return log_a, beta, gamma, phi

    def forward(self, u, recurrent=False):
        Bsz, T, D = u.shape
        x = self.in_x(u)
        if self.conv: x = F.silu(self.cv(x.transpose(1, 2))[..., :T].transpose(1, 2))
        Bm = self.bnorm(self.in_B(u).view(Bsz, T, 1, self.N, self.R).transpose(-1, -2)).transpose(-1, -2) + self.b_bias
        Cm = self.cnorm(self.in_C(u).view(Bsz, T, 1, self.N, self.R).transpose(-1, -2)).transpose(-1, -2) + self.c_bias
        log_a, beta, gamma, phi = self.coeffs(u)
        xh = x.view(Bsz, T, self.H, self.P)
        X = xh[..., None] * self.wx                                       # (B,T,H,P,R)
        Y = (ssm_recurrent if recurrent else ssm_parallel)(X, Bm, Cm, log_a, beta, gamma, phi)
        y = (Y * self.wy).sum(-1) + self.D[:, None] * xh
        y = y.reshape(Bsz, T, D) * F.silu(self.in_z(u))
        return self.out(self.onorm(y))

class LM(nn.Module):
    def __init__(self, V, d=64, n_layers=2, **kw):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.mix = nn.ModuleList([Mamba3(d, **kw) for _ in range(n_layers)])
        self.mlp = nn.ModuleList([nn.Sequential(RMSNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(n_layers)])
        self.n = nn.ModuleList([RMSNorm(d) for _ in range(n_layers)]); self.fn = RMSNorm(d); self.head = nn.Linear(d, V)
    def forward(self, x, recurrent=False):
        h = self.emb(x)
        for m, f, n in zip(self.mix, self.mlp, self.n): h = h + m(n(h), recurrent); h = h + f(h)
        return self.head(self.fn(h))

m = LM(10, R=2).to(device); xx = torch.randint(0, 10, (2, 24), device=device)
with torch.no_grad(): print(f"full layer, parallel vs recurrent: {(m(xx) - m(xx, recurrent=True)).abs().max().item():.2e}")
#%% md
## 3. State tracking: parity
Input: a start token then a random bit string. Target at every position: the parity (XOR) of the bits so far. This is the paper's headline synthetic task. A state that can only *decay* can at best "add everything up, then take mod 2", and the mod-2 step has to be memorized by the MLP for every count it has seen. A state that *rotates* by $\pi$ on each 1 tracks parity exactly, at any length. We train on lengths 16–64 and test on 64 and on 256, 4× longer than anything seen in training.
#%%
def parity_batch(B, T):
    bits = torch.randint(0, 2, (B, T), device=device)
    x = torch.cat([torch.full((B, 1), 2, device=device), bits], 1)
    return x, torch.cumsum(bits, 1) % 2

def train_parity(steps=1500, **kw):
    seed_everything(0); m = LM(3, **kw).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps, pct_start=0.1)
    for s in range(steps):
        x, y = parity_batch(64, random.randint(16, 64)); loss = F.cross_entropy(m(x)[:, 1:, :2].reshape(-1, 2), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    m.eval(); res = []
    with torch.no_grad():
        for T in (64, 256):
            correct = []
            for _ in range(10):                                            # 500 strings, in chunks (the T×T parallel form is memory-hungry)
                x, y = parity_batch(50, T); correct.append((m(x)[:, 1:, :2].argmax(-1) == y).float()[:, -1])   # parity of the whole string
            res.append(torch.cat(correct).mean().item())
    return res

variants = [("Mamba-2 style (Euler, real, short conv)", dict(trap=False, rope=False, conv=True)),
            ("Mamba-3 without complex state (no RoPE)", dict(trap=True, rope=False)),
            ("Mamba-3 (trapezoid + complex state)", dict(trap=True, rope=True)),
            ("Mamba-3 MIMO R=4", dict(trap=True, rope=True, R=4))]
for name, kw in variants:
    t0 = time.time(); a32, a128 = train_parity(**kw)
    print(f"{name:42s} parity of the whole string: T=64 {a32:.3f} | T=256 (4x longer) {a128:.3f}   ({time.time() - t0:.0f}s)")
#%% md
Chance is 0.5. Both real-valued models stay at chance even on lengths they trained on: with lengths varying from 16 to 64, "count the 1s, then memorize which counts are odd" would need the MLP to memorize parity for every count up to 64 from a decaying, blurry count, and it doesn't. (Trained on one fixed length, a real-valued model *can* partly memorize it; that is why we vary the length.) The complex (rotating) state solves the task in distribution and mostly keeps solving it at 4× the training length, because what it learned is a rotation, not a lookup. Its remaining errors at 256 come from the learned angle and decay being only approximately $\pi$ and 1, so small errors accumulate over very long strings. Gated DeltaNet can also do this if its eigenvalues are allowed to reach −1 (the paper's "GDN [−1,1]" row), another way to get a sign flip.
#%% md
## 4. What did the state learn? The rotation angles
For parity, the ideal per-step angle is $\Delta\theta = \pi$ (mod $2\pi$) on a 1 and $0$ on a 0. Measure the angle increment each bit causes in the trained model's first layer, for the channel pair that separates them most.
#%%
seed_everything(0); m = LM(3, trap=True, rope=True).to(device)
opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
for s in range(1500):
    x, y = parity_batch(64, random.randint(16, 64)); loss = F.cross_entropy(m(x)[:, 1:, :2].reshape(-1, 2), y.reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
wrap = lambda a: torch.remainder(a + math.pi, 2 * math.pi) - math.pi
best = None
with torch.no_grad():
    x, _ = parity_batch(256, 32); h = m.emb(x); bit = x[:, 1:]
    for li, (layer, n, f) in enumerate(zip(m.mix, m.n, m.mlp)):
        _, _, _, phi = layer.coeffs(n(h))
        inc = torch.diff(phi, dim=1)                                       # angle added by token t (t = 1..T)
        one, zero = inc[bit == 1], inc[bit == 0]                           # (n, H, N/2)
        sep = wrap(one.mean(0) - zero.mean(0)).abs()
        if best is None or sep.max() > best[0]:
            h_i, p_i = divmod(sep.argmax().item(), sep.shape[1])
            best = (sep.max(), li, h_i, p_i, wrap(one[:, h_i, p_i] - zero[:, h_i, p_i].mean()), zero[:, h_i, p_i] - zero[:, h_i, p_i].mean())
        h = h + layer(n(h)); h = h + f(h)
_, li, h_i, p_i, d1, d0 = best
print(f"best channel pair (layer {li}, head {h_i}, pair {p_i}): a 1 adds {d1.abs().mean().item() / math.pi:.2f}π relative to a 0 (ideal 1.00π); spread {d1.std().item() / math.pi:.2f}π")
plt.figure(figsize=(7, 3)); plt.hist(d0.cpu() / math.pi, 50, alpha=0.6, label="bit 0"); plt.hist(d1.cpu() / math.pi, 50, alpha=0.6, label="bit 1")
plt.xlabel("rotation added this step (units of π), relative to a 0"); plt.legend(); plt.title("Mamba-3 learned to rotate by π on a 1"); plt.show()
#%% md
## 5. Language modelling: Mamba-2 vs Mamba-3 SISO vs MIMO
A small character LM on the procedural corpus, 4 layers, same width. MIMO adds a few parameters (the rank-R B/C projections); the paper parameter-matches by shrinking the MLP, which we skip at this scale.
#%%
{{PROCEDURAL}}
#%%
def lm_batch(src, B=32, T=128):
    ix = torch.randint(0, len(src) - T - 1, (B,), device=device)
    return src[ix[:, None] + torch.arange(T, device=src.device)], src[ix[:, None] + torch.arange(1, T + 1, device=src.device)]

def train_lm(steps=600, **kw):
    seed_everything(0); m = LM(V, d=96, n_layers=4, heads=4, **kw).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps, pct_start=0.1); t0 = time.time()
    for s in range(steps):
        x, y = lm_batch(train_data); loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    m.eval(); g = torch.Generator(device="cpu").manual_seed(1)
    with torch.no_grad():
        vals = []
        for _ in range(20):
            ix = torch.randint(0, len(val_data) - 129, (32,), generator=g).to(device)
            x = val_data[ix[:, None] + torch.arange(128, device=val_data.device)]; y = val_data[ix[:, None] + torch.arange(1, 129, device=val_data.device)]
            vals.append(F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)).item())
    return np.mean(vals), sum(p.numel() for p in m.parameters()), time.time() - t0

for name, kw in [("Mamba-2 style", dict(trap=False, rope=False, conv=True)),
                 ("Mamba-3 SISO", dict(trap=True, rope=True)),
                 ("Mamba-3 MIMO R=4", dict(trap=True, rope=True, R=4))]:
    vl, n_p, sec = train_lm(**kw)
    print(f"{name:18s} val loss {vl:.4f} | {n_p / 1e3:.0f}K params | {sec:.0f}s")
#%% md
## Takeaways
- **Trapezoid:** a principled, slightly better discretization. It gives the recurrence a built-in 2-tap convolution, so the extra short-conv layer can go.
- **Complex state:** rotation plus decay gives the state a sign and phase, so it can track state (parity, modular counting) where real decays cannot. Implemented for free as data-dependent RoPE on B and C.
- **MIMO:** a rank-R update raises arithmetic intensity, using compute that decoding leaves idle, for better quality at similar latency.
- At this toy scale the LM differences are small (the paper's gains are ~0.1 perplexity at 1.5B parameters and 100B tokens). The parity result is the clear, qualitative difference.
- Where it fits: Mamba-2 layers power NVIDIA's Nemotron 3 hybrids (notebook 62) and many other 2026 hybrids; Mamba-3 is the state-space line's answer to Gated DeltaNet / KDA (notebooks 49, 59, 63).
