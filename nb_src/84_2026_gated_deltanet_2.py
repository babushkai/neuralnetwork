#%% md
# 2026 — Gated DeltaNet-2: Decoupling Erase and Write

**Paper:** Hatamizadeh, Choi, Kautz (NVIDIA), *"Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"* (arXiv 2605.22791, May 2026). Code: NVlabs/GatedDeltaNet-2.

The delta rule (notebooks 70, 80) updates a memory matrix $S$ by **erasing** what a key currently retrieves and **writing** a new value, both scaled by **one scalar** $\beta_t$:
$$\text{KDA:}\quad S_t = (I - \beta_t k_t k_t^\top)\,D_t S_{t-1} + \beta_t k_t v_t^\top,\qquad D_t = \operatorname{Diag}(\alpha_t)$$
So a layer cannot erase strongly while writing weakly, or the reverse; the paper calls this the **scalar tie**. Gated DeltaNet-2 replaces $\beta_t$ with two **channel-wise** gates, one on each axis of the state:
$$S_t = \big(I - k_t\,(b_t\odot k_t)^\top\big)\,D_t S_{t-1} + k_t\,(w_t\odot v_t)^\top,\qquad b_t = \sigma(W_b x_t)\in[0,1]^{d_k},\quad w_t = \sigma(W_w x_t)\in[0,1]^{d_v}$$

- **Erase gate** $b_t$ (key axis): which key channels the read-before-write uses, i.e. *how much and where* old content is removed.
- **Write gate** $w_t$ (value axis): which value channels of the new content get written.
- **Special cases:** $b_t=w_t=\beta_t\mathbf 1$ gives KDA; also tying $\alpha_t$ to one scalar gives Gated DeltaNet.

**Paper results (1.3B, 100B tokens):** best average among Mamba-2, GDN, KDA, Mamba-3 SISO/MIMO (recurrent 53.11 vs KDA 52.28; hybrid 53.97 vs 52.68), Wiki perplexity 15.90 vs 16.81 (KDA), and much better multi-key retrieval (MK-NIAH-1 @1K: 72.6 vs 54.0). The gate ablation shows **most of the gain comes from the channel-wise erase gate**.
#%% md
![Diagram: Gated DeltaNet-2 (NVIDIA, 2026)](https://raw.githubusercontent.com/babushkai/neuralnetwork/main/assets/diagrams/84_2026_gated_deltanet_2.png)
#%%
{{SETUP}}
#%% md
## 1. Recurrence and chunkwise algorithm
Split form (paper Eq. 9): $\bar S = D_tS_{t-1}$, $r_t = \bar S^\top e_t$ with $e_t = b_t\odot k_t$, then $S_t = \bar S + k_t(z_t - r_t)^\top$ with $z_t = w_t\odot v_t$.

Chunkwise, exactly as in notebook 80 but with the two gates placed **inside** the triangular system (the paper's point: they cannot be pulled out as a scalar). Writing $S_t = \operatorname{Diag}(\gamma_t)S_0 + \sum_{i\le t}\operatorname{Diag}(\gamma_t/\gamma_i)k_iu_i^\top$, the corrections $u$ of a chunk solve
$$\big(I + \operatorname{StrictTril}[\,e_t^\top \operatorname{Diag}(\gamma_t/\gamma_i)\,k_i\,]\big)\,U = Z - (\gamma\odot E)\,S_0$$
and $O = (\gamma\odot Q)S_0 + \operatorname{Tril}[\,q_t^\top\operatorname{Diag}(\gamma_t/\gamma_i)k_i\,]\,U$. All ratios $\gamma_t/\gamma_i$ are computed as $\exp(G_t-G_i)\le 1$ in log space.
#%%
def gdn2_recurrent(q, k, v, alpha, b, w):
    """q,k,alpha,b: (B,H,T,dk)  v,w: (B,H,T,dv). Token-by-token reference (paper Eq. 9)."""
    B, H, T, dk = k.shape; S = k.new_zeros(B, H, dk, v.shape[-1]); out = []
    for t in range(T):
        Sb = alpha[:, :, t, :, None] * S                                       # D_t S_{t-1}
        r = torch.einsum("bhk,bhkv->bhv", b[:, :, t] * k[:, :, t], Sb)         # read with the erase direction e_t
        S = Sb + k[:, :, t, :, None] * (w[:, :, t] * v[:, :, t] - r)[..., None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))
    return torch.stack(out, 2)

def gdn2_chunked(q, k, v, log_alpha, b, w, C=16):
    B, H, T, dk = k.shape; dv = v.shape[-1]
    S = k.new_zeros(B, H, dk, dv); outs = []
    for s in range(0, T, C):
        qc, kc, vc, lc, bc, wc = (x[:, :, s:s + C] for x in (q, k, v, log_alpha, b, w))
        n = kc.shape[2]; ec, zc = bc * kc, wc * vc
        G = torch.cumsum(lc, 2)
        causal = torch.ones(n, n, dtype=torch.bool, device=k.device).tril()
        E = torch.exp((G[:, :, :, None, :] - G[:, :, None, :, :]).masked_fill(~causal[None, None, :, :, None], float("-inf")))
        EK = torch.einsum("bhtc,bhic,bhtic->bhti", ec, kc, E)
        QK = torch.einsum("bhtc,bhic,bhtic->bhti", qc, kc, E)
        g = torch.exp(G)
        U = torch.linalg.solve_triangular(torch.eye(n, device=k.device) + torch.tril(EK, -1), zc - (g * ec) @ S, upper=False, unitriangular=True)
        outs.append((g * qc) @ S + QK @ U)
        S = torch.exp(G[:, :, -1, :, None]) * S + (torch.exp(G[:, :, -1:, :] - G) * kc).transpose(-1, -2) @ U
    return torch.cat(outs, 2)

B, H, T, dk, dv = 2, 3, 80, 16, 16
q, k = (F.normalize(torch.randn(B, H, T, dk, device=device), dim=-1) for _ in range(2))
v = torch.randn(B, H, T, dv, device=device)
alpha = torch.exp(-0.5 * torch.rand(B, H, T, dk, device=device))
bg, wg = torch.rand(B, H, T, dk, device=device), torch.rand(B, H, T, dv, device=device)
print(f"GDN-2 chunked vs recurrent: max diff {(gdn2_chunked(q, k, v, alpha.log(), bg, wg) - gdn2_recurrent(q, k, v, alpha, bg, wg)).abs().max().item():.2e}")
beta = torch.rand(B, H, T, 1, device=device)
print(f"KDA special case (b = w = β·1): max diff {(gdn2_chunked(q, k, v, alpha.log(), beta.expand(-1, -1, -1, dk), beta.expand(-1, -1, -1, dv)) - gdn2_recurrent(q, k, v, alpha, beta.expand(-1, -1, -1, dk), beta.expand(-1, -1, -1, dv))).abs().max().item():.2e}")
#%% md
## 2. One layer class for GDN, KDA, GDN-2 and the ablations
Projections as in the paper (Sec. 3.5): q, k, v through a short causal conv and SiLU, L2-normalized q and k, decay $g_t = -e^{a}\odot\operatorname{softplus}(W_fx_t+\delta)$, an RMS-normalized output with a SiLU output gate. `decay` ∈ {scalar, channel}; `erase`/`write` ∈ {scalar, channel}.
#%%
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6): super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x): return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class ShortConv(nn.Module):
    def __init__(self, d, k=4): super().__init__(); self.c = nn.Conv1d(d, d, k, groups=d, padding=k - 1)
    def forward(self, x): return self.c(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)

class DeltaLayer(nn.Module):
    def __init__(self, d, heads=2, decay="channel", erase="channel", write="channel"):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.decay, self.erase, self.write = decay, erase, write
        self.wq, self.wk, self.wv = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.cq, self.ck, self.cv = ShortConv(d), ShortConv(d), ShortConv(d)
        self.w_b = nn.Linear(d, d if erase == "channel" else heads)
        self.w_w = nn.Linear(d, d if write == "channel" else heads) if not (erase == write == "scalar") else None   # tied scalar = one β
        na = d if decay == "channel" else heads
        self.w_f = nn.Linear(d, na); self.a = nn.Parameter(torch.log(torch.empty(na).uniform_(1, 16)))
        with torch.no_grad(): self.w_f.bias.copy_(torch.log(torch.expm1(torch.empty(na).uniform_(1e-3, 0.1))))
        self.gate, self.onorm, self.wo = nn.Linear(d, d, bias=False), RMSNorm(self.dk), nn.Linear(d, d, bias=False)
    def _expand(self, t, kind, B, T):
        return t.view(B, T, self.h, self.dk).transpose(1, 2) if kind == "channel" else t.transpose(1, 2)[..., None].expand(B, self.h, T, self.dk)
    def forward(self, x):
        B, T, D = x.shape
        sh = lambda t: t.view(B, T, self.h, self.dk).transpose(1, 2)
        q = F.normalize(F.silu(sh(self.cq(self.wq(x)))), dim=-1); k = F.normalize(F.silu(sh(self.ck(self.wk(x)))), dim=-1)
        v = F.silu(sh(self.cv(self.wv(x))))
        g = -torch.exp(self.a) * F.softplus(self.w_f(x))
        log_alpha = self._expand(g, self.decay, B, T)
        b = self._expand(torch.sigmoid(self.w_b(x)), self.erase, B, T)
        w = b if self.w_w is None else self._expand(torch.sigmoid(self.w_w(x)), self.write, B, T)
        o = self.onorm(gdn2_chunked(q, k, v, log_alpha, b, w)).transpose(1, 2).reshape(B, T, D)
        return self.wo(F.silu(self.gate(x)) * o)

class Model(nn.Module):
    def __init__(self, V, d=64, n_layers=2, **kw):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.mix = nn.ModuleList([DeltaLayer(d, **kw) for _ in range(n_layers)])
        self.mlps = nn.ModuleList([nn.Sequential(RMSNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(n_layers)])
        self.n = nn.ModuleList([RMSNorm(d) for _ in range(n_layers)]); self.head = nn.Sequential(RMSNorm(d), nn.Linear(d, V))
    def forward(self, x):
        h = self.emb(x)
        for m, f, n in zip(self.mix, self.mlps, self.n): h = h + m(n(h)); h = h + f(h)
        return self.head(h)

VARIANTS = {
    "Gated DeltaNet (scalar decay, tied β)":        dict(decay="scalar", erase="scalar", write="scalar"),
    "KDA (channel decay, tied β)":                  dict(decay="channel", erase="scalar", write="scalar"),
    "ablation: scalar erase, channel write":        dict(decay="channel", erase="scalar", write="channel"),
    "ablation: channel erase, scalar write":        dict(decay="channel", erase="channel", write="scalar"),
    "Gated DeltaNet-2 (channel erase + write)":     dict(decay="channel", erase="channel", write="channel"),
}
#%% md
## 3. Multi-key recall under memory pressure, with deletions
A purely recurrent model (no attention layer, so the state must do all the work). The stream contains `SET key value` and `DEL key` commands, then queries. The answer is the key's latest value, or `NONE` if it was deleted after its last SET. Several keys share a small state ($d_k=16$ per head for 12 keys), so the layer must erase precisely to avoid interference. This is the regime where the paper's multi-key needle-in-a-haystack gains appeared.
#%%
NK, NV, NW, NQ = 24, 24, 24, 8
SET, DEL, QRY = NK + NV, NK + NV + 1, NK + NV + 2
NONE = NK + NV + 3; Vt = NONE + 1
def recall_batch(B):
    keys = torch.rand(B, NK).argsort(1)[:, :12]                                   # 12 keys in play
    ck = keys.gather(1, torch.randint(0, 12, (B, NW)))
    is_del = torch.rand(B, NW) < 0.25
    vals = torch.randint(NK, NK + NV, (B, NW))
    state = torch.full((B, NK), NONE, dtype=torch.long)
    rows = []
    for t in range(NW):
        bi = torch.arange(B)
        state[bi, ck[:, t]] = torch.where(is_del[:, t], torch.full((B,), NONE), vals[:, t])
        rows.append(torch.where(is_del[:, t, None], torch.stack([torch.full((B,), DEL), ck[:, t], torch.full((B,), DEL)], 1),
                                torch.stack([torch.full((B,), SET), ck[:, t], vals[:, t]], 1)))
    qk = keys.gather(1, torch.rand(B, 12).argsort(1)[:, :NQ])
    target = state.gather(1, qk)
    x = torch.cat(rows + [torch.stack([torch.full((B, NQ), QRY), qk], 2).view(B, 2 * NQ)], 1)
    return x.to(device), target.to(device)
q_pos = torch.arange(3 * NW + 1, 3 * NW + 2 * NQ, 2, device=device)
x, y = recall_batch(2); print("sequence length", x.shape[1], "| fraction of NONE answers", (recall_batch(2000)[1] == NONE).float().mean().item())

def train_eval(kw, steps=1500, lr=3e-3, B=64, seed=0):
    seed_everything(seed); m = Model(Vt, **kw).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    for s in range(steps):
        x, y = recall_batch(B); loss = F.cross_entropy(m(x)[:, q_pos].reshape(-1, Vt), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    m.eval()
    with torch.no_grad():
        x, y = recall_batch(2000); pred = m(x)[:, q_pos].argmax(-1)
    return m, (pred == y).float().mean().item(), (pred == y)[y == NONE].float().mean().item(), (pred == y)[y != NONE].float().mean().item()

res = {}
for name, kw in VARIANTS.items():
    accs = []
    for seed in (0, 1):
        t0 = time.time(); m, acc, acc_none, acc_val = train_eval(kw, seed=seed); accs.append((acc, acc_none, acc_val))
    a = np.mean(accs, 0); res[name] = (m, a)
    print(f"{name:42s} recall {a[0]:.3f} (deleted keys {a[1]:.3f}, live keys {a[2]:.3f})  [2 seeds, last {time.time() - t0:.0f}s]")
#%% md
## 4. What do the decoupled gates do?
In the trained GDN-2 model, compare the mean erase gate $\bar b$ and write gate $\bar w$ on `SET` tokens vs. `DEL` tokens (at the command token and the key token that follows). A decoupled layer can erase without writing.
#%%
m = res["Gated DeltaNet-2 (channel erase + write)"][0]; layer = m.mix[0]
x, _ = recall_batch(500)
with torch.no_grad():
    hin = m.n[0](m.emb(x))
    bgate, wgate = torch.sigmoid(layer.w_b(hin)), torch.sigmoid(layer.w_w(hin))
cmd = x[:, : 3 * NW : 3]; key_pos = torch.arange(1, 3 * NW, 3, device=device)
for label, mask in [("SET", cmd == SET), ("DEL", cmd == DEL)]:
    bk, wk = bgate[:, key_pos][mask], wgate[:, key_pos][mask]
    print(f"at the key token of a {label}: mean erase gate b = {bk.mean():.3f}, mean write gate w = {wk.mean():.3f}, mean |b - w| per channel = {(bk - wk).abs().mean():.3f}")
#%% md
## Takeaways
- KDA ties how much to erase and how much to write to one scalar per token. GDN-2 gives each its own **channel-wise** gate: erase on the key axis, write on the value axis. KDA and GDN are special cases.
- The chunkwise algorithm survives: the gates move inside the triangular solve (they cannot be applied as a post-hoc scalar), and decays stay in log space.
- Read the recall table with care: two seeds of a 2-layer toy model. In the paper the clearest gains are in multi-key retrieval and are driven mainly by the **erase** gate. Check whether that ordering shows up above; at this scale the variants may be within noise.
- Lineage: DeltaNet (2024) → Gated DeltaNet (2025, Qwen3-Next/3.5) → KDA (Kimi Linear, K3) → GDN-2 (May 2026). The same "make the scalar a vector" move made KDA out of GDN; GDN-2 applies it to β.
