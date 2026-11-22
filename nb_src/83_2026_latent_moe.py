#%% md
# 2026 — LatentMoE: Experts in a Compressed Space

**Sources:** NVIDIA, *"Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning"* (arXiv 2604.12374, 2026) and *Nemotron 3 Ultra* (arXiv 2606.15007, June 2026); Sebastian Raschka's *Nemotron 3 Ultra Latent MoE note* (June 2026). In October 2026, Meta's **Muse Spark Open** (816B) and **Nemotron 3.5 Super VL** also shipped LatentMoE layers.

Notebook 64 built a mixture of experts (MoE): a router sends each token to $K$ of $N$ expert MLPs. Since DeepSeek-V3, the trend has been **fine-grained** experts: more, smaller experts and a larger $K$, because more combinations of experts improves quality. That costs **communication**: in expert-parallel training and serving, each token's $d$-dimensional hidden state is sent to $K$ devices, so the all-to-all payload grows as $K\cdot d$.

**LatentMoE** keeps fine granularity without that cost. It projects each token into a smaller latent space first, and the experts live there (verbatim from the report):
- "Each input token $x\in\mathbb R^d$ is first projected into a lower-dimensional latent space $\mathbb R^\ell$ via a learnable down-projection matrix $W_\downarrow\in\mathbb R^{\ell\times d}$."
- Experts operate entirely in $\mathbb R^\ell$; their weighted outputs are "projected back to dimension $d$ via a learnable up-projection matrix $W_\uparrow\in\mathbb R^{d\times\ell}$."
- The savings are spent on more experts: "$N' = N\cdot d/\ell$ ... and the top-K active experts per token from $K$ to $K' = K\cdot d/\ell$."
- The router, the shared expert and all non-expert layers stay at the full width $d$.

Nemotron 3 Super: $d=4096 \to \ell=1024$, 512 experts, top-22. Ultra: $d=8192\to\ell=2048$, 512 experts, top-22, 550B total / 55B active.

**Bookkeeping, and an honest caveat (also Raschka's).** With $N'=4N$, $K'=4K$ and experts of the same hidden width $m$, the expert parameters ($N'\ell m = Ndm$), the per-token FLOPs ($K'\ell m = Kdm$) and the all-to-all payload ($K'\ell = Kd$) all **equal** the original coarse MoE. So the 4× bottleneck is not a 4× speedup. The gain is quality: $\binom{N'}{K'}$ expert combinations instead of $\binom{N}{K}$, at the coarse MoE's cost. The fair comparison is a **fine-grained** MoE with the same combinations and parameters, which pays $K'd$, i.e. 4× more communication.
#%%
{{SETUP}}
#%%
{{PROCEDURAL}}
#%% md
## The layers
`MoE(d, N, K, m)`: standard experts $d\to m\to d$. `LatentMoE(d, ell, N, K, m)`: router on $x$ (full $d$), $z = W_\downarrow x$, experts $\ell\to m\to\ell$, weighted sum in $\ell$, then $W_\uparrow$. Both have a shared expert on $d$ and a standard load-balancing loss. Experts are SwiGLU.
#%%
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6): super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x): return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class Experts(nn.Module):
    """N SwiGLU experts of width m acting on vectors of size `din`, stored as batched weights."""
    def __init__(self, N, din, m):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(N, din, 2 * m) / math.sqrt(din))
        self.w2 = nn.Parameter(torch.randn(N, m, din) / math.sqrt(m))
    def forward(self, x, idx, gate):
        """x: (n, din); idx, gate: (n, K). Returns sum_k gate_k * expert_{idx_k}(x)."""
        out = torch.zeros_like(x)
        for e in idx.unique().tolist():
            tok, slot = (idx == e).nonzero(as_tuple=True)
            a, b = (x[tok] @ self.w1[e]).chunk(2, -1)
            out.index_add_(0, tok, gate[tok, slot, None] * ((F.silu(a) * b) @ self.w2[e]))
        return out

class MoE(nn.Module):
    def __init__(self, d, N, K, m, ell=None, shared_m=None):
        super().__init__()
        self.N, self.K, self.ell = N, K, ell
        self.router = nn.Linear(d, N, bias=False)                         # always on full d
        self.down = nn.Linear(d, ell, bias=False) if ell else None
        self.up = nn.Linear(ell, d, bias=False) if ell else None
        self.experts = Experts(N, ell or d, m)
        sm = shared_m or 2 * m
        self.shared = nn.Sequential(nn.Linear(d, sm, bias=False), nn.GELU(), nn.Linear(sm, d, bias=False))
        self.aux = 0.0; self.load = None
    def forward(self, x):
        B, T, D = x.shape; xf = x.reshape(-1, D)
        probs = self.router(xf).softmax(-1)
        gate, idx = probs.topk(self.K, -1); gate = gate / gate.sum(-1, keepdim=True)
        load = torch.zeros(self.N, device=x.device).index_add_(0, idx.flatten(), torch.ones(idx.numel(), device=x.device)) / idx.numel()
        self.aux = self.N * (load * probs.mean(0)).sum(); self.load = load.detach()
        z = self.down(xf) if self.down is not None else xf                # into the latent space (LatentMoE)
        y = self.experts(z, idx, gate)
        if self.up is not None: y = self.up(y)                            # one up-projection per token, after the weighted sum
        return (y + self.shared(xf)).view(B, T, D)

class Block(nn.Module):
    def __init__(self, d, ffn, heads=4):
        super().__init__(); self.n1, self.n2, self.h = RMSNorm(d), RMSNorm(d), heads
        self.qkv, self.o, self.ffn = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False), ffn
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        x = x + self.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D))
        return x + self.ffn(self.n2(x))

class LM(nn.Module):
    def __init__(self, make_ffn, d=128, n_layers=4, ctx=128):
        super().__init__(); self.ctx = ctx
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, make_ffn()) for _ in range(n_layers)])
        self.ln, self.head = RMSNorm(d), nn.Linear(d, V, bias=False)
    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks: x = b(x)
        return self.head(self.ln(x))
    def aux(self): return sum(b.ffn.aux for b in self.blocks if isinstance(b.ffn, MoE))
#%% md
## Three MoEs with equal total and active expert parameters
$d=128$, scaling factor $d/\ell = 4$:

| | experts $N$ | top-$K$ | expert input | expert width $m$ | combinations | all-to-all floats / token |
|---|---|---|---|---|---|---|
| coarse MoE | 8 | 2 | $d=128$ | 128 | $\binom{8}{2}$ | $Kd$ = 256 |
| fine-grained MoE | 32 | 8 | $d=128$ | 32 | $\binom{32}{8}$ | $K'd$ = 1024 |
| **LatentMoE** | 32 | 8 | $\ell=32$ | 128 | $\binom{32}{8}$ | $K'\ell$ = 256 |

Fine-grained and LatentMoE have the same combinations; coarse MoE and LatentMoE have the same communication.
#%%
d, ell = 128, 32
configs = {
    "coarse MoE (8, top-2)":          lambda: MoE(d, N=8, K=2, m=128),
    "fine-grained MoE (32, top-8)":   lambda: MoE(d, N=32, K=8, m=32),
    "LatentMoE (32, top-8, ℓ=32)":    lambda: MoE(d, N=32, K=8, m=128, ell=ell),
}
print(f"{'':30s} {'expert params':>13s} {'active expert FLOPs/token':>26s} {'all-to-all floats/token':>24s} {'log10 combos':>13s}")
for name, mk in configs.items():
    f = mk()
    ep = f.experts.w1.numel() + f.experts.w2.numel()
    per_expert = ep // f.N
    flops = 2 * f.K * per_expert + (2 * 2 * d * ell if f.ell else 0)
    payload = f.K * (f.ell or d)
    print(f"{name:30s} {ep:13,d} {flops:26,d} {payload:24,d} {math.log10(math.comb(f.N, f.K)):13.1f}")
#%% md
## Train all three (plus a dense baseline)
#%%
def get_batch(src, B=32):
    ix = torch.randint(0, len(src) - 129, (B,), device=device)
    return src[ix[:, None] + torch.arange(128, device=src.device)], src[ix[:, None] + torch.arange(1, 129, device=src.device)]

def train(make_ffn, steps=800, lr=3e-3):
    seed_everything(0); m = LM(make_ffn).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1); t0 = time.time()
    for s in range(steps):
        x, y = get_batch(train_data)
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)) + 0.01 * m.aux()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    m.eval(); g = torch.Generator(device="cpu").manual_seed(1); vals = []
    with torch.no_grad():
        for _ in range(20):
            ix = torch.randint(0, len(val_data) - 129, (32,), generator=g).to(device)
            x = val_data[ix[:, None] + torch.arange(128, device=val_data.device)]; y = val_data[ix[:, None] + torch.arange(1, 129, device=val_data.device)]
            vals.append(F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)).item())
    return m, float(np.mean(vals)), time.time() - t0

results = {}
all_cfgs = {"dense MLP (no experts)": lambda: nn.Sequential(nn.Linear(d, 512, bias=False), nn.GELU(), nn.Linear(512, d, bias=False)), **configs}
for name, mk in all_cfgs.items():
    m, vl, sec = train(mk); results[name] = (m, vl)
    print(f"{name:30s} val loss {vl:.4f} | total params {sum(p.numel() for p in m.parameters()) / 1e3:.0f}K | {sec:.0f}s")
#%% md
## Expert load
A balanced router uses every expert. The bars show the fraction of routing slots per expert in the last layer, for the LatentMoE model (uniform = 1/32).
#%%
m = results["LatentMoE (32, top-8, ℓ=32)"][0]
with torch.no_grad(): m(get_batch(val_data, 64)[0])
load = m.blocks[-1].ffn.load.cpu()
plt.figure(figsize=(9, 2.5)); plt.bar(range(32), load); plt.axhline(1 / 32, c="gray", ls=":"); plt.xlabel("expert"); plt.ylabel("load"); plt.title("LatentMoE expert load (last layer)"); plt.show()
print(f"load: min {load.min():.3f}, max {load.max():.3f} (uniform {1 / 32:.3f})")
#%% md
## Takeaways
- LatentMoE puts experts in a compressed space ($d/\ell = 4$ in Nemotron 3) and spends the saving on **4× more experts and 4× larger top-K**, getting fine-grained routing at coarse-MoE communication and parameter cost.
- The bookkeeping matters: total expert parameters, active FLOPs and all-to-all payload all match the coarse MoE. It is **not** a 4× speedup; it buys the combinatorial richness of a fine-grained MoE without the fine-grained MoE's 4× communication.
- At this toy scale (one GPU, so no all-to-all), compare the losses only within the noise of one seed; whether LatentMoE beats fine-grained MoE on quality is not something a 4-layer character model can settle. The cost table above is the part that holds at any scale.
- Assumptions: router on full $d$ (the report's text; its Figure 3 caption suggests otherwise), softmax top-K routing with a standard balance loss (production uses sigmoid scores with bias-based balancing, notebook 64), and a shared expert added after $W_\uparrow$.
