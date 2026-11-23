#%% md
# 2026 — Cheaper Indexers for Sparse Attention: Block Compression and Index Reuse

**Sources:** Qwen Team, *"On the Design of Qwen3.8-Next Architecture"* (arXiv 2608.30320, August 2026): **Qwen Sparse Attention (QSA)**. DeepSeek-AI, *"DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression"* (arXiv 2609.19969, September 2026): **CSA2** with **Full / Reindex / Reuse** layer modes. Also in production: Kimi K3.1 Pro (November 2026) adds a DSA indexer to its 24 global layers, and MiniMax M3.1 and Mistral Large 4 ship their own sparse attention.

DeepSeek Sparse Attention (notebook 75) made the *main* attention sparse, with each query reading only its top-$k$ keys chosen by a tiny "lightning indexer". But the indexer itself still scores **every** past token, in **every** sparse layer: $O(T^2)$ per layer. At 1M tokens that is the next bottleneck. The fixes that shipped in late 2026:

**1. QSA: score blocks, not tokens (Qwen3.8).** Average-pool the indexer keys over blocks of $r=4$ tokens, score the blocks, and expand the winners back to tokens:
$$\hat k_b = \operatorname{RMSNorm}\Big(\tfrac1r\textstyle\sum_{j=br}^{br+r-1}k_j\Big),\qquad I_{ib} = \sum_h\operatorname{ReLU}(\langle q^h_i,\hat k_b\rangle)\ \text{ if } br+r-1\le i,\qquad \mathcal B_i = \operatorname{TopK}_{\lceil K/r\rceil}(I_{i,:})$$
The selected token set is the selected blocks plus the current, still incomplete block. Indexer cost falls from $O(T^2)$ to $O(T^2/r)$. The indexer is trained by distilling the main attention, **max-pooled per block** and L1-normalized: $\mathcal L = D_{KL}(\hat a_i\,\|\,\operatorname{softmax}(I_{i,:}))$. Qwen3.8-Flash-Next: $r=4$, budget 2,048 tokens = 512 blocks, an indexer with 4 query heads and one shared key head. At 1M tokens, the reported kernel speedups vs dense attention are 7.6× prefill and 4.9× decode.

**2. Index reuse across layers (DeepSeek-V4.1, CSA2).** Neighbouring layers tend to want the same tokens, so not every layer needs its own indexer:
- **Full** layer: computes its own indexer and fresh top-$k$.
- **Reindex** layer: reuses the previous indexer *keys* but computes its own indexer *query* and re-scores.
- **Reuse** layer: reuses the latest top-$k$ indices outright, with no indexer at all.

V4.1-Flash's encoder repeats groups of [1 Full + 5 Reuse]; its decoder uses [Full or Reindex] + 3 Reuse.

**This notebook:** the long-context recall task from notebook 75, at 256 tokens, with three layers (local → global → global). We convert a trained dense model to (a) DSA in both global layers, (b) QSA in both, and (c) Full + Reuse, then compare accuracy against indexer work.
#%% md
![Diagram: Cheaper sparse-attention indexers (Qwen3.8 QSA; DeepSeek-V4.1 CSA2, 2026)](https://raw.githubusercontent.com/babushkai/neuralnetwork/main/assets/diagrams/86_2026_cheaper_sparse_attention_indexers.png)
#%%
{{SETUP}}
#%%
NK, NV, P, NQ, T_ctx = 16, 16, 8, 4, 256
FILL = NK + NV; QRY = FILL + 1; V = QRY + 1
def batch(B):
    x = torch.full((B, T_ctx), FILL)
    keys = torch.rand(B, NK).argsort(1)[:, :P]; vals = torch.randint(NK, NK + NV, (B, P))
    slots = torch.rand(B, (T_ctx - 2 * NQ) // 2).argsort(1)[:, :P] * 2
    x.scatter_(1, slots, keys); x.scatter_(1, slots + 1, vals)
    pick = torch.rand(B, P).argsort(1)[:, :NQ]
    x[:, -2 * NQ:] = torch.stack([torch.full((B, NQ), QRY), keys.gather(1, pick)], 2).view(B, 2 * NQ)
    return x.to(device), vals.gather(1, pick).to(device)
#%%
def rope(x, base=10000.0):
    T, d = x.shape[-2], x.shape[-1]
    inv = 1.0 / base ** (torch.arange(0, d, 2, device=x.device).float() / d)
    ang = torch.arange(T, device=x.device).float()[:, None] * inv
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * ang.cos() - x2 * ang.sin(), x1 * ang.sin() + x2 * ang.cos()], -1).flatten(-2)

class TokenIndexer(nn.Module):
    """DSA lightning indexer (notebook 75): per-token scores."""
    def __init__(self, d, n_heads=4, d_idx=16):
        super().__init__(); self.h, self.di = n_heads, d_idx
        self.q, self.k, self.w = nn.Linear(d, n_heads * d_idx, bias=False), nn.Linear(d, d_idx, bias=False), nn.Linear(d, n_heads, bias=False)
    def forward(self, x):
        B, T, _ = x.shape
        s = F.relu(torch.einsum("bthd,bsd->bths", self.q(x).view(B, T, self.h, self.di), self.k(x)))
        return (self.w(x).unsqueeze(-1) * s).sum(2)                                   # (B, T, T)

class BlockIndexer(nn.Module):
    """QSA indexer: RMSNorm'd queries, one shared key head, average-pooled + RMSNorm'd block keys."""
    def __init__(self, d, n_heads=4, d_idx=16, r=4):
        super().__init__(); self.h, self.di, self.r = n_heads, d_idx, r
        self.q, self.k = nn.Linear(d, n_heads * d_idx, bias=False), nn.Linear(d, d_idx, bias=False)
    def forward(self, x):
        B, T, _ = x.shape
        q = F.rms_norm(self.q(x).view(B, T, self.h, self.di), (self.di,))
        kb = F.rms_norm(self.k(x).view(B, T // self.r, self.r, self.di).mean(2), (self.di,))
        return F.relu(torch.einsum("bthd,bnd->bthn", q, kb)).sum(2)                  # (B, T, T/r)

class Attention(nn.Module):
    """mode: 'dense' | 'window' | 'dsa' | 'qsa' | 'reuse' (take the selection of the previous sparse layer)."""
    def __init__(self, d, n_heads=4, mode="dense", window=16, topk=32, r=4):
        super().__init__()
        self.h, self.dh, self.mode, self.window, self.topk, self.r = n_heads, d // n_heads, mode, window, topk, r
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.indexer = TokenIndexer(d) if mode == "dsa" else BlockIndexer(d, r=r) if mode == "qsa" else None
        self.sparse_on, self.kl = True, torch.tensor(0.0)

    def select(self, x, scores, causal):
        """Returns a boolean (B, T, T) selection and sets self.kl (the indexer's distillation loss)."""
        B, T, _ = x.shape
        teacher = scores.detach().masked_fill(~causal, float("-inf")).softmax(-1).sum(1)       # main attention, summed over heads
        if self.mode == "dsa":
            idx = self.indexer(x).masked_fill(~causal, float("-inf"))
            sel = torch.zeros_like(causal).scatter_(-1, idx.topk(self.topk, -1).indices, True) & causal if self.sparse_on else causal
            tgt = teacher.masked_fill(~sel, 0); tgt = tgt / tgt.sum(-1, keepdim=True)
            logp = idx.masked_fill(~sel, float("-inf")).log_softmax(-1).masked_fill(~sel, 0)
        else:                                                                                   # qsa
            r, nb = self.r, T // self.r
            i = torch.arange(T, device=x.device); b = torch.arange(nb, device=x.device)
            complete = (b[None, :] * r + r - 1) <= i[:, None]                                    # block fully in the past
            idx = self.indexer(x).masked_fill(~complete, -1e9)                                 # finite, so all-masked rows stay NaN-free
            kb = min(math.ceil(self.topk / r), nb)
            if self.sparse_on:
                top = idx.topk(kb, -1).indices
                bsel = torch.zeros(B, T, nb, dtype=torch.bool, device=x.device).scatter_(-1, top, True) & complete
            else:
                bsel = complete.expand(B, T, nb)
            tail = (i[None, :] >= (r * ((i[:, None] + 1) // r))) & causal[0]                   # current incomplete block
            sel = bsel.repeat_interleave(r, -1) | tail | torch.eye(T, dtype=torch.bool, device=x.device)   # always keep the token itself
            sel = sel & causal
            tb = teacher.view(B, T, nb, r).amax(-1).masked_fill(~(bsel & complete), 0)          # block max-pool
            tgt = tb / tb.sum(-1, keepdim=True).clamp_min(1e-9)
            m = bsel & complete
            logp = idx.masked_fill(~m, -1e9).log_softmax(-1).masked_fill(~m, 0)
            valid = m.any(-1)                                                                  # queries with >=1 complete block
            self.kl = (tgt * (tgt.clamp_min(1e-9).log() - logp)).sum(-1)[valid].mean()
            return sel
        self.kl = (tgt * (tgt.clamp_min(1e-9).log() - logp)).sum(-1).mean()
        return sel

    def forward(self, x, shared_sel=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        i = torch.arange(T, device=x.device)
        causal = (i[None, :] <= i[:, None]).expand(B, T, T)
        allowed, sel = causal, None
        if self.mode == "window": allowed = causal & (i[:, None] - i[None, :] < self.window)
        elif self.mode in ("dsa", "qsa"): sel = allowed = self.select(x, scores, causal)
        elif self.mode == "reuse": allowed = shared_sel if (shared_sel is not None and self.sparse_on) else causal
        a = scores.masked_fill(~allowed.unsqueeze(1), float("-inf")).softmax(-1)
        return self.o((a @ v).transpose(1, 2).reshape(B, T, D)), sel

class LM(nn.Module):
    def __init__(self, modes, d=64):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.attn = nn.ModuleList([Attention(d, mode=m) for m in modes])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in modes])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in modes]); self.n2 = nn.ModuleList([nn.LayerNorm(d) for _ in modes])
        self.head = nn.Linear(d, V)
    def forward(self, x):
        h, sel = self.emb(x), None
        for a, m, n1, n2 in zip(self.attn, self.mlp, self.n1, self.n2):
            o, s = a(n1(h), shared_sel=sel); sel = s if s is not None else sel
            h = h + o; h = h + m(n2(h))
        return self.head(h[:, -2 * NQ + 1::2])
    def indexer_kl(self): return sum(a.kl for a in self.attn if a.indexer is not None)
    def set_sparse(self, on):
        for a in self.attn: a.sparse_on = on

def fit(model, steps, lr=2e-3, params=None, kl_weight=0.0, task_weight=1.0):
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    for s in range(steps):
        x, y = batch(32)
        loss = task_weight * F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)) + kl_weight * model.indexer_kl()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return model

@torch.no_grad()
def accuracy(model):
    model.eval(); accs = []
    for _ in range(4):
        x, y = batch(500); accs.append((model(x).argmax(-1) == y).float().mean().item())
    model.train(); return float(np.mean(accs))
#%% md
## The dense teacher: local → global → global
#%%
seed_everything(0); t0 = time.time()
dense = fit(LM(["window", "dense", "dense"]).to(device), 2500)
print(f"dense model accuracy {accuracy(dense):.3f}   ({time.time() - t0:.0f}s)")
#%% md
## Convert to three kinds of sparse global layers
Same recipe for each: copy the dense weights, (1) warm up the indexers with dense attention and the KL loss (backbone frozen), then (2) switch on sparsity and train everything briefly. Budget: $k=32$ of up to 256 tokens per query (QSA: 8 blocks of 4, plus the current block).
#%%
variants = {"DSA in both global layers": ["window", "dsa", "dsa"],
            "QSA (block indexer, r=4) in both": ["window", "qsa", "qsa"],
            "CSA2-style: DSA Full + Reuse": ["window", "dsa", "reuse"]}
results = {"dense (teacher)": accuracy(dense)}
for name, modes in variants.items():
    seed_everything(0); t0 = time.time()
    m = LM(modes).to(device); m.load_state_dict(dense.state_dict(), strict=False)
    m.set_sparse(True); untrained = accuracy(m)
    m.set_sparse(False); fit(m, 300, lr=3e-3, params=[p for n, p in m.named_parameters() if "indexer" in n], kl_weight=1.0, task_weight=0.0)
    m.set_sparse(True); warm = accuracy(m)
    fit(m, 400, lr=1e-3, kl_weight=1.0); final = accuracy(m)
    results[name] = final
    print(f"{name:34s} untrained indexer {untrained:.3f} -> after warm-up {warm:.3f} -> after sparse training {final:.3f}   ({time.time() - t0:.0f}s)")
#%% md
## Indexer work vs. accuracy
Indexer dot products per query in the two global layers at context length $T$ (indexer heads $H_I=4$). Token indexers score $T$ positions per layer; block indexers score $T/r$; a Reuse layer scores nothing.
#%%
def indexer_work(modes, T, r=4, H=4):
    return sum(H * T if m == "dsa" else H * T // r if m == "qsa" else 0 for m in modes)
print(f"{'':34s} {'accuracy':>8s} {'indexer work @256':>18s} {'@1M tokens':>14s}")
for name, modes in [("dense (teacher)", ["window", "dense", "dense"]), *variants.items()]:
    print(f"{name:34s} {results[name]:8.3f} {indexer_work(modes, 256):18,d} {indexer_work(modes, 1_000_000):14,d}")
plt.figure(figsize=(8, 3.5))
names = list(variants); plt.scatter([indexer_work(variants[n], 256) for n in names], [results[n] for n in names])
for n in names: plt.annotate(n, (indexer_work(variants[n], 256), results[n]), fontsize=8, xytext=(5, 5), textcoords="offset points")
plt.axhline(results["dense (teacher)"], ls="--", c="gray", label="dense teacher"); plt.xlabel("indexer dot products per query (T=256)"); plt.ylabel("recall accuracy"); plt.legend(); plt.show()
#%% md
## Do neighbouring layers want the same tokens?
Index reuse rests on this. Take the DSA model (each global layer has its own indexer) and measure the overlap of the two layers' top-32 sets for the query tokens.
#%%
m = LM(variants["DSA in both global layers"]).to(device); m.load_state_dict(dense.state_dict(), strict=False)
m.set_sparse(False); fit(m, 300, lr=3e-3, params=[p for n, p in m.named_parameters() if "indexer" in n], kl_weight=1.0, task_weight=0.0); m.set_sparse(True)
x, _ = batch(200); sels = []
with torch.no_grad():
    h = m.emb(x)
    for a, mm, n1, n2 in zip(m.attn, m.mlp, m.n1, m.n2):
        o, s = a(n1(h)); h = h + o; h = h + mm(n2(h))
        if s is not None: sels.append(s[:, -2 * NQ + 1::2])
overlap = ((sels[0] & sels[1]).sum(-1).float() / sels[1].sum(-1)).mean().item()
print(f"overlap of the two global layers' top-32 sets at the query tokens: {overlap:.2f} (random would be ~{32 / 256:.2f})")
#%% md
## Takeaways
- Sparse attention moved the bottleneck from the attention to the indexer. Late-2026 models attack the indexer itself: **score blocks instead of tokens** (QSA: 4× less indexer work at $r=4$) and **share selections across layers** (CSA2 Reuse/Reindex: most layers run no indexer).
- Both rely on structure in attention: what a query needs comes in contiguous chunks, and neighbouring layers need similar chunks. The overlap measurement above checks the second assumption directly.
- At this scale all variants should land near the dense teacher after the two-stage conversion (accuracy differences of a few points are within run-to-run noise). The indexer-work column is where they differ, and that column is what grows with context length.
- Simplifications: no partial RoPE in the QSA indexer (the paper rotates 64 of 128 indexer dims, with block keys at their start position), no Reindex mode, and no V4.1 hierarchical (block-then-token) indexer.
