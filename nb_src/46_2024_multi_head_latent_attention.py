#%% md
# 2024 — Multi-head Latent Attention (MLA) and the KV-Cache Problem

**Papers:** DeepSeek-AI, *"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"* (2024), also used in DeepSeek-V3/R1 and Kimi K2. Background: Multi-Query Attention (Shazeer, 2019); Grouped-Query Attention (Ainslie et al., 2023); RoPE (Su et al., 2021).

During generation an LLM caches the keys and values of every past token. For long contexts this **KV cache**, not the weights, dominates memory. Per token and layer:

| method | cached floats per token |
|---|---|
| MHA (multi-head) | $2\,n_h d_h$ |
| GQA ($g$ KV groups) | $2\,g\,d_h$ |
| MQA | $2\,d_h$ |
| **MLA** | $d_c + d_h^{R}$ |

**MLA** compresses keys and values jointly into a small **latent** $c^{KV}_t = W^{DKV}x_t$ and caches only that. Per-head keys/values are reconstructed by up-projections $k = W^{UK}c$, $v = W^{UV}c$. Because RoPE (position-dependent rotation) would break this, a small **decoupled RoPE key** $k^R_t$ shared across heads is cached alongside.

**Weight absorption:** $q^\top k = q^\top W^{UK} c = (W^{UK\top} q)^\top c$, so at inference you can attend directly in latent space and never materialise the full keys.

Below: MLA from scratch → train a tiny LM → check (1) incremental decoding with the latent cache equals the full forward pass, and (2) the absorbed computation gives identical outputs.
#%%
{{SETUP}}
#%%
def rope_cache(T, d, base=10000.0, device=None):
    inv = 1.0 / base ** (torch.arange(0, d, 2, device=device).float() / d)
    ang = torch.arange(T, device=device).float()[:, None] * inv[None]
    return ang.cos(), ang.sin()

def apply_rope(x, cos, sin):                      # x (..., T, d): rotate consecutive pairs
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6): super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x): return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class MLA(nn.Module):
    def __init__(self, d, n_h=4, d_nope=32, d_rope=16, d_v=32, d_c=32, d_cq=64):
        super().__init__()
        self.n_h, self.d_nope, self.d_rope, self.d_v, self.d_c = n_h, d_nope, d_rope, d_v, d_c
        self.W_DQ, self.q_norm = nn.Linear(d, d_cq, bias=False), RMSNorm(d_cq)              # query compression
        self.W_UQ = nn.Linear(d_cq, n_h * (d_nope + d_rope), bias=False)
        self.W_DKV, self.kv_norm = nn.Linear(d, d_c, bias=False), RMSNorm(d_c)              # joint KV compression -> CACHED
        self.W_UK = nn.Linear(d_c, n_h * d_nope, bias=False)
        self.W_UV = nn.Linear(d_c, n_h * d_v, bias=False)
        self.W_KR = nn.Linear(d, d_rope, bias=False)                                         # decoupled RoPE key -> CACHED
        self.W_O = nn.Linear(n_h * d_v, d, bias=False)

    def queries(self, x, pos0):
        B, T, _ = x.shape
        q = self.W_UQ(self.q_norm(self.W_DQ(x))).view(B, T, self.n_h, -1).transpose(1, 2)
        q_nope, q_rope = q.split([self.d_nope, self.d_rope], -1)
        cos, sin = rope_cache(pos0 + T, self.d_rope, device=x.device)
        return q_nope, apply_rope(q_rope, cos[pos0:], sin[pos0:])

    def latents(self, x, pos0):
        c = self.kv_norm(self.W_DKV(x))                                                      # (B, T, d_c)
        cos, sin = rope_cache(pos0 + x.shape[1], self.d_rope, device=x.device)
        k_rope = apply_rope(self.W_KR(x), cos[pos0:], sin[pos0:])                            # (B, T, d_rope)
        return c, k_rope

    def forward(self, x, cache=None, absorb=False):
        B, T, _ = x.shape
        pos0 = 0 if cache is None else cache[0].shape[1]
        q_nope, q_rope = self.queries(x, pos0)
        c, k_rope = self.latents(x, pos0)
        if cache is not None: c, k_rope = torch.cat([cache[0], c], 1), torch.cat([cache[1], k_rope], 1)
        S = c.shape[1]
        scale = 1 / math.sqrt(self.d_nope + self.d_rope)
        rope_scores = q_rope @ k_rope.unsqueeze(1).transpose(-2, -1)                         # shared rope key for all heads
        if not absorb:   # naive: decompress the latent into per-head keys and values
            k_nope = self.W_UK(c).view(B, S, self.n_h, -1).transpose(1, 2)
            v = self.W_UV(c).view(B, S, self.n_h, -1).transpose(1, 2)
            scores = (q_nope @ k_nope.transpose(-2, -1) + rope_scores) * scale
        else:            # absorbed: move W_UK onto the query side, attend in latent space
            W_UK = self.W_UK.weight.view(self.n_h, self.d_nope, self.d_c)
            q_lat = torch.einsum("bhtn,hnc->bhtc", q_nope, W_UK)
            scores = (q_lat @ c.unsqueeze(1).transpose(-2, -1) + rope_scores) * scale
        mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(S - T)
        a = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        if not absorb:
            o = a @ v
        else:            # o_h = W_UV_h (sum_s a_s c_s): aggregate latents first, then up-project
            W_UV = self.W_UV.weight.view(self.n_h, self.d_v, self.d_c)
            o = torch.einsum("bhtc,hvc->bhtv", a @ c.unsqueeze(1), W_UV)
        return self.W_O(o.transpose(1, 2).reshape(B, T, -1)), (c, k_rope)
#%% md
## KV-cache size comparison
For a DeepSeek-V2-like configuration (128 heads × 128 dims, $d_c = 512$, $d^R_h = 64$), per token and per layer:
#%%
n_h, d_h = 128, 128
rows = [("MHA", 2 * n_h * d_h), ("GQA (8 groups)", 2 * 8 * d_h), ("MQA", 2 * d_h), ("MLA (d_c=512, d_rope=64)", 512 + 64)]
for name, n in rows: print(f"{name:26s} {n:7,d} floats  ({n / rows[0][1]:.1%} of MHA)")
ctx, layers, bytes_ = 128_000, 60, 2
print(f"\nfor a 128K-token context, 60 layers, bf16: MHA {2 * n_h * d_h * ctx * layers * bytes_ / 1e9:.0f} GB vs MLA {(512 + 64) * ctx * layers * bytes_ / 1e9:.1f} GB")
#%% md
## A tiny MLA language model
#%%
{{CORPUS}}
chars = sorted(set(CORPUS)); V = len(chars); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in CORPUS], device=device)

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n1, self.n2, self.attn = RMSNorm(d), RMSNorm(d), MLA(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False))
    def forward(self, x, cache=None, absorb=False):
        a, new_cache = self.attn(self.n1(x), cache, absorb)
        x = x + a
        return x + self.mlp(self.n2(x)), new_cache

class MLALM(nn.Module):
    def __init__(self, V, d=128, n=3):
        super().__init__()
        self.emb, self.blocks, self.norm = nn.Embedding(V, d), nn.ModuleList([Block(d) for _ in range(n)]), RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
    def forward(self, idx, caches=None, absorb=False):
        x, new = self.emb(idx), []
        for i, b in enumerate(self.blocks):
            x, c = b(x, None if caches is None else caches[i], absorb); new.append(c)
        return self.head(self.norm(x)), new

model = MLALM(V).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.1)
ctx = 64
for step in range(1000):
    ix = torch.randint(0, len(data) - ctx - 1, (32,), device=device)
    x = torch.stack([data[i:i + ctx] for i in ix]); y = torch.stack([data[i + 1:i + ctx + 1] for i in ix])
    loss = F.cross_entropy(model(x)[0].reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0: print(f"step {step} loss {loss.item():.3f}")
#%% md
## Check 1: incremental decoding with the latent cache = full forward
Check 2: the absorbed (latent-space) attention gives the same logits.
#%%
model.eval()
prompt = torch.tensor([[stoi[c] for c in "Attention changed "]], device=device)
with torch.no_grad():
    full, _ = model(prompt)
    logits, caches = model(prompt[:, :1])                       # feed tokens one at a time, reusing the cache
    steps = [logits]
    for t in range(1, prompt.shape[1]):
        logits, caches = model(prompt[:, t:t + 1], caches); steps.append(logits)
    incremental = torch.cat(steps, 1)
    absorbed, _ = model(prompt, absorb=True)
print(f"max |incremental - full| = {(incremental - full).abs().max():.2e}")
print(f"max |absorbed - naive|   = {(absorbed - full).abs().max():.2e}")
print(f"cache per layer per token: latent {caches[0][0].shape[-1]} + rope key {caches[0][1].shape[-1]} = {caches[0][0].shape[-1] + caches[0][1].shape[-1]} floats "
      f"(an MHA cache for this model would hold {2 * 4 * 32})")
#%%
with torch.no_grad():
    idx = prompt; logits, caches = model(idx); out = []
    for _ in range(150):
        nxt = torch.multinomial((logits[:, -1] / 0.7).softmax(-1), 1); out.append(nxt.item())
        logits, caches = model(nxt, caches, absorb=True)      # cached + absorbed decoding
print("Attention changed " + "".join(chars[i] for i in out))
