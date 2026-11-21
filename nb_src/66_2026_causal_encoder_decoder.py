#%% md
# 2026 — The Causal Encoder–Decoder (DeepSeek-V4.1-Flash)

**Paper:** DeepSeek-AI, *"DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression"* (arXiv 2609.19969, September 2026). It is inspired by YOCO ("You Only Cache Once", 2024). Raschka remarked that it "should have been called DeepSeek V5".

Every decoder-only LLM since GPT (notebook 28) runs **all** layers on **every** prompt token and caches K/V for every layer. V4.1-Flash splits its 40 layers in two:

- a **20-layer causal encoder**: ordinary causal layers. Its final hidden state $H_{L/2}$ is the only thing the upper half needs from the past.
- a **20-layer decoder** whose global keys/values are *projections of the encoder's final state* (paper Eq. 1):
$$C_l = H_{L/2}\,W^{KV}_l,\qquad l > L/2$$
plus a **local sliding-window branch** over the decoder layer's own states ($n_{win}=128$).

Consequences, as stated in the paper:
- **Prefill runs only the encoder** for the prompt ("compute only the first half of the layers during the prefill phase"). Then the last $n_{win}$ prompt tokens are **replayed** through the decoder to build its local cache. Hence "8B active in prefill, 16B in decode".
- **Cache:** the decoder needs only $H_{L/2}$ (or its projections) for global context, plus a short window. Combined with CSA2 compression and FP4 keys, the global cache is **890 bytes per token**, about 1/4 of V4-Flash.
- Quality: Table 1 puts V4.1-Flash (8B/16B active) above V4-Flash (13B active) on MMLU-Pro 74.1 vs 68.3 and HumanEval 79.4 vs 69.5, though many other things changed between versions too.

This notebook builds a miniature, trains it against a decoder-only model of the same depth, verifies that encoder-only prefill plus replay reproduces the full forward pass, and does the cache and compute accounting.
#%%
{{SETUP}}
#%%
{{PROCEDURAL}}
#%%
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6): super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x): return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def rope(x, pos, base=10000.0):
    d = x.shape[-1]; inv = 1.0 / base ** (torch.arange(0, d, 2, device=x.device).float() / d)
    ang = pos.float()[:, None] * inv
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * ang.cos() - x2 * ang.sin(), x1 * ang.sin() + x2 * ang.cos()], -1).flatten(-2)

class SelfAttnLayer(nn.Module):
    """Standard causal layer (decoder-only model and the CED encoder)."""
    def __init__(self, d, heads=4):
        super().__init__(); self.h, self.dh = heads, d // heads
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False))
    def forward(self, x, pos):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos), rope(k, pos)
        x = x + self.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.n2(x))

class DecoderLayer(nn.Module):
    """CED decoder layer: own queries; global K/V projected from the ENCODER's final state (Eq. 1); local K/V from its own input, window W.
    One softmax over [global keys (causal) ; local keys (in window)]."""
    def __init__(self, d, heads=4, window=16):
        super().__init__(); self.h, self.dh, self.W = heads, d // heads, window
        self.n1, self.n2, self.ne = RMSNorm(d), RMSNorm(d), RMSNorm(d)
        self.q, self.kv_glob, self.kv_loc = nn.Linear(d, d, bias=False), nn.Linear(d, 2 * d, bias=False), nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False))
    def forward(self, x, pos, h_enc, pos_enc):
        """x, pos: the decoder positions being computed (a suffix during replay). h_enc, pos_enc: encoder states for all positions so far."""
        B, T, D = x.shape; S = h_enc.shape[1]
        sh = lambda t, n: t.view(B, n, self.h, self.dh).transpose(1, 2)
        xn = self.n1(x)
        q = rope(sh(self.q(xn), T), pos)
        kg, vg = self.kv_glob(self.ne(h_enc)).chunk(2, -1); kg = rope(sh(kg, S), pos_enc); vg = sh(vg, S)
        kl, vl = self.kv_loc(xn).chunk(2, -1); kl = rope(sh(kl, T), pos); vl = sh(vl, T)
        mask_g = pos_enc[None, :] <= pos[:, None]                                               # causal over the encoder states
        dpos = pos[:, None] - pos[None, :]
        mask_l = (dpos >= 0) & (dpos < self.W)                                                  # sliding window over own states
        k, v = torch.cat([kg, kl], 2), torch.cat([vg, vl], 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.cat([mask_g, mask_l], 1))
        x = x + self.o(y.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.n2(x))

class DecoderOnly(nn.Module):
    def __init__(self, d=128, n_layers=8):
        super().__init__(); self.tok = nn.Embedding(V, d)
        self.layers = nn.ModuleList([SelfAttnLayer(d) for _ in range(n_layers)]); self.ln, self.head = RMSNorm(d), nn.Linear(d, V, bias=False)
    def forward(self, idx):
        pos = torch.arange(idx.shape[1], device=idx.device); x = self.tok(idx)
        for l in self.layers: x = l(x, pos)
        return self.head(self.ln(x))

class CausalEncoderDecoder(nn.Module):
    def __init__(self, d=128, n_layers=8, window=16):
        super().__init__(); self.tok = nn.Embedding(V, d)
        self.enc = nn.ModuleList([SelfAttnLayer(d) for _ in range(n_layers // 2)])
        self.dec = nn.ModuleList([DecoderLayer(d, window=window) for _ in range(n_layers // 2)])
        self.ln, self.head = RMSNorm(d), nn.Linear(d, V, bias=False)
    def encode(self, idx):
        pos = torch.arange(idx.shape[1], device=idx.device); x = self.tok(idx)
        for l in self.enc: x = l(x, pos)
        return x                                                                               # H_{L/2}
    def decode(self, h_enc, start=0):
        """Run the decoder for positions start..S-1 only (start>0 = the paper's replay of a prompt suffix)."""
        S = h_enc.shape[1]; pos_enc = torch.arange(S, device=h_enc.device); pos = pos_enc[start:]
        x = h_enc[:, start:]
        for l in self.dec: x = l(x, pos, h_enc, pos_enc)
        return self.head(self.ln(x))
    def forward(self, idx): return self.decode(self.encode(idx))
#%% md
## Train: decoder-only vs. causal encoder–decoder (same depth, same width)
#%%
def get_batch(src, B=32, T=256):
    ix = torch.randint(0, len(src) - T - 1, (B,), device=src.device)
    w = src[ix[:, None] + torch.arange(T + 1, device=src.device)]                 # one gather, no per-row Python loop
    return w[:, :-1], w[:, 1:]

def train(model, steps=1000, lr=2e-3):
    seed_everything(0); model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1); t0 = time.time()
    for s in range(steps):
        x, y = get_batch(train_data); loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    model.eval(); g = torch.Generator(device="cpu").manual_seed(1); vals = []
    with torch.no_grad():
        for _ in range(20):
            ix = torch.randint(0, len(val_data) - 257, (32,), generator=g).to(device)
            w = val_data[ix[:, None] + torch.arange(257, device=device)]; x, y = w[:, :-1], w[:, 1:]
            vals.append(F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item())
    return model, float(np.mean(vals)), time.time() - t0

seed_everything(0); dec_only, vl_d, t_d = train(DecoderOnly())
seed_everything(0); ced, vl_c, t_c = train(CausalEncoderDecoder())
print(f"decoder-only (8 layers)          val loss {vl_d:.4f}  params {sum(p.numel() for p in dec_only.parameters()) / 1e6:.2f}M  ({t_d:.0f}s)")
print(f"causal encoder-decoder (4 + 4)   val loss {vl_c:.4f}  params {sum(p.numel() for p in ced.parameters()) / 1e6:.2f}M  ({t_c:.0f}s)")
#%% md
## Prefill with the encoder only, then replay
Serving a 256-token prompt:
1. Run the 4 encoder layers on all 256 tokens. Keep $H_{L/2}$.
2. **Replay** only a short suffix of the prompt through the decoder, to get the next-token logits and the decoder's local (sliding-window) states.

How long must the replay be? The decoder's global branch reads $H_{L/2}$, which is complete. But its local branch at layer $l$ reads the window of layer $l$'s own states, which depend on the window of layer $l-1$, and so on. Exactness needs a suffix of about $(\#\text{dec layers})\times(W-1)+1$ tokens. The paper replays just $n_{win}$ and reports "only a negligible impact on response quality". Let's measure both.
#%%
x, _ = get_batch(val_data, 16)
with torch.no_grad():
    full = ced(x)[:, -1]                                                       # logits for the next token, full forward
    h = ced.encode(x)
    W, nd = 16, len(ced.dec)
    for name, start in [(f"replay last W = {W} tokens (paper's choice)", x.shape[1] - W),
                        (f"replay last {nd}(W-1)+1 = {nd * (W - 1) + 1} tokens (exact)", x.shape[1] - (nd * (W - 1) + 1)),
                        ("replay 1 token only", x.shape[1] - 1)]:
        out = ced.decode(h, start)[:, -1]
        same = (out.argmax(-1) == full.argmax(-1)).float().mean().item()
        print(f"{name:44s} max |Δlogit| {(out - full).abs().max().item():.2e} | same next-token argmax {same:.2f}")
#%% md
## Cache and compute accounting
Per token, in floats, for this model ($d=128$, 8 layers, window 16) and scaled to a long context. Decoder-only caches K and V in every layer. The CED caches K/V for the 4 encoder layers plus **one** $H_{L/2}$ vector per token (decoder layers can recompute their global K/V from it, or V4.1 shares one main KV across them). The decoder's local caches are bounded by the window. Prefill FLOPs: decoder-only runs 8 layers on every prompt token; the CED runs 4, plus a short replay.
#%%
def accounting(T, d=128, L=8, W=16, replay=16):
    dec_cache = L * 2 * d * T
    ced_cache = (L // 2) * 2 * d * T + d * T + (L // 2) * 2 * d * W
    per_layer = 1.0                                                             # cost unit: one layer applied to one token
    dec_prefill = L * T * per_layer
    ced_prefill = (L // 2) * T * per_layer + (L // 2) * replay * per_layer
    return dec_cache, ced_cache, dec_prefill, ced_prefill
print(f"{'context':>10s} {'cache dec-only':>15s} {'cache CED':>12s} {'ratio':>6s} {'prefill dec-only':>17s} {'prefill CED':>12s} {'ratio':>6s}")
for T in (256, 8192, 1_000_000):
    a, b, c, e = accounting(T)
    print(f"{T:10,d} {a:15,.0f} {b:12,.0f} {b / a:6.2f} {c:17,.0f} {e:12,.0f} {e / c:6.2f}")
#%% md
In V4.1-Flash the encoder half also uses compressed CSA2 caches and the decoder shares one FP4 main KV, which is how the paper gets to ~1/4 of V4-Flash's global cache. The structural saving shown here is the part that comes from the encoder/decoder split alone.

## Generate
#%%
@torch.no_grad()
def generate(model, prompt, n=200):
    idx = torch.tensor([encode(prompt)], device=device)
    for _ in range(n):
        h = model.encode(idx[:, -256:])
        logits = model.decode(h, h.shape[1] - (len(model.dec) * 15 + 1))[:, -1]   # encoder prefill + exact-length replay
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits / 0.8, -1), 1)], 1)
    return decode(idx[0].tolist())
print(generate(ced, "The model learns"))
#%% md
## Takeaways
- **Split the stack.** A causal encoder produces one state per token; the decoder reads its global context only from that state (per-layer projections, Eq. 1) plus a short local window. Global memory is cached once instead of in every upper layer.
- **Prefill gets cheaper:** prompt tokens skip the decoder, except a short replayed suffix. For agentic workloads, where prompts (tool outputs, code) are long and outputs short, that is most of the compute. Hence V4.1-Flash's 8B prefill / 16B decode active parameters.
- **Replay length is a real trade-off:** exactness needs about depth × window tokens; the paper replays one window and accepts a tiny approximation. The measurement above shows the size of that approximation in our model.
- At this scale, the two models' losses are close, not identical; the paper reports quality "comparable to the baseline" and gives no decoder-only ablation table. The cache and prefill savings are structural and do not depend on scale.
- Simplifications: plain dense attention everywhere (V4.1 uses CSA2 sparse attention with Full/Reindex/Reuse modes, notebook 65), no MoE, no mHC, no Engram.
