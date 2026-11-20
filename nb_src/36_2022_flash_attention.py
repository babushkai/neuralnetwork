#%% md
# 2022 — FlashAttention: IO-Aware Exact Attention

**Papers:** Tri Dao, Daniel Fu, Stefano Ermon, Atri Rudra & Christopher Ré, *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"* (2022); FlashAttention-2 (2023), FlashAttention-3 (2024). Built on the *online softmax* (Milakov & Gimelshein, 2018) and memory-efficient attention (Rabe & Staats, 2021).

Standard attention writes the whole $T\times T$ score matrix to GPU memory (HBM), reads it back for the softmax, then again for $PV$. At long context that matrix is huge, and moving it is slower than the arithmetic itself.

FlashAttention computes **exactly the same result** without ever storing the $T\times T$ matrix:
- split $Q$, $K$, $V$ into **tiles** that fit in fast on-chip memory (SRAM)
- for each query tile, stream over key/value tiles and keep a **running max $m$, running sum $\ell$ and running output $O$**:
$$m' = \max(m, \operatorname{rowmax}(S)),\quad \ell' = e^{m-m'}\ell + \operatorname{rowsum}(e^{S-m'}),\quad O' = e^{m-m'}O + e^{S-m'}V$$
- divide by $\ell$ at the end. The backward pass **recomputes** tiles instead of storing them.

Memory becomes $O(T)$ instead of $O(T^2)$, and it runs 2–4× faster. It's the reason long-context training is practical; every modern LLM stack (and PyTorch's `F.scaled_dot_product_attention`) uses it or a successor.

We implement the tiled algorithm in plain PyTorch. It won't be faster than a fused kernel here, since that needs custom GPU code, but it shows the algorithm is exact and how the memory saving comes about.
#%%
{{SETUP}}
#%%
def naive_attention(q, k, v, causal=True):
    S = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])                 # (T, T) materialised in memory
    if causal: S = S.masked_fill(torch.ones_like(S, dtype=torch.bool).triu(1), float("-inf"))
    return S.softmax(-1) @ v

def flash_attention(q, k, v, causal=True, Bq=64, Bk=64):
    """Tiled attention with online softmax. Never builds more than a (Bq, Bk) block of scores."""
    T, d = q.shape; scale = 1 / math.sqrt(d)
    out = torch.empty_like(q); peak_block = 0
    for i in range(0, T, Bq):
        qi = q[i:i + Bq] * scale
        m = torch.full((len(qi), 1), float("-inf"), device=q.device)       # running row max
        l = torch.zeros(len(qi), 1, device=q.device)                        # running softmax denominator
        o = torch.zeros(len(qi), d, device=q.device)                        # running (unnormalised) output
        for j in range(0, (i + Bq) if causal else T, Bk):                   # causal: skip tiles entirely in the future
            S = qi @ k[j:j + Bk].T                                          # one (Bq, Bk) tile of scores
            if causal:
                qpos = torch.arange(i, i + len(qi), device=q.device)[:, None]; kpos = torch.arange(j, j + len(S[0]), device=q.device)[None]
                S = S.masked_fill(kpos > qpos, float("-inf"))
            peak_block = max(peak_block, S.numel())
            m_new = torch.maximum(m, S.max(-1, keepdim=True).values)
            P = torch.exp(S - m_new)
            correction = torch.exp(m - m_new)                               # rescale what we accumulated so far
            l = correction * l + P.sum(-1, keepdim=True)
            o = correction * o + P @ v[j:j + Bk]
            m = m_new
        out[i:i + Bq] = o / l
    return out, peak_block

T, d = 1024, 64
q, k, v = (torch.randn(T, d, device=device) for _ in range(3))
ref = naive_attention(q, k, v)
fa, peak = flash_attention(q, k, v)
print(f"max |flash - naive| = {(fa - ref).abs().max().item():.2e}   (exact up to float rounding)")
print(f"largest score block held at once: {peak:,} floats vs {T * T:,} for the full matrix ({T * T / peak:.0f}x smaller)")
#%% md
## The online softmax, step by step
Processing a row of scores in chunks while correcting earlier chunks with $e^{m_{old}-m_{new}}$ gives exactly the same softmax-weighted sum as the full computation:
#%%
s = torch.randn(12) * 3; vals = torch.randn(12)
full = (s.softmax(0) * vals).sum()
m, l, acc = float("-inf"), 0.0, 0.0
for chunk in range(0, 12, 4):
    sc, vc = s[chunk:chunk + 4], vals[chunk:chunk + 4]
    m_new = max(m, sc.max().item())
    l = math.exp(m - m_new) * l + torch.exp(sc - m_new).sum().item()
    acc = math.exp(m - m_new) * acc + (torch.exp(sc - m_new) * vc).sum().item()
    m = m_new
    print(f"after chunk {chunk // 4 + 1}: running max {m:.3f}, running estimate {acc / l:.5f}")
print(f"full softmax result: {full.item():.5f}")
#%% md
## Memory scaling: why it matters
The memory a naive implementation needs just for the score matrix, per head, in fp16, against what a tiled kernel holds:
#%%
for T in [2_048, 32_768, 131_072, 1_048_576]:
    naive_gb = T * T * 2 / 1e9; tile_mb = 128 * 128 * 2 / 1e6
    print(f"T = {T:>9,}: naive scores {naive_gb:10.2f} GB per head | FlashAttention tile {tile_mb:.3f} MB + O(T) running stats")
#%% md
## In practice: `F.scaled_dot_product_attention`
PyTorch dispatches to a FlashAttention or memory-efficient kernel when one is available for your device. Timing (first run includes warm-up):
#%%
def bench(fn, reps=10):
    fn()
    if device.type == "cuda": torch.cuda.synchronize()
    if device.type == "mps": torch.mps.synchronize()
    t = time.time()
    for _ in range(reps): fn()
    if device.type == "cuda": torch.cuda.synchronize()
    if device.type == "mps": torch.mps.synchronize()
    return (time.time() - t) / reps * 1000
B, H, T, d = 4, 8, 2048, 64
Q, K, Vv = (torch.randn(B, H, T, d, device=device) for _ in range(3))
t_naive = bench(lambda: naive_attention(Q, K, Vv))
t_sdpa = bench(lambda: F.scaled_dot_product_attention(Q, K, Vv, is_causal=True))
print(f"naive attention {t_naive:.1f} ms | F.scaled_dot_product_attention {t_sdpa:.1f} ms  (device: {device})")
print("max difference:", (naive_attention(Q, K, Vv) - F.scaled_dot_product_attention(Q, K, Vv, is_causal=True)).abs().max().item())
