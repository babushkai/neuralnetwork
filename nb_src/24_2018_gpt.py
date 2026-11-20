#%% md
# 2018 — GPT: Generative Pre-Training (Decoder-Only Transformers)

**Papers:** Alec Radford et al., *"Improving Language Understanding by Generative Pre-Training"* (GPT, 2018); *"Language Models are Unsupervised Multitask Learners"* (GPT-2, 2019); Brown et al., *"Language Models are Few-Shot Learners"* (GPT-3, 2020).

Take only the **decoder** of the Transformer (causal self-attention, no cross-attention) and train it on one objective, **next-token prediction**:

$$\mathcal L = -\sum_t \log p_\theta(x_t \mid x_{<t})$$

That simple objective, scaled up, is the basis of today's large language models. Changes from 2017 to GPT-2 that we use here:
- **Pre-LayerNorm** (normalise *before* each sub-layer), which is more stable for deep stacks
- **Learned positional embeddings** and **GELU** activations
- **Weight tying** between the input embedding and the output layer

A tiny character-level GPT trained on a ~2 KB corpus. It will mostly memorise; the point is the mechanics.
#%%
{{SETUP}}
#%%
{{CORPUS}}
chars = sorted(set(CORPUS)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in CORPUS], device=device)
encode = lambda s: [stoi[c] for c in s]; decode = lambda ids: "".join(chars[i] for i in ids)
#%%
class CausalSelfAttention(nn.Module):
    def __init__(self, d, h, ctx, p):
        super().__init__()
        self.qkv, self.proj, self.h = nn.Linear(d, 3 * d), nn.Linear(d, d), h
        self.drop = nn.Dropout(p)
        self.register_buffer("mask", torch.tril(torch.ones(ctx, ctx, dtype=torch.bool)))
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // self.h)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf")).softmax(-1)
        return self.proj(self.drop(att @ v).transpose(1, 2).reshape(B, T, D))

class Block(nn.Module):
    def __init__(self, d, h, ctx, p):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, h, ctx, p)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(p))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))          # pre-norm residual blocks
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, V, d=128, h=4, n=4, ctx=64, p=0.1):
        super().__init__()
        self.ctx = ctx
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(ctx, d)
        self.blocks = nn.Sequential(*[Block(d, h, ctx, p) for _ in range(n)])
        self.ln_f, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)
        self.head.weight = self.tok.weight                       # weight tying
        self.apply(lambda m: nn.init.normal_(m.weight, std=0.02) if isinstance(m, (nn.Linear, nn.Embedding)) else None)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        return self.head(self.ln_f(self.blocks(x)))

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8, top_k=10):
        for _ in range(n):
            logits = self(idx[:, -self.ctx:])[:, -1] / temperature
            v, _ = logits.topk(top_k); logits[logits < v[:, -1:]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(logits.softmax(-1), 1)], 1)
        return idx
#%%
model = GPT(V).to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}K")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
steps, ctx, B = 1500, 64, 32
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
losses = []
for step in range(steps):
    model.train()
    ix = torch.randint(0, len(data) - ctx - 1, (B,), device=device)
    x = torch.stack([data[i:i + ctx] for i in ix]); y = torch.stack([data[i + 1:i + ctx + 1] for i in ix])
    loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    losses.append(loss.item())
    if step % 300 == 0: print(f"step {step:4d}  loss {loss.item():.3f}")
plt.plot(losses); plt.yscale("log"); plt.xlabel("step"); plt.ylabel("next-char cross-entropy"); plt.show()
#%%
model.eval()
for prompt in ["Attention ", "The perceptron ", "Today the "]:
    out = model.generate(torch.tensor([encode(prompt)], device=device), 200)
    print(repr(decode(out[0].tolist())), "\n")
#%% md
## Takeaways
- Decoder-only + next-token prediction + scale = GPT. The architecture above is essentially GPT-2 at toy size.
- The **scaling laws** (Kaplan et al. 2020; Chinchilla, Hoffmann et al. 2022) found that loss falls predictably as a power law in parameters, data and compute.
- See `38_2026_modern_llm_recipe` for how this block evolved (RMSNorm, RoPE, GQA, SwiGLU, ...).
