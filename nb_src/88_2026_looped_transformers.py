#%% md
# 2026 — Looped Transformers: Reusing Depth

**Background and 2026 models:** Universal Transformer (Dehghani et al., 2018); recurrent-depth LMs (Geiping et al., 2025); Ouro (48 blocks × 4 passes, 2025); Mixture-of-Recursions (2025). In 2026: **Nanbeige4.2-3B** (July: 22 blocks each run twice, so 44 block applications with no extra weights), **LOTUS** (arXiv 2606.31779: looped latent reasoning within 1–1.5 points of explicit chain-of-thought), SMELT (September), and **Dory 1.0** (4B, October), the looped model in Raschka's October roundup. Reports that OpenAI's **GPT-6 Astra** (September 2026) is looped remain unconfirmed rumour; Raschka's own verdict was "likely" but not proven.

A looped transformer applies the **same block several times**:
$$h^{(0)} = e(x),\qquad h^{(i+1)} = \operatorname{Block}_\theta\big(h^{(i)} + e(x)\big),\quad i = 0..R-1$$
(re-injecting the input embedding each pass is the usual stabiliser). It decouples **compute depth** ($R$ passes) from **parameter count** (one block).

What it does and doesn't buy (Raschka, *OpenAI Astra and Looped Transformers*, September 2026):
- **Parameters / weight memory:** shrink by $R$×.
- **FLOPs per token:** unchanged vs. an unrolled model of the same depth.
- **KV cache:** *not* saved. Each pass has its own keys and values, so the cache is as large as the unrolled model's.
- **Upside:** the same weights can be run for **more passes at test time**, a form of inference-time scaling, and iterative algorithms fit the structure naturally.

**This notebook:** multi-hop pointer chasing, a task whose difficulty grows with the number of sequential steps. We compare a shallow model, a deep model and a looped model, then add loops at test time.
#%%
{{SETUP}}
#%% md
## The task: follow a permutation for k hops
The context lists a random permutation of 8 symbols as pairs `a → π(a)` in shuffled order. Then come queries `(start, hop-k)`, and the answer is $\pi^k(\text{start})$. Each hop needs one look-up (find the pair whose left side is the current symbol), so $k$ hops need roughly $k$ sequential attention steps.
#%%
NS, MAXH, NQ = 8, 6, 4
HOP0 = NS; SEP = HOP0 + MAXH; Vtok = SEP + 1
def hop_batch(B, hops=(1, 4)):
    perm = torch.rand(B, NS).argsort(1)
    order = torch.rand(B, NS).argsort(1)
    pairs = torch.stack([order, perm.gather(1, order)], 2).view(B, 2 * NS)
    start = torch.randint(0, NS, (B, NQ)); k = torch.randint(hops[0], hops[1] + 1, (B, NQ))
    ans = start.clone()
    for step in range(MAXH):
        ans = torch.where(k > step, perm.gather(1, ans), ans)
    x = torch.cat([pairs, torch.full((B, 1), SEP), torch.stack([start, HOP0 + k - 1], 2).view(B, 2 * NQ)], 1)
    return x.to(device), ans.to(device), k.to(device)
q_pos = torch.arange(2 * NS + 2, 2 * NS + 1 + 2 * NQ, 2, device=device)        # the hop-token positions
x, y, k = hop_batch(1); print("tokens:", x[0].tolist(), "| hops", k[0].tolist(), "-> answers", y[0].tolist())
#%%
class Block(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__(); self.n1, self.n2, self.h = nn.LayerNorm(d), nn.LayerNorm(d), heads
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        x = x + self.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.n2(x))

class Net(nn.Module):
    """n_unique distinct blocks, applied `loops` times in sequence (loops=1: ordinary transformer)."""
    def __init__(self, n_unique, loops=1, d=96, T=2 * NS + 1 + 2 * NQ):
        super().__init__(); self.loops = loops
        self.tok, self.pos = nn.Embedding(Vtok, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(n_unique)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, NS)
    def forward(self, x, loops=None):
        e = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device)); h = e
        for i in range(loops or self.loops):
            if i > 0: h = h + e                                                       # input injection before every re-entry
            for b in self.blocks: h = b(h)
        return self.head(self.ln(h))[:, q_pos]

def train(model, steps=3000, lr=1e-3):
    seed_everything(0); model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1); t0 = time.time()
    for s in range(steps):
        x, y, _ = hop_batch(128); loss = F.cross_entropy(model(x).reshape(-1, NS), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return model.eval(), time.time() - t0

@torch.no_grad()
def acc_by_hops(model, loops=None):
    x, y, k = hop_batch(4000, hops=(1, MAXH)); pred = model(x, loops).argmax(-1)
    return [((pred == y) & (k == h)).sum().item() / (k == h).sum().item() for h in range(1, MAXH + 1)]
#%% md
## Shallow vs. deep vs. looped
Trained on 1–4 hops. *Shallow*: 2 distinct blocks. *Deep*: 8 distinct blocks (4× the parameters). *Looped*: 2 distinct blocks run 4 times, so the same compute as Deep with Shallow's parameters.
#%%
models = {"shallow: 2 blocks": Net(2), "deep: 8 blocks": Net(8), "looped: 2 blocks × 4": Net(2, loops=4)}
for name, m in models.items():
    _, sec = train(m)
    accs = acc_by_hops(m)
    print(f"{name:22s} params {sum(p.numel() for p in m.parameters()) / 1e3:5.0f}K | accuracy by hops 1..6: " + " ".join(f"{a:.2f}" for a in accs) + f"   ({sec:.0f}s)")
#%% md
Hops 5 and 6 were never seen in training (and their hop tokens were never trained either), so expect failures there for every model; that is an extrapolation question, not a capacity one.
#%% md
## Inference-time scaling: more loops at test time
A looped model can be run for more passes than it was trained with, at no parameter cost. Does accuracy on the trained hop range hold or improve with more loops, and does it help the unseen hop counts? (An unrolled model has no such knob.)
#%%
m = models["looped: 2 blocks × 4"]
loop_range = [1, 2, 3, 4, 5, 6, 8, 12]
grid = np.array([acc_by_hops(m, loops=L) for L in loop_range])
for L, row in zip(loop_range, grid): print(f"loops {L:2d}: " + " ".join(f"{a:.2f}" for a in row))
plt.figure(figsize=(8, 4))
for h in range(4): plt.plot(loop_range, grid[:, h], "o-", label=f"{h + 1} hop(s)")
plt.axvline(4, c="gray", ls=":", label="trained with 4 loops"); plt.xlabel("loops at test time"); plt.ylabel("accuracy"); plt.legend(); plt.title("Looped model: compute at test time"); plt.show()
#%% md
## Training with a random number of loops
Recurrent-depth models (Geiping et al. 2025; Ouro) sample the loop count during training, so the model learns to **refine** its state and stay stable as loops increase, instead of relying on one exact depth. Train the same looped network with loops ∈ {2..6} per batch and re-check.
#%%
seed_everything(0); m2 = Net(2, loops=4).to(device)
opt = torch.optim.AdamW(m2.parameters(), lr=1e-3, weight_decay=0.01); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=3000, pct_start=0.1)
for s in range(3000):
    x, y, _ = hop_batch(128); loss = F.cross_entropy(m2(x, loops=random.randint(2, 6)).reshape(-1, NS), y.reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0); opt.step(); sched.step()
m2.eval(); grid2 = np.array([acc_by_hops(m2, loops=L) for L in loop_range])
for L, row in zip(loop_range, grid2): print(f"loops {L:2d}: " + " ".join(f"{a:.2f}" for a in row))
#%% md
## Cost accounting
#%%
d, T = 96, 2 * NS + 1 + 2 * NQ
for name, (uniq, loops) in {"shallow 2": (2, 1), "deep 8": (8, 1), "looped 2×4": (2, 4), "looped 2×8 (test time)": (2, 8)}.items():
    applications = uniq * loops
    print(f"{name:24s} block weights {uniq:2d} | block applications (≈FLOPs) {applications:2d} | KV-cache entries per token {applications:2d}")
#%% md
## Takeaways
- On a task that needs sequential steps, depth matters: the shallow model fails beyond a hop or two. The looped model gets the deep model's *compute* with the shallow model's *parameters*, and typically lands near the deep model.
- Looping saves weights, not FLOPs and not KV cache, the point Raschka stressed when the Astra rumours spread.
- Extra loops at test time are not free accuracy: a model trained with a fixed loop count may degrade when run longer. Training with a **random loop count** makes extra loops safe and sometimes helpful; this is the recipe used by recurrent-depth LMs.
- Extrapolating to unseen hop counts is hard for all models here; looping is not a general solution to length/depth generalisation at this scale.
