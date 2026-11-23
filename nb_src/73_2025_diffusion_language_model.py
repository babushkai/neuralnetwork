#%% md
# 2025 — Diffusion Language Models (Masked Diffusion, LLaDA)

**Papers:** Austin et al., *D3PM* (2021); Sahoo et al., *MDLM* (2024); Shen Nie et al., *"Large Language Diffusion Models"* (**LLaDA**, February 2025), an 8B model competitive with LLaMA-3 8B. In production: Inception Labs' *Mercury* (2025) and Google's *Gemini Diffusion* (May 2025) generate text with diffusion, many tokens per step.

Autoregressive LMs write strictly left to right, one token per forward pass. A **masked diffusion LM** instead:
- **forward process:** pick $t\sim U(0,1)$ and replace each token with `[MASK]` independently with probability $t$
- **model:** a *bidirectional* Transformer (no causal mask) predicts every masked token at once
- **loss:** cross-entropy on the masked positions, weighted by $1/t$, which is an upper bound on the negative log-likelihood:
$$\mathcal L = \mathbb E_{t, x_0}\Big[\frac1t \sum_{i:\,x_t^i=\text{[M]}} -\log p_\theta(x_0^i \mid x_t)\Big]$$
- **sampling:** start from an all-`[MASK]` sequence and, over $K$ steps, unmask the tokens the model is most confident about (LLaDA's *low-confidence remasking*)

Parallel decoding can be much faster, and the model sees **both directions**, so it can **infill**: fill a gap given text on both sides, which a left-to-right model cannot do directly.
#%% md
![Diagram: Masked diffusion language model (LLaDA, Nie et al., 2025)](https://raw.githubusercontent.com/babushkai/neuralnetwork/main/assets/diagrams/73_2025_diffusion_language_model.png)
#%%
{{SETUP}}
#%%
{{PROCEDURAL}}
MASK = V; VM = V + 1                                                                    # extra [MASK] token id
L = 64
#%%
class Denoiser(nn.Module):
    def __init__(self, d=192, n_layers=6, heads=6):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VM, d), nn.Parameter(torch.randn(1, L, d))   # same scale as token embeddings: a bidirectional model needs position from the start
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)   # no causal mask: bidirectional
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, V)
    def forward(self, x): return self.head(self.ln(self.body(self.tok(x) + self.pos)))

def get_batch(src, B=64):
    ix = torch.randint(0, len(src) - L, (B,), device=device)
    return src[ix[:, None] + torch.arange(L, device=src.device)]

def diffusion_loss(model, x0):
    t = torch.rand(len(x0), 1, device=device).clamp(min=1e-3)
    masked = torch.rand(x0.shape, device=device) < t                                     # each token masked with prob t
    xt = torch.where(masked, torch.full_like(x0, MASK), x0)
    ce = F.cross_entropy(model(xt).transpose(1, 2), x0, reduction="none")              # (B, L)
    return ((ce * masked) / t).sum() / masked.numel()

seed_everything(0)
model = Denoiser().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
steps = 2500
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05)
t0 = time.time()
for s in range(steps):
    loss = diffusion_loss(model, get_batch(train_data))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if s % 500 == 0: print(f"step {s:4d}  loss {loss.item():.3f}")
print(f"trained in {time.time() - t0:.0f}s")
#%% md
## Sampling by iterative unmasking
#%%
@torch.no_grad()
def sample(n_steps, x=None, temperature=0.7, show=False):
    model.eval()
    x = torch.full((1, L), MASK, device=device) if x is None else x.clone()
    n_masked = (x == MASK).sum().item()
    for k in range(n_steps):
        masked = x == MASK
        if not masked.any(): break
        probs = F.softmax(model(x) / temperature, -1)
        pred = torch.multinomial(probs[0], 1).T                                           # a guess for every position
        conf = probs.gather(-1, pred[..., None]).squeeze(-1).masked_fill(~masked, -1.0)
        n_unmask = math.ceil(n_masked * (k + 1) / n_steps) - (n_masked - masked.sum().item())
        keep = conf[0].topk(max(1, n_unmask)).indices                                    # commit the most confident guesses
        x[0, keep] = pred[0, keep]
        if show and k in (0, n_steps // 4, n_steps // 2, n_steps - 1):
            print(f"step {k + 1:3d}: " + "".join("_" if t == MASK else chars[t] for t in x[0].tolist()))
    return x

seed_everything(1)
print("Watching one sequence appear (64 tokens, 16 denoising steps):")
_ = sample(16, show=True)
#%%
print("Samples with different numbers of denoising steps (fewer steps = more tokens committed per forward pass):")
for n_steps in [4, 8, 16, 64]:
    seed_everything(2)
    print(f"{n_steps:3d} steps ({L / n_steps:4.1f} tokens/pass): " + decode(sample(n_steps)[0].tolist()))
#%% md
## Infilling: text on both sides
Give the model a prefix *and* a suffix and let it fill the middle. A bidirectional denoiser does this natively.
#%%
def infill(prefix, suffix, n_steps=16):
    gap = L - len(prefix) - len(suffix)
    x = torch.tensor([encode(prefix) + [MASK] * gap + encode(suffix)], device=device)
    out = decode(sample(n_steps, x)[0].tolist())
    return prefix + "[" + out[len(prefix):len(prefix) + gap] + "]" + suffix
for pre, suf in [("The teacher ", " because the router copies its weights. "),
                 ("A student learns ", " step by step. Each layer forgets the loss. ")]:
    seed_everything(3); print(infill(pre, suf))
#%% md
## Takeaways
- The same Transformer, trained to **denoise masked text** instead of to continue it, becomes a generator that writes many tokens per step and can fill gaps.
- At scale the trade-offs are still being worked out (quality per step, KV caching, variable-length output), but 2025 showed diffusion LMs can be competitive, very fast, and good at editing and infilling.
