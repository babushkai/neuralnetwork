#%% md
# 2023 — Sparse Autoencoders: Reading Features out of a Network

**Papers:** Nelson Elhage et al., *"Toy Models of Superposition"* (Anthropic, 2022); Trenton Bricken et al., *"Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"* (Anthropic, 2023); Hoagy Cunningham et al., *"Sparse Autoencoders Find Highly Interpretable Features in Language Models"* (2023); Adly Templeton et al., *"Scaling Monosemanticity"* (Anthropic, 2024). Dictionary learning itself goes back to Olshausen & Field (1996).

Every other notebook asks how to make networks better. This one asks **what they have learned**. Individual neurons are often **polysemantic**: one neuron fires for unrelated things. The **superposition hypothesis** explains why. A layer with $d$ neurons can represent many more than $d$ features as *directions* in activation space, as long as the features are **sparse** (rarely active at the same time). Nearly orthogonal directions interfere only a little, and the network accepts that small cost to store more.

If so, the features are recoverable by **dictionary learning**: find an overcomplete set of directions such that each activation vector is a sparse combination of them. A **sparse autoencoder (SAE)** does this:

$$f = \operatorname{ReLU}(W_{enc}(x - b_{dec}) + b_{enc}), \qquad \hat x = W_{dec} f + b_{dec}, \qquad \mathcal L = \lVert x-\hat x\rVert^2 + \lambda\lVert f\rVert_1$$

with many more latents than input dimensions. In 2024, SAEs with millions of latents on Claude 3 Sonnet found features for concepts such as the Golden Gate Bridge, code vulnerabilities and sycophancy, and turning them up or down steered the model's behaviour.
#%% md
![Diagram: Sparse autoencoders for interpretability (2023–2024)](https://raw.githubusercontent.com/babushkai/neuralnetwork/main/assets/diagrams/65_2023_sparse_autoencoders.png)
#%%
{{SETUP}}
#%%
class SAE(nn.Module):
    """ReLU SAE with an L1 penalty, or (k given) a TopK SAE that keeps only the k largest latents (Gao et al., 2024)."""
    def __init__(self, d, n_latents, k=None):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(n_latents, d) * 0.1); self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.W_dec = nn.Parameter(self.W_enc.data.T.clone()); self.b_dec = nn.Parameter(torch.zeros(d)); self.k = k
    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        if self.k is None: return F.relu(pre)
        v, i = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, i, F.relu(v))
    def forward(self, x):
        f = self.encode(x); return f @ self.W_dec.T + self.b_dec, f
    def normalize(self):
        with torch.no_grad(): self.W_dec /= self.W_dec.norm(dim=0, keepdim=True)  # unit-norm dictionary directions

def train_sae(acts, n_latents, l1=0.0, k=None, steps=8000, lr=1e-3, B=512, seed=0):
    seed_everything(seed); sae = SAE(acts.shape[1], n_latents, k).to(acts.device); opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for s in range(steps):
        x = acts[torch.randint(0, len(acts), (B,), device=acts.device)]
        xh, f = sae(x); loss = (xh - x).pow(2).sum(-1).mean() + l1 * f.abs().sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sae.normalize()
    with torch.no_grad():
        xh, f = sae(acts); fve = 1 - (xh - acts).pow(2).sum() / (acts - acts.mean(0)).pow(2).sum()
    return sae, fve.item(), (f > 1e-3).float().sum(-1).mean().item()
#%% md
## Part 1: superposition in a controlled setting
Make activations exactly as the hypothesis describes: 64 ground-truth features, stored as random directions in a **16-dimensional** space. Each example activates each feature independently with probability 3%, with a random positive magnitude. Can we get the 64 directions back from the 16-d activations alone?
#%%
seed_everything(0)
d, n_feat = 16, 64
true_dirs = F.normalize(torch.randn(n_feat, d), dim=-1).to(device)
def sample_acts(n, p=0.03):
    on = (torch.rand(n, n_feat, device=device) < p).float()
    return (on * torch.rand(n, n_feat, device=device)) @ true_dirs
acts = sample_acts(100_000)
print(f"{n_feat} features in {d} dimensions; max |cos| between two true directions {(true_dirs @ true_dirs.T - torch.eye(n_feat, device=device)).abs().max():.2f}")

def recovered(dirs, thr=0.9):
    """For each true feature: best |cosine| with any learned direction; count how many exceed thr."""
    cos = (F.normalize(dirs, dim=-1) @ true_dirs.T).abs().max(0).values
    return (cos > thr).sum().item(), cos.mean().item()
U, S_, Vt = torch.linalg.svd(acts - acts.mean(0), full_matrices=False)
print(f"PCA ({d} components):        recovers {recovered(Vt)[0]:2d}/{n_feat} features (mean best cosine {recovered(Vt)[1]:.2f})")
neurons = torch.eye(d, device=device)
print(f"the {d} neurons themselves:  recovers {recovered(neurons)[0]:2d}/{n_feat} features (mean best cosine {recovered(neurons)[1]:.2f})")
for l1, k in [(0.0, None), (0.03, None), (0.3, None), (0.0, 2), (0.0, 4), (0.0, 8)]:
    sae, fve, l0 = train_sae(acts, 128, l1=l1, k=k)
    n, m = recovered(sae.W_dec.T)
    name = f"L1 SAE, λ = {l1:<4}" if k is None else f"TopK SAE, k = {k}  "
    print(f"{name}: recovers {n:2d}/{n_feat} features (mean best cosine {m:.2f}) | variance explained {fve:.3f} | active latents per input {l0:.1f}")
#%% md
PCA can only return 16 orthogonal directions, and the raw neurons are mixtures, so neither finds the features. The SAEs show that **the amount of sparsity is what matters**. The data has about 2 active features per input. A TopK SAE that keeps exactly 2 (or 4) latents recovers every planted direction, at the cost of reconstructing a bit less of the variance; keeping 8 is already too loose. With an L1 penalty, a small $\lambda$ leaves dozens of latents active and recovers *fewer* features than no penalty at all; only a large $\lambda$ pushes activity down to a few latents and recovers most of them. Reconstruction quality alone is a poor guide: the best reconstructions come from the least interpretable dictionaries.
## Part 2: features inside a small language model
Train a small character-level GPT on the procedural corpus, collect its MLP activations (512-d) in the last block, and train a TopK SAE with 2,048 latents (16 active per position) on them. Then, for a few latents, list the contexts where they fire most strongly.
#%%
{{PROCEDURAL}}
#%%
{{TINYGPT}}
#%%
seed_everything(0); t0 = time.time()
gpt = train_gpt(GPT(V, ctx=64), train_data, steps=1000); print(f"GPT val loss {eval_loss(gpt, val_data):.3f} ({time.time() - t0:.0f}s)")
store = {}
hook = gpt.blocks[-1].mlp[1].register_forward_hook(lambda m, i, o: store.__setitem__("h", o.detach()))
with torch.no_grad():
    starts = torch.arange(0, 60_000, 64, device=device)
    windows = torch.stack([data[s:s + 64] for s in starts.tolist()])
    gpt(windows); H = store["h"].reshape(-1, store["h"].shape[-1])          # one activation vector per character position
hook.remove()
print(f"{len(H):,} activation vectors of size {H.shape[1]}")
H = H / H.norm(dim=-1).mean()                                                # normalise scale so λ is meaningful
sae, fve, l0 = train_sae(H, 2048, k=16, steps=6000)
with torch.no_grad(): Fz = sae.encode(H)
alive = (Fz > 0).any(0)
print(f"SAE: variance explained {fve:.3f}, {l0:.1f} active latents per position on average, {alive.sum().item()} of 2048 latents ever active")
#%%
flat = windows.reshape(-1)
def top_contexts(j, k=6, width=24):
    vals, pos = Fz[:, j].topk(k)
    out = []
    for v, p in zip(vals.tolist(), pos.tolist()):
        a = max(0, p - width); s = decode(flat[a:p].tolist()); c = decode([flat[p].item()])
        out.append(f"  {v:5.2f}  …{s}[{c}]")
    return out
freq = (Fz > 0).float().mean(0)
cands = torch.where((freq > 0.002) & (freq < 0.05))[0]                         # latents that fire neither always nor never
seed_everything(1); show = cands[torch.randperm(len(cands))[:6]].tolist()
for j in show:
    print(f"latent {j} (active on {freq[j]:.1%} of positions); strongest activations, the [character] is where it fires:")
    print("\n".join(top_contexts(j)))
#%% md
## How monosemantic are the latents? Compare with raw neurons
A rough automatic test: for each unit, take its 50 strongest activations and ask how often the **next character to be predicted** is the same. A unit that encodes one thing ("about to write 'carefully'") should score high; a polysemantic neuron mixes several.
#%%
nxt = torch.cat([flat[1:], flat[:1]])
def purity(acts_matrix, units):
    scores = []
    for j in units:
        idx = acts_matrix[:, j].topk(50).indices
        scores.append(torch.bincount(nxt[idx], minlength=V).max().item() / 50)
    return np.mean(scores)
neuron_ids = torch.randperm(H.shape[1])[:200].tolist(); latent_ids = cands[torch.randperm(len(cands))[:200]].tolist()
print(f"next-character purity of top activations: raw MLP neurons {purity(H, neuron_ids):.2f} | SAE latents {purity(Fz, latent_ids):.2f}")
#%% md
## Takeaways
- **Superposition:** networks store more features than they have neurons, as sparse, nearly orthogonal directions. That is why single neurons are hard to interpret.
- **Sparse dictionary learning recovers them, if the sparsity is right.** In the controlled test a TopK SAE with $ near the true number of active features recovers all 64 planted directions, which PCA and the raw neurons cannot; a weak L1 penalty does worse than none. On the small LM, TopK latents are purer than raw neurons by a simple next-character test (0.85 vs. 0.77), and individual latents read cleanly (one fires on "the next [t]", another on the space in "then the[ ]"). This corpus is highly templated, so treat that as a sanity check, not evidence about large models.
- **Caveats** (all active research): the number of latents and λ change which features you get ("feature splitting"); some reconstruction error remains unexplained ("dark matter"); and an interpretable latent is a hypothesis to test, for example by **steering** (adding the decoder direction to the activations) and checking that behaviour changes as predicted. Variants in use include TopK SAEs (OpenAI, 2024), JumpReLU SAEs (Gemma Scope, 2024) and transcoders / cross-layer transcoders for circuit tracing (Anthropic, 2025).
