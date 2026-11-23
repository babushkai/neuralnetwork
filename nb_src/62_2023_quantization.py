#%% md
# 2023 — Quantization: Running Big Models in Low Precision

**Papers:** Dettmers et al., *"LLM.int8()"* (2022), the outlier features; Elias Frantar et al., *"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"* (2022/ICLR 2023); Lin et al., *AWQ* (2023); Dettmers et al., *QLoRA* (2023), which introduced 4-bit NF4 and finetuning on top. In 2025–2026, models also ship or train natively in FP8 and FP4 (e.g. DeepSeek-V3/V4, gpt-oss MXFP4).

Weights stored in 16 bits take 2 bytes per parameter. A 70B model needs 140 GB just to load. Storing weights in **8, 4 or even 3 bits** cuts memory and memory bandwidth (the real bottleneck when generating) 2–5×.

Quantization maps each weight to one of $2^b$ levels: $\hat w = s \cdot \operatorname{round}(w / s)$, with scale $s = \max|w| / (2^{b-1}-1)$. Key choices:
- **Granularity:** one scale per tensor, per output channel, or per **group** of 32–128 weights. Finer is more accurate.
- **Outliers:** a few huge values stretch the scale and wipe out precision for everything else.
- **GPTQ:** quantize the columns of a weight matrix one at a time, and **adjust the not-yet-quantized columns to compensate** for each rounding error, using second-order information $H = X^\top X$ from a little calibration data.
#%% md
![Diagram: Quantization (LLM.int8, GPTQ and others, 2022–2023)](https://raw.githubusercontent.com/babushkai/neuralnetwork/main/assets/diagrams/62_2023_quantization.png)
#%%
{{SETUP}}
import copy
#%%
{{PROCEDURAL}}
#%%
{{TINYGPT}}

seed_everything(0)
model = train_gpt(GPT(V, d=128, n_layers=4, heads=4), train_data, steps=800)
base_loss = eval_loss(model, val_data)
print(f"full-precision (fp32) model: val loss {base_loss:.4f}")
#%% md
## Round-to-nearest (RTN) at different granularities
#%%
def quantize(w, bits, group=None, per_channel=True):
    """Symmetric round-to-nearest. w: (out, in). group: quantize each run of `group` input weights with its own scale."""
    qmax = 2 ** (bits - 1) - 1
    if group:
        wg = w.reshape(w.shape[0], -1, group); s = wg.abs().amax(-1, keepdim=True) / qmax
        return (torch.round(wg / s).clamp(-qmax - 1, qmax) * s).reshape_as(w)
    s = (w.abs().amax(1, keepdim=True) if per_channel else w.abs().max()) / qmax
    return torch.round(w / s).clamp(-qmax - 1, qmax) * s

def quantized_copy(fn):
    m = copy.deepcopy(model)
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and name != "head":
            mod.weight.data = fn(mod.weight.data)
    return m

rows = []
for bits in [8, 4, 3]:
    for label, kw in [("per-tensor", dict(per_channel=False)), ("per-channel", {}), ("group-32", dict(group=32))]:
        L = eval_loss(quantized_copy(lambda w: quantize(w, bits, **kw)), val_data); rows.append((bits, label, L))
        print(f"int{bits} {label:12s} val loss {L:.4f}  (+{L - base_loss:.4f})")
#%% md
## The outlier problem
LLMs develop a few **outlier channels** with huge magnitudes (LLM.int8() found them in every model above ~6.7B parameters). We plant one large weight per output row and look at what happens to everything else in the row.
#%%
w = model.blocks[0].mlp[0].weight.data.clone()
w_out = w.clone(); w_out[:, 5] *= 40                                                  # one outlier input channel
for name, W in [("normal weights", w), ("with an outlier channel", w_out)]:
    err_tensor = (quantize(W, 4, per_channel=False) - W)[:, 40:].abs().mean().item()        # columns outside the outlier's group
    err_group = (quantize(W, 4, group=32) - W)[:, 40:].abs().mean().item()
    print(f"{name:26s} int4 error on the *other* weights: per-tensor {err_tensor:.4f} | group-32 {err_group:.4f}")
print("Per-tensor scaling lets one outlier channel ruin every weight; with groups of 32, weights outside the outlier's group are unaffected.")
#%% md
## GPTQ: error-compensating quantization
For a linear layer $y = Wx$, minimise the output error $\|WX - \hat W X\|^2$ on calibration inputs $X$. With $H = 2XX^\top$, GPTQ goes column by column: quantize column $j$, then spread its error over the remaining columns using row $j$ of the inverse Hessian (computed once via a Cholesky factorisation).
#%%
@torch.no_grad()
def gptq(W, H, bits, group=32, damp=0.01):
    W = W.clone().float(); n = W.shape[1]
    H = H + damp * H.diag().mean() * torch.eye(n, device=W.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)   # upper Cholesky of H^-1
    Q = torch.zeros_like(W); qmax = 2 ** (bits - 1) - 1
    for j in range(n):
        if j % group == 0:                                                            # scale for the next group, from the updated weights
            s = W[:, j:j + group].abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax
        q = torch.round(W[:, j:j + 1] / s).clamp(-qmax - 1, qmax) * s
        Q[:, j:j + 1] = q
        err = (W[:, j:j + 1] - q) / Hinv[j, j]
        W[:, j + 1:] -= err * Hinv[j, j + 1:][None]                                   # compensate the remaining columns
    return Q

def collect_hessians(m, n_batches=8):
    """Calibration: record H = 2 X^T X of the input to every Linear layer."""
    H, hooks = {}, []
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and name != "head":
            H[name] = torch.zeros(mod.in_features, mod.in_features, device=device)
            def hook(mod, inp, out, name=name):                      # must return None: a returned value would replace the output
                x = inp[0].reshape(-1, inp[0].shape[-1]); H[name].add_(2 * x.T @ x)
            hooks.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        for _ in range(n_batches):
            ix = torch.randint(0, len(train_data) - m.ctx, (8,), device=device)
            m(train_data[ix[:, None] + torch.arange(m.ctx, device=train_data.device)])
    for h in hooks: h.remove()
    return H

Hs = collect_hessians(model)
def gptq_copy(bits):
    m = copy.deepcopy(model)
    for name, mod in m.named_modules():
        if name in Hs: mod.weight.data = gptq(mod.weight.data, Hs[name], bits)
    return m

print(f"{'bits':>5s} {'RTN group-32':>14s} {'GPTQ group-32':>14s}   (fp32 baseline {base_loss:.4f})")
for bits in [4, 3, 2]:
    rtn = eval_loss(quantized_copy(lambda w: quantize(w, bits, group=32)), val_data)
    gq = eval_loss(gptq_copy(bits), val_data)
    print(f"int{bits:<2d} {rtn:14.4f} {gq:14.4f}")
#%% md
## Memory
#%%
n = sum(p.numel() for p in model.parameters())
for label, b in [("fp32", 32), ("bf16", 16), ("int8", 8 + 0.5), ("int4 (group-32, 16-bit scales)", 4 + 0.5), ("int3 (group-32)", 3 + 0.5)]:
    print(f"{label:32s} {n * b / 8 / 1e6:6.2f} MB for this model | a 70B model would need {70e9 * b / 8 / 1e9:6.0f} GB")
