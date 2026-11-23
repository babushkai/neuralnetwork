#%% md
# 2026 — Pre-Training Without Learning-Rate Decay: Constant LR + Checkpoint Averaging

**Sources:** Sebastian Raschka, *"Pre-Training Without Learning Rate Decay"* (November 17, 2026), surveying recent recipes. **Kolibri-1** (78B, October 2026) and DatologyAI's models train with a **constant** learning rate and **average the last checkpoints** instead of decaying. Background: Warmup-Stable-Decay (WSD; MiniCPM 2024), "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations" (Hägele et al. 2024), and weight averaging (Polyak; LAWA; model soups).

Schedules used by 2026 open models (from Raschka's note):

| schedule | how it works | used by |
|---|---|---|
| **cosine** | warm up, then decay smoothly to ~0 by a horizon fixed in advance | Llama 3, OLMo 2/3, GLM-4.5/5, Kimi K3, Phi-4 |
| **WSD** | warm up, hold constant, decay quickly over the last ~10–20% | DeepSeek V3/V4, Kimi K2 / Kimi Linear, Nemotron 3, SmolLM3, MiniMax M2 |
| **constant + averaging** | never decay; report the *average of recent checkpoints* | Kolibri-1, DatologyAI |

**Why it works:** with a constant LR, SGD-type optimizers bounce around a valley floor with noise proportional to the LR. Decaying the LR shrinks the bouncing. **Averaging the iterates** cancels the bouncing directly, so you get most of the decay's benefit without lowering the LR and without fixing the run length in advance. Training can continue from the constant-LR weights at any time, so every checkpoint is "anytime".

We train the same small GPT with each schedule and track validation loss throughout.
#%% md
![Diagram: Constant learning rate + checkpoint averaging (2026)](https://raw.githubusercontent.com/babushkai/neuralnetwork/main/assets/diagrams/89_2026_constant_lr_checkpoint_averaging.png)
#%%
{{SETUP}}
#%%
{{PROCEDURAL}}
#%%
{{TINYGPT}}
#%%
STEPS, PEAK, WARM = 2000, 3e-3, 100
def lr_at(s, kind):
    if s < WARM: return PEAK * (s + 1) / WARM
    if kind == "cosine": return PEAK * 0.5 * (1 + math.cos(math.pi * (s - WARM) / (STEPS - WARM)))
    if kind == "wsd":
        d0 = int(0.8 * STEPS)
        return PEAK if s < d0 else PEAK * (1 - (s - d0) / (STEPS - d0))
    return PEAK                                                                     # constant

@torch.no_grad()
def val_loss(model, n=10):
    g = torch.Generator(device="cpu").manual_seed(123); losses = []
    model.eval()
    for _ in range(n):
        ix = torch.randint(0, len(val_data) - 129, (32,), generator=g).to(device)
        x = val_data[ix[:, None] + torch.arange(128, device=val_data.device)]; y = val_data[ix[:, None] + torch.arange(1, 129, device=val_data.device)]
        losses.append(F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item())
    model.train(); return float(np.mean(losses))

def run(kind, eval_every=100, avg_every=10, avg_window=400, ema_decay=0.995):
    seed_everything(0); model = GPT(V, d=128, n_layers=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK, weight_decay=0.1, betas=(0.9, 0.95))
    ema = [p.detach().clone() for p in model.parameters()]
    snaps = []                                                                        # checkpoints for the windowed average
    curves = {"raw": [], "ema": [], "window": []}; t0 = time.time()
    for s in range(STEPS):
        for g_ in opt.param_groups: g_["lr"] = lr_at(s, kind)
        ix = torch.randint(0, len(train_data) - 129, (32,), device=device)
        x = train_data[ix[:, None] + torch.arange(128, device=train_data.device)]; y = train_data[ix[:, None] + torch.arange(1, 129, device=train_data.device)]
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for e, p in zip(ema, model.parameters()): e.lerp_(p, 1 - ema_decay)
        if (s + 1) % avg_every == 0:
            snaps.append([p.detach().clone() for p in model.parameters()])
            snaps = snaps[-(avg_window // avg_every):]
        if (s + 1) % eval_every == 0:
            curves["raw"].append(val_loss(model))
            backup = [p.detach().clone() for p in model.parameters()]
            for name, weights in [("ema", ema), ("window", [torch.stack(ws).mean(0) for ws in zip(*snaps)])]:
                with torch.no_grad():
                    for p, w in zip(model.parameters(), weights): p.copy_(w)
                curves[name].append(val_loss(model))
            with torch.no_grad():
                for p, w in zip(model.parameters(), backup): p.copy_(w)
    return curves, time.time() - t0

results = {}
for kind in ("cosine", "wsd", "constant"):
    results[kind], sec = run(kind)
    c = results[kind]
    print(f"{kind:8s} final val loss: raw weights {c['raw'][-1]:.4f} | EMA {c['ema'][-1]:.4f} | average of last 400 steps {c['window'][-1]:.4f}   ({sec:.0f}s)")
#%%
steps_axis = np.arange(100, STEPS + 1, 100)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot([lr_at(s, "cosine") for s in range(STEPS)], label="cosine")
axes[0].plot([lr_at(s, "wsd") for s in range(STEPS)], label="WSD"); axes[0].plot([lr_at(s, "constant") for s in range(STEPS)], label="constant")
axes[0].set(xlabel="step", ylabel="learning rate", title="Schedules"); axes[0].legend()
axes[1].plot(steps_axis, results["cosine"]["raw"], label="cosine (raw)")
axes[1].plot(steps_axis, results["wsd"]["raw"], label="WSD (raw)")
axes[1].plot(steps_axis, results["constant"]["raw"], label="constant (raw)", alpha=0.6)
axes[1].plot(steps_axis, results["constant"]["window"], label="constant + checkpoint average", lw=2)
axes[1].plot(steps_axis, results["constant"]["ema"], "--", label="constant + EMA")
axes[1].set(xlabel="step", ylabel="validation loss", title="Validation loss during training", ylim=(None, results["constant"]["raw"][3])); axes[1].legend()
plt.tight_layout(); plt.show()
#%% md
## The "anytime" property
Suppose you must stop at step 1000 instead of 2000. A cosine schedule tuned for 2000 steps is still at a high LR at step 1000, so you'd have to rerun with a new horizon. Constant LR + averaging can hand over a good model at any point, and WSD only needs a short decay from any stable checkpoint.
#%%
i = 1000 // 100 - 1
print(f"validation loss at step 1000: cosine(2000-step horizon) raw {results['cosine']['raw'][i]:.4f} | constant raw {results['constant']['raw'][i]:.4f} | constant + average {results['constant']['window'][i]:.4f}")
#%% md
## Takeaways
- With a constant LR, the raw weights are noisy. Averaging recent checkpoints (or an EMA) removes most of that noise, and typically lands close to a decayed schedule's final loss, without having to decide the run length in advance.
- WSD is the middle ground most 2026 frontier labs chose: train at constant LR (so you can branch or extend anytime), then decay briefly at the end.
- The exact ranking between cosine, WSD and averaging at this toy scale depends on the seed and the averaging window. The robust observation is the size of the gap between "constant, raw" and "constant, averaged".
- This is a training-recipe change, not an architectural one. It made Raschka's list because it simplified how 2026 models like Kolibri-1 were trained.
