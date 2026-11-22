#%% md
# 2010 — Weight Initialization (Glorot/Xavier and He)

**Papers:** Xavier Glorot & Yoshua Bengio, *"Understanding the difficulty of training deep feedforward neural networks"*, AISTATS (2010); Kaiming He, Xiangyu Zhang, Shaoqing Ren & Jian Sun, *"Delving Deep into Rectifiers"*, ICCV (2015). Earlier: LeCun, Bottou, Orr & Müller, *"Efficient BackProp"* (1998).

Before 2006 deep networks were thought to be nearly untrainable, and layer-wise pretraining (notebook 15) was the workaround. Glorot & Bengio showed that much of the problem was simply **the scale of the random initial weights**.

Each layer computes $y = Wx$. If the weights have variance $\sigma^2$ and $x$ has $n_{in}$ inputs of variance $v$, then $\operatorname{Var}(y) = n_{in}\sigma^2 v$. Across $L$ layers the signal is multiplied by $(n_{in}\sigma^2)^L$: it **explodes or vanishes exponentially** unless $n_{in}\sigma^2 \approx 1$. The gradients flowing back obey the same rule with $n_{out}$.

- **Xavier/Glorot** (for tanh, roughly linear near 0): $\sigma^2 = \dfrac{2}{n_{in} + n_{out}}$, a compromise between the forward and the backward condition.
- **He/Kaiming** (for ReLU, which zeroes half its inputs and halves the variance): $\sigma^2 = \dfrac{2}{n_{in}}$.
#%%
{{SETUP}}
#%%
def make_mlp(depth, width, act, init, d_in=64, d_out=10):
    layers, dims = [], [d_in] + [width] * depth
    for a, b in zip(dims[:-1], dims[1:]):
        lin = nn.Linear(a, b)
        std = {"N(0, 1)": 1.0, "N(0, 0.01²)": 0.01, "Xavier": math.sqrt(2 / (a + b)), "He": math.sqrt(2 / a)}[init]
        nn.init.normal_(lin.weight, 0, std); nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh() if act == "tanh" else nn.ReLU()]
    layers.append(nn.Linear(width, d_out))
    return nn.Sequential(*layers)
#%% md
## Forward signal and backward gradient, layer by layer
A 50-layer, 256-wide network at initialisation, no training. Left: the standard deviation of each layer's activations on a batch of random inputs. Right: the norm of the gradient that reaches each layer's weights.
#%%
seed_everything(0)
x = torch.randn(512, 64)
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for row, act in enumerate(["tanh", "relu"]):
    for init in ["N(0, 1)", "N(0, 0.01²)", "Xavier", "He"]:
        seed_everything(0); net = make_mlp(50, 256, act, init)
        acts, h = [], x
        for m in net:
            h = m(h)
            if isinstance(m, (nn.Tanh, nn.ReLU)): h.retain_grad(); acts.append(h)
        net(x).pow(2).mean().backward()
        stds = [a.std().item() for a in acts]
        grads = [m.weight.grad.norm().item() for m in net if isinstance(m, nn.Linear)][:-1]
        axes[row, 0].semilogy(np.array(stds) + 1e-30, label=init); axes[row, 1].semilogy(np.array(grads) + 1e-30, label=init)
        print(f"{act:4s} {init:12s} activation std: layer 1 {stds[0]:.2e} → layer 50 {stds[-1]:.2e} | weight-grad norm: layer 1 {grads[0]:.2e}, layer 50 {grads[-1]:.2e}")
    axes[row, 0].set(title=f"{act}: activation std per layer", xlabel="layer"); axes[row, 1].set(title=f"{act}: weight-gradient norm per layer", xlabel="layer")
    axes[row, 0].legend(fontsize=8)
plt.tight_layout(); plt.show()
#%% md
Read the plots:
- **N(0, 0.01²):** every layer multiplies the signal by $\sqrt{256}\times 0.01 = 0.16$ (a ~6× shrink), so by layer 50 activations and gradients underflow to exactly zero.
- **N(0, 1):** each layer multiplies the signal by $\sqrt{256} = 16$ before the nonlinearity. tanh units saturate at ±1, and the backward pass, which multiplies by $W^\top$ at every layer, explodes to infinity. ReLU has no saturation, so its forward signal itself overflows (`nan`).
- **Xavier** keeps tanh in a usable range, although the signal still decays slowly with depth because tanh's slope is below 1 (PyTorch's `calculate_gain('tanh') = 5/3` compensates for that). For ReLU, Xavier loses a factor of $\sqrt2$ per layer, because ReLU discards half the variance: after 50 layers the signal is ~$10^{-8}$. **He** fixes exactly that factor and keeps ReLU steady.
## Does it matter for training?
A 20-layer plain MLP (no normalization, no residuals) on the 8×8 digits (inputs standardised), SGD with momentum as in Glorot & Bengio, 30 epochs, the same learning rate for every run.
#%%
{{DIGITS}}
Xtr, ytr, Xte, yte = [t.flatten(1).to(device) if t.dim() > 2 else t.to(device) for t in load_digits_tensors()]
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
def train(act, init, depth=20, epochs=30):
    seed_everything(0); net = make_mlp(depth, 128, act, init).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9); losses = []
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=device); tot = 0
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]; loss = F.cross_entropy(net(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        losses.append(tot / len(Xtr))
    with torch.no_grad(): acc = (net(Xte).argmax(1) == yte).float().mean().item()
    return losses, acc
runs = {}
for act, init in [("tanh", "N(0, 0.01²)"), ("tanh", "N(0, 1)"), ("tanh", "Xavier"), ("relu", "N(0, 0.01²)"), ("relu", "Xavier"), ("relu", "He")]:
    runs[(act, init)] = train(act, init)
    print(f"{act:4s} + {init:12s} final train loss {runs[(act, init)][0][-1]:.3f}   test acc {runs[(act, init)][1]:.3f}")
for (act, init), (l, _) in runs.items(): plt.plot(l, label=f"{act} + {init}")
plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("training loss"); plt.title("20-layer plain MLP, same optimiser: initialisation decides if it trains"); plt.legend(); plt.show()
#%% md
## Takeaways
- **Variance bookkeeping:** keep $n\sigma^2$ (times the activation's gain) equal to 1 per layer, forward and backward. Xavier does this for tanh, He for ReLU (PyTorch's default `nn.Linear` init is a uniform LeCun-style $1/n_{in}$ variant).
- Good initialisation made deep supervised networks trainable from scratch and, together with ReLU and GPUs, helped end the pretraining era (AlexNet, notebook 17).
- Later tools reduced the sensitivity further: batch/layer normalization (notebook 28) re-scale activations during training, and residual connections (notebook 29) make the identity the default. Modern LLMs still choose init scales carefully (e.g. scaling output projections by $1/\sqrt{2L}$ in GPT-2, or μP's width-dependent rules) so the same hyperparameters work across sizes.
