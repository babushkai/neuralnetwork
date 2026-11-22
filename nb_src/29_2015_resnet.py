#%% md
# 2015 — Residual Networks (ResNet)

**Paper:** Kaiming He, Xiangyu Zhang, Shaoqing Ren & Jian Sun, *"Deep Residual Learning for Image Recognition"* (2015). Won ImageNet 2015 with a 152-layer network.

**The degradation problem:** a deeper "plain" network had *higher training error* than a shallower one. This is not overfitting; the deeper net is simply harder to optimise, even though it could in principle copy the shallow one and use identity layers for the rest.

**The fix:** let each block learn a *residual* and add a shortcut:
$$y = x + \mathcal F(x)$$
Identity is now the default (just set $\mathcal F = 0$), and gradients flow straight through the shortcuts: $\partial y/\partial x = I + \partial\mathcal F/\partial x$. Residual connections are in almost every modern architecture, including every Transformer.
#%%
{{SETUP}}
#%%
{{DIGITS}}
Xtr, ytr, Xte, yte = [t.to(device) for t in load_digits_tensors(size=16)]
#%%
class Block(nn.Module):
    def __init__(self, c, residual):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        self.conv2, self.bn2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        self.residual = residual
    def forward(self, x):
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(out + x if self.residual else out)          # <- the only difference

class Net(nn.Module):
    def __init__(self, n_blocks, residual, c=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(c, residual) for _ in range(n_blocks)])
        self.head = nn.Linear(c, 10)
    def forward(self, x):
        return self.head(self.blocks(self.stem(x)).mean((2, 3)))

def train(n_blocks, residual, epochs=12):
    seed_everything(0)
    m = Net(n_blocks, residual).to(device)
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.05, total_steps=epochs * ((len(Xtr) + 63) // 64))
    losses = []
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xtr), device=device); tot = 0
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            loss = F.cross_entropy(m(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(idx)
        losses.append(tot / len(Xtr))
    m.eval()
    with torch.no_grad(): acc = (m(Xte).argmax(1) == yte).float().mean().item()
    return losses, acc
#%% md
## Plain vs. residual at increasing depth
Depth = 2 × blocks + 2 layers. Watch the **training** loss: plain networks get *worse* as they get deeper, while residual networks do not.
#%%
results = {}
for n_blocks in [2, 8, 20]:
    for residual in [False, True]:
        results[(n_blocks, residual)] = train(n_blocks, residual)
        l, acc = results[(n_blocks, residual)]
        print(f"{2 * n_blocks + 2:3d} layers  {'ResNet' if residual else 'plain ':6s}  final train loss {l[-1]:.4f}  test acc {acc:.3f}")
#%%
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, residual in zip(axes, [False, True]):
    for n_blocks in [2, 8, 20]:
        ax.plot(results[(n_blocks, residual)][0], label=f"{2 * n_blocks + 2} layers")
    ax.set(yscale="log", title="ResNet" if residual else "Plain network", xlabel="epoch", ylabel="training loss"); ax.legend()
plt.tight_layout(); plt.show()
#%% md
## Gradient norm reaching the first layer
#%%
for residual in [False, True]:
    seed_everything(0); m = Net(20, residual).to(device)
    F.cross_entropy(m(Xtr[:128]), ytr[:128]).backward()
    g = [b.conv1.weight.grad.norm().item() for b in m.blocks]
    plt.semilogy(g, marker=".", label="ResNet" if residual else "plain")
plt.xlabel("block index (input → output)"); plt.ylabel("‖grad‖ of conv1"); plt.legend(); plt.title("Gradient flow at initialisation (42 layers)"); plt.show()
