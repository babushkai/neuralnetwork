#%% md
# 2014 — VGG and Inception: How to Build a Deeper CNN

**Papers:** Karen Simonyan & Andrew Zisserman, *"Very Deep Convolutional Networks for Large-Scale Image Recognition"* (VGG, 2014); Christian Szegedy et al., *"Going Deeper with Convolutions"* (GoogLeNet/Inception, 2014); Min Lin, Qiang Chen & Shuicheng Yan, *"Network In Network"* (2013), the source of 1×1 convolutions and global average pooling.

After AlexNet (notebook 17), the 2014 ImageNet challenge was won by going deeper, with two different design philosophies:

- **VGG: only 3×3 convolutions, stacked.** Two 3×3 layers see a 5×5 region and three see 7×7, but with fewer weights ($2\cdot 9C^2 = 18C^2$ vs $25C^2$; $27C^2$ vs $49C^2$) and an extra nonlinearity in between. VGG-16/19 were simple, uniform and became the standard feature extractor for years (style transfer, perceptual losses, detection backbones).
- **Inception: wide, multi-scale, and cheap.** Each module runs 1×1, 3×3 and 5×5 convolutions and a pooling branch **in parallel** and concatenates them. **1×1 convolutions** first shrink the channel count (a "bottleneck"), so the expensive branches are cheap. GoogLeNet had 22 layers but ~12× fewer parameters than AlexNet, partly thanks to **global average pooling** instead of big fully connected layers.

ResNet (notebook 29) then took depth further with residual connections, keeping VGG's 3×3 style and Inception's 1×1 bottlenecks.
#%%
{{SETUP}}
#%% md
## Receptive fields: two 3×3 layers really do see 5×5
Measure the receptive field empirically: take the gradient of the centre output pixel with respect to the input and see which input pixels it touches.
#%%
def receptive_field(net, size=21):
    x = torch.zeros(1, 1, size, size, requires_grad=True)
    y = net(x); y[0, 0, size // 2, size // 2].backward()
    nz = (x.grad[0, 0].abs() > 0).nonzero()
    return (nz[:, 0].max() - nz[:, 0].min() + 1).item()
seed_everything(0)
for name, net in [("one 5×5", nn.Conv2d(1, 1, 5, padding=2)),
                  ("two 3×3", nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.Tanh(), nn.Conv2d(4, 1, 3, padding=1))),
                  ("one 7×7", nn.Conv2d(1, 1, 7, padding=3)),
                  ("three 3×3", nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.Tanh(), nn.Conv2d(4, 4, 3, padding=1), nn.Tanh(), nn.Conv2d(4, 1, 3, padding=1)))]:
    print(f"{name:10s} receptive field {receptive_field(net)}×{receptive_field(net)}")
C = 256
print(f"\nweights for C={C} channels in and out: 5×5 = {25 * C * C:,} vs two 3×3 = {18 * C * C:,};  7×7 = {49 * C * C:,} vs three 3×3 = {27 * C * C:,}")
#%% md
## The Inception module and its 1×1 bottlenecks
#%%
class Inception(nn.Module):
    """GoogLeNet-style module: four parallel branches, concatenated along channels."""
    def __init__(self, c_in, c1, c3r, c3, c5r, c5, cp, bottleneck=True):
        super().__init__()
        bn = lambda ci, co: [nn.Conv2d(ci, co, 1), nn.ReLU()] if bottleneck else []
        r3, r5 = (c3r, c5r) if bottleneck else (c_in, c_in)
        self.b1 = nn.Sequential(nn.Conv2d(c_in, c1, 1), nn.ReLU())
        self.b3 = nn.Sequential(*bn(c_in, c3r), nn.Conv2d(r3, c3, 3, padding=1), nn.ReLU())
        self.b5 = nn.Sequential(*bn(c_in, c5r), nn.Conv2d(r5, c5, 5, padding=2), nn.ReLU())
        self.bp = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(c_in, cp, 1), nn.ReLU())
    def forward(self, x): return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], 1)

def count(m): return sum(p.numel() for p in m.parameters())
def flops(m, shape):
    """Multiply-accumulates of all convolutions, counted with forward hooks."""
    total = []
    def hook(mod, inp, out): total.append(out.numel() * mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1] // mod.groups)
    hs = [mm.register_forward_hook(hook) for mm in m.modules() if isinstance(mm, nn.Conv2d)]
    with torch.no_grad(): m(torch.zeros(shape))
    for h in hs: h.remove()
    return sum(total)
# GoogLeNet's "inception (3a)" configuration: 192 channels in, 256 out, on a 28×28 map
for b in [False, True]:
    m = Inception(192, 64, 96, 128, 16, 32, 32, bottleneck=b)
    print(f"inception(3a) {'with' if b else 'without'} 1×1 bottlenecks: {count(m):>9,} params, {flops(m, (1, 192, 28, 28)) / 1e6:6.0f}M MACs")
#%% md
## Three small networks on the same task
On 16×16 digits, with a similar parameter budget:
- **AlexNet-style:** large 5×5 kernels and a big fully connected head
- **VGG-style:** stacks of 3×3 convolutions, then a small head
- **Inception-style:** two Inception modules and global average pooling
#%%
{{DIGITS}}
Xtr, ytr, Xte, yte = [t.to(device) for t in load_digits_tensors(size=16)]
def alex_style():
    return nn.Sequential(nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 48, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(48 * 16, 96), nn.ReLU(), nn.Linear(96, 10))
def vgg_style():
    c = lambda i, o: [nn.Conv2d(i, o, 3, padding=1), nn.ReLU()]
    return nn.Sequential(*c(1, 32), *c(32, 32), nn.MaxPool2d(2), *c(32, 64), *c(64, 64), nn.MaxPool2d(2), *c(64, 96),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 10))
def inception_style():
    return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         Inception(32, 16, 24, 32, 8, 16, 16), Inception(80, 32, 32, 48, 8, 16, 16), nn.MaxPool2d(2),
                         Inception(112, 32, 48, 64, 8, 16, 16), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10))
def train(make, epochs=25):
    seed_everything(0); m = make().to(device); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((len(Xtr) + 63) // 64))
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]; loss = F.cross_entropy(m(Xtr[idx]), ytr[idx]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad(): return (m(Xte).argmax(1) == yte).float().mean().item(), m
for name, make in [("AlexNet-style (5×5, FC head)", alex_style), ("VGG-style (3×3 stacks, GAP)", vgg_style), ("Inception-style (modules, GAP)", inception_style)]:
    accs = [train(make)[0] for _ in range(1)]
    m = make()
    head = sum(p.numel() for mod in m.modules() if isinstance(mod, nn.Linear) for p in mod.parameters())
    print(f"{name:32s} params {count(m):>7,} (of which fully connected {head:>6,})  conv MACs {flops(m, (1, 1, 16, 16)) / 1e6:5.2f}M  test acc {accs[0]:.3f}")
#%% md
## Takeaways
- **Small kernels, more depth:** stacked 3×3 convolutions cover the same receptive field as large kernels with fewer weights and more nonlinearities. Almost every CNN since VGG uses 3×3 as the default.
- **1×1 convolutions** mix channels cheaply. As bottlenecks they cut Inception's cost by several times; ResNet's bottleneck blocks, MobileNet's pointwise convolutions and a Transformer's per-token MLP are all the same operation.
- **Global average pooling** removes most of the parameters that AlexNet spent in its fully connected layers.
- On these small digit images all three reach similar accuracy; the differences that mattered on ImageNet were in parameters and compute, which is what the tables above compare. Depth beyond ~20 layers still failed to train well until residual connections (notebook 29) and batch normalization (notebook 28; Inception-v2 introduced it).
