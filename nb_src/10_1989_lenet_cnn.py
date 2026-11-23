#%% md
# 1989 — Convolutional Networks (LeNet)

**Papers:** Yann LeCun et al., *"Backpropagation Applied to Handwritten Zip Code Recognition"* (1989); *"Gradient-Based Learning Applied to Document Recognition"* (LeNet-5, 1998). Inspired by Fukushima's *Neocognitron* (1980).

Ideas that make images tractable:
- **Local receptive fields:** each unit sees a small patch.
- **Weight sharing:** the same filter slides over the whole image, so features are translation-equivariant and parameters are few.
- **Subsampling (pooling):** build tolerance to small shifts.

We build LeNet-5 (32×32 input → C1 → S2 → C3 → S4 → C5 → F6 → output) and train it on handwritten digits upscaled to 32×32. From here on, notebooks use PyTorch and **run on a GPU if one is available** (CUDA or Apple MPS), otherwise CPU.
#%% md
![Diagram: LeNet-5 (LeCun et al., 1989/1998)](https://raw.githubusercontent.com/babushkai/neuralnetwork/main/assets/diagrams/10_1989_lenet_cnn.png)
#%%
{{SETUP}}
#%%
{{DIGITS}}

Xtr, ytr, Xte, yte = load_digits_tensors(size=32)
Xtr, ytr, Xte, yte = Xtr.to(device), ytr.to(device), Xte.to(device), yte.to(device)
print(Xtr.shape, Xte.shape)
#%%
class LeNet5(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5)       # 32 -> 28
        self.s2 = nn.AvgPool2d(2)          # 28 -> 14
        self.c3 = nn.Conv2d(6, 16, 5)      # 14 -> 10
        self.s4 = nn.AvgPool2d(2)          # 10 -> 5
        self.c5 = nn.Conv2d(16, 120, 5)    # 5 -> 1
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.s2(torch.tanh(self.c1(x)))
        x = self.s4(torch.tanh(self.c3(x)))
        x = torch.tanh(self.c5(x)).flatten(1)
        return self.out(torch.tanh(self.f6(x)))

model = LeNet5().to(device)
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))
#%%
def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
hist = []
for epoch in range(15):
    model.train()
    perm = torch.randperm(len(Xtr), device=device)
    for i in range(0, len(Xtr), 64):
        idx = perm[i:i + 64]
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    hist.append((accuracy(model, Xtr, ytr), accuracy(model, Xte, yte)))
    if epoch % 3 == 2: print(f"epoch {epoch + 1:2d}  train {hist[-1][0]:.3f}  test {hist[-1][1]:.3f}")
#%%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot([h[0] for h in hist], label="train"); axes[0].plot([h[1] for h in hist], label="test")
axes[0].set(xlabel="epoch", ylabel="accuracy"); axes[0].legend()
w = model.c1.weight.detach().cpu()[:, 0]
grid = torch.cat([torch.cat([w[i], torch.full((5, 1), float("nan"))], 1) for i in range(6)], 1)
axes[1].imshow(grid, cmap="RdBu"); axes[1].set_title("C1: the six learned 5×5 filters"); axes[1].axis("off")
plt.tight_layout(); plt.show()
#%% md
## Feature maps and translation tolerance
The C1 maps show edge/stroke detectors. Convolutions are translation-*equivariant* and pooling adds some *local* invariance, but it is limited: small shifts cost a little accuracy, large ones a lot. That is why later CNNs (2012) added **data augmentation** with random shifts.
#%%
x = Xte[:1]
with torch.no_grad():
    maps = torch.tanh(model.c1(x))[0].cpu()
fig, axes = plt.subplots(1, 7, figsize=(12, 2))
axes[0].imshow(x[0, 0].cpu(), cmap="gray_r"); axes[0].set_title(f"input (label {yte[0].item()})")
for i in range(6): axes[i + 1].imshow(maps[i], cmap="viridis"); axes[i + 1].set_title(f"C1 map {i}")
for a in axes: a.axis("off")
plt.show()

model.eval()
with torch.no_grad():
    for shift in [0, 2, 4]:
        acc = (model(torch.roll(Xte, shifts=shift, dims=3)).argmax(1) == yte).float().mean().item()
        print(f"shift right by {shift}px -> test accuracy {acc:.3f}")
