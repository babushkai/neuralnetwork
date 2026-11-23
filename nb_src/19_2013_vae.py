#%% md
# 2013 — Variational Autoencoders (VAE)

**Papers:** Diederik Kingma & Max Welling, *"Auto-Encoding Variational Bayes"* (2013); Rezende, Mohamed & Wierstra (2014).

A plain autoencoder's code space has holes, so decoding a random code gives garbage. A VAE makes it a proper **generative model** $p_\theta(x) = \int p_\theta(x|z)\,p(z)\,dz$ with prior $p(z) = \mathcal N(0, I)$:

- the encoder outputs a **distribution** $q_\phi(z|x) = \mathcal N(\mu_\phi(x), \sigma^2_\phi(x))$
- training maximises the **ELBO** $\;\mathbb E_{q}[\log p_\theta(x|z)] - \mathrm{KL}(q_\phi(z|x)\,\|\,p(z)) \le \log p_\theta(x)$
- the **reparameterisation trick** $z = \mu + \sigma \odot \epsilon,\ \epsilon\sim\mathcal N(0,I)$ lets gradients flow through sampling.

Then you can **generate** new data by decoding $z \sim \mathcal N(0, I)$.
#%% md
![Diagram: Variational autoencoder (Kingma & Welling, 2013)](https://raw.githubusercontent.com/babushkai/neuralnetwork/main/assets/diagrams/19_2013_vae.png)
#%%
{{SETUP}}
#%%
{{DIGITS}}
Xtr, ytr, Xte, yte = load_digits_tensors()
Xtr, Xte = Xtr.flatten(1).to(device), Xte.flatten(1).to(device)
#%%
class VAE(nn.Module):
    def __init__(self, d_in=64, h=256, z=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.mu, self.logvar = nn.Linear(h, z), nn.Linear(h, z)
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d_in))

    def forward(self, x):
        e = self.enc(x)
        mu, logvar = self.mu(e), self.logvar(e)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)        # reparameterisation trick
        return self.dec(z), mu, logvar

def elbo_loss(x, logits, mu, logvar):
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)   # -E_q[log p(x|z)]
    kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / len(x)          # closed-form Gaussian KL
    return rec + kl, rec, kl

vae = VAE().to(device)
opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
hist = []
for step in range(6000):
    x = Xtr[torch.randint(0, len(Xtr), (128,), device=device)]
    loss, rec, kl = elbo_loss(x, *vae(x))
    opt.zero_grad(); loss.backward(); opt.step()
    hist.append((rec.item(), kl.item()))
    if step % 1500 == 0: print(f"step {step:4d}  -ELBO {loss.item():.2f}  (recon {rec.item():.2f}, KL {kl.item():.2f})")
h = np.array(hist)
plt.plot(h[:, 0], label="reconstruction"); plt.plot(h[:, 1], label="KL"); plt.legend(); plt.xlabel("step"); plt.show()
#%% md
## The latent space is smooth and close to $\mathcal N(0, I)$
#%%
vae.eval()
with torch.no_grad():
    mu = vae.mu(vae.enc(Xte)).cpu()
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
sc = axes[0].scatter(mu[:, 0], mu[:, 1], c=yte, cmap="tab10", s=10); fig.colorbar(sc, ax=axes[0], ticks=range(10))
axes[0].set_title("encoder means of test digits")
# decode a grid of latent points
grid = torch.linspace(-2.5, 2.5, 15)
zz = torch.stack(torch.meshgrid(grid, grid.flip(0), indexing="xy"), -1).reshape(-1, 2).to(device)
with torch.no_grad():
    imgs = torch.sigmoid(vae.dec(zz)).cpu().view(15, 15, 8, 8)
canvas = imgs.permute(0, 2, 1, 3).reshape(15 * 8, 15 * 8)
axes[1].imshow(canvas, cmap="gray_r", extent=[-2.5, 2.5, -2.5, 2.5]); axes[1].set_title("decoder output over the latent grid")
plt.show()
#%% md
## Sampling new digits from the prior
#%%
with torch.no_grad():
    samples = torch.sigmoid(vae.dec(torch.randn(32, 2, device=device))).cpu()
fig, axes = plt.subplots(2, 16, figsize=(12, 1.8))
for a, s in zip(axes.flat, samples): a.imshow(s.view(8, 8), cmap="gray_r"); a.axis("off")
plt.suptitle("z ~ N(0, I)  ->  decoder"); plt.show()
