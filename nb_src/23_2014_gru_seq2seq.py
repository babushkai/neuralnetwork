#%% md
# 2014 — GRU and Sequence-to-Sequence Learning

**Papers:** Kyunghyun Cho et al., *"Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation"* (2014), which introduced the **GRU**; Ilya Sutskever, Oriol Vinyals & Quoc Le, *"Sequence to Sequence Learning with Neural Networks"* (2014).

**GRU** is a simpler LSTM with two gates and no separate cell state:

$$
z_t = \sigma(W_z[x_t,h_{t-1}]),\quad r_t = \sigma(W_r[x_t,h_{t-1}]),\quad \tilde h_t = \tanh(W_h[x_t,\ r_t\odot h_{t-1}]),\quad h_t = (1-z_t)\odot h_{t-1} + z_t\odot\tilde h_t
$$

**Seq2seq:** an *encoder* RNN reads the input sequence into a fixed vector, and a *decoder* RNN generates the output sequence from it. This was the first neural approach to work well for machine translation. It is trained with **teacher forcing** (the decoder is fed the true previous token) and decoded greedily.

Task: **sort** a sequence of digits, e.g. `5 1 4 1` → `1 1 4 5`. Input and output lengths can differ in general, and seq2seq handles that naturally.
#%%
{{SETUP}}
#%%
class GRUCell(nn.Module):
    def __init__(self, n_in, n_hid):
        super().__init__()
        self.x2 = nn.Linear(n_in, 3 * n_hid); self.h2 = nn.Linear(n_hid, 3 * n_hid)
    def forward(self, x, h):
        xr, xz, xn = self.x2(x).chunk(3, -1); hr, hz, hn = self.h2(h).chunk(3, -1)
        r, z = torch.sigmoid(xr + hr), torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        return (1 - z) * n + z * h          # PyTorch's convention: z gates the *old* state

cell, ref = GRUCell(4, 6), nn.GRUCell(4, 6)
with torch.no_grad():
    cell.x2.weight.copy_(ref.weight_ih); cell.x2.bias.copy_(ref.bias_ih)
    cell.h2.weight.copy_(ref.weight_hh); cell.h2.bias.copy_(ref.bias_hh)
x, h = torch.randn(3, 4), torch.randn(3, 6)
print("max difference vs nn.GRUCell:", (cell(x, h) - ref(x, h)).abs().max().item())
#%% md
## Data: sort digits
Tokens 0–9 are digits, 10 = `<sos>` (start of output), 11 = `<eos>`.
#%%
SOS, EOS, V, L = 10, 11, 12, 8
def batch(B, L=L):
    x = torch.randint(0, 10, (B, L))
    y = torch.cat([x.sort(1).values, torch.full((B, 1), EOS)], 1)
    return x.to(device), y.to(device)

class Seq2Seq(nn.Module):
    def __init__(self, V, d=128):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.encoder = nn.GRU(d, d, batch_first=True)
        self.decoder = nn.GRU(d, d, batch_first=True)
        self.out = nn.Linear(d, V)

    def forward(self, src, tgt):                       # teacher forcing
        _, h = self.encoder(self.emb(src))             # h: the whole input squeezed into one vector
        dec_in = torch.cat([torch.full_like(tgt[:, :1], SOS), tgt[:, :-1]], 1)
        o, _ = self.decoder(self.emb(dec_in), h)
        return self.out(o)

    @torch.no_grad()
    def generate(self, src, max_len):
        _, h = self.encoder(self.emb(src))
        tok = torch.full((len(src), 1), SOS, device=src.device); out = []
        for _ in range(max_len):
            o, h = self.decoder(self.emb(tok), h)
            tok = self.out(o).argmax(-1); out.append(tok)
        return torch.cat(out, 1)

model = Seq2Seq(V).to(device)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
for step in range(3001):
    x, y = batch(128)
    loss = F.cross_entropy(model(x, y).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 500 == 0:
        xt, yt = batch(500)
        pred = model.generate(xt, L + 1)
        print(f"step {step:4d} loss {loss.item():.3f} | token acc {(pred == yt).float().mean():.3f} | full-sequence acc {(pred == yt).all(1).float().mean():.3f}")
#%%
xt, yt = batch(5)
for a, b in zip(xt.tolist(), model.generate(xt, L + 1).tolist()):
    print(a, "->", b[:-1])
#%% md
## Two weaknesses
**1. Length generalisation.** The model only ever saw length-8 inputs, so it fails at *any* other length, even shorter ones.
#%%
for Ltest in [4, 8, 12]:
    xt, yt = batch(500, Ltest)
    pred = model.generate(xt, Ltest + 1)
    print(f"test length {Ltest:2d}: token accuracy {(pred == yt).float().mean():.3f}")
#%% md
**2. The fixed-size bottleneck.** The encoder has to squeeze the whole input into one vector. Train fresh models (same size, same budget) on longer sequences: the longer the input, the harder it gets. Cho et al. saw translation quality drop sharply with sentence length for exactly this reason. **Attention** (next notebook) removes the bottleneck.
#%%
for Ltrain in [8, 16, 24]:
    seed_everything(0)
    m = Seq2Seq(V).to(device); o = torch.optim.Adam(m.parameters(), lr=2e-3)
    for step in range(1500):
        x, y = batch(128, Ltrain)
        loss = F.cross_entropy(m(x, y).reshape(-1, V), y.reshape(-1))
        o.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); o.step()
    xt, yt = batch(500, Ltrain); pred = m.generate(xt, Ltrain + 1)
    print(f"trained & tested on length {Ltrain:2d}: token acc {(pred == yt).float().mean():.3f} | full-sequence acc {(pred == yt).all(1).float().mean():.3f}")
