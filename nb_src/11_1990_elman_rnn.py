#%% md
# 1990 — Recurrent Networks (Elman / "Simple RNN")

**Paper:** Jeffrey Elman, *"Finding Structure in Time"* (1990). (Jordan networks, 1986, are a close relative.)

Feed-forward nets see a fixed-size input. An RNN keeps a **hidden state** that is fed back at every step, giving it memory:

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h), \qquad y_t = W_{hy} h_t + b_y$$

Training = **backpropagation through time** (BPTT): unroll the loop and backprop through the unrolled graph.

Elman trained his network to **predict the next letter** in a stream and found that prediction error peaks at word boundaries — the network discovered "words" without being told. We reproduce that here with a character-level RNN written by hand.
#%%
{{SETUP}}
#%%
{{CORPUS}}
text = CORPUS.lower()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], device=device)
V = len(chars)
print("vocab size:", V)
#%%
class ElmanRNN(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Embedding(vocab, hidden)          # = W_xh applied to a one-hot input
        self.W_hh = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x, h=None):
        B, T = x.shape
        h = torch.zeros(B, self.hidden, device=x.device) if h is None else h
        e = self.embed(x)
        outs = []
        for t in range(T):                                 # the recurrence, written out explicitly
            h = torch.tanh(e[:, t] + self.W_hh(h))
            outs.append(h)
        H = torch.stack(outs, 1)
        return self.out(H), h

model = ElmanRNN(V, 128).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
T, B = 48, 32
losses = []
for step in range(1500):
    ix = torch.randint(0, len(data) - T - 1, (B,), device=device)
    x = data[ix[:, None] + torch.arange(T, device=data.device)]; y = data[ix[:, None] + torch.arange(1, T + 1, device=data.device)]
    logits, _ = model(x)
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # tames exploding gradients in BPTT
    opt.step(); losses.append(loss.item())
    if step % 300 == 0: print(f"step {step:4d} loss {loss.item():.3f}")
plt.plot(losses); plt.xlabel("step"); plt.ylabel("cross-entropy"); plt.show()
#%% md
## Elman's observation: surprise peaks at word boundaries
Per-character prediction error (−log p of the true next character) along a sentence. The first letter of each word is hard to predict; the rest of the word is easy.
#%%
sentence = "the perceptron changed its weights a little every time"
x = torch.tensor([[stoi[c] for c in sentence]], device=device)
model.eval()
with torch.no_grad():
    logits, _ = model(x[:, :-1])
    nll = -F.log_softmax(logits, -1)[0].gather(1, x[0, 1:, None]).squeeze().cpu()
plt.figure(figsize=(13, 3))
plt.bar(range(len(nll)), nll)
plt.xticks(range(len(nll)), list(sentence[1:]))
plt.ylabel("surprise (nats)"); plt.title("Error is high at the start of each word"); plt.show()
#%% md
## Sampling
#%%
def sample(model, prompt, n=300, temperature=0.7):
    x = torch.tensor([[stoi[c] for c in prompt]], device=device)
    with torch.no_grad():
        logits, h = model(x)
        out = prompt
        for _ in range(n):
            p = F.softmax(logits[:, -1] / temperature, -1)
            nxt = torch.multinomial(p, 1)
            out += chars[nxt.item()]
            logits, h = model(nxt, h)
    return out
print(sample(model, "the network "))
#%% md
## Takeaways
- Memory through a feedback loop; trained by BPTT.
- Plain RNNs struggle with **long-range** dependencies because gradients vanish or explode as they pass through many $W_{hh}$ multiplications — the motivation for the LSTM (next notebook).
