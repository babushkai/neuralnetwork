#%% md
# 2013 — word2vec: Learning Word Embeddings

**Papers:** Tomas Mikolov et al., *"Efficient Estimation of Word Representations in Vector Space"* and *"Distributed Representations of Words and Phrases and their Compositionality"* (2013). Builds on Bengio et al.'s neural language model (2003).

**Distributional hypothesis:** "you shall know a word by the company it keeps". **Skip-gram with negative sampling (SGNS)** learns a vector $u_w$ for each word and $v_c$ for each context word, so that true (word, context) pairs score high and random pairs score low:

$$\mathcal L = -\log\sigma(u_w^\top v_c) - \sum_{k=1}^{K} \log\sigma(-u_w^\top v_{n_k}), \qquad n_k \sim P_{\text{noise}}(w) \propto \text{count}(w)^{3/4}$$

To keep this offline and fast we build a **synthetic corpus** from a small grammar: words of the same kind (animals, foods, royals, …) appear in similar contexts, which is the statistical structure word2vec exploits.
#%%
{{SETUP}}
from collections import Counter
#%%
groups = {
    "animal": ["cat", "dog", "horse", "rabbit", "tiger", "wolf"],
    "food": ["bread", "apple", "rice", "cheese", "soup", "cake"],
    "royal": ["king", "queen", "prince", "princess", "emperor", "duke"],
    "place": ["castle", "palace", "village", "forest", "city", "farm"],
    "tool": ["hammer", "knife", "shovel", "saw", "drill", "axe"],
}
templates = [
    ("the {animal} eats the {food}", 3), ("a hungry {animal} found some {food}", 2),
    ("the {animal} sleeps in the {place}", 2), ("the {royal} rules the {place}", 3),
    ("the {royal} lives in the {place}", 2), ("the {royal} eats {food} at the {place}", 1),
    ("the worker uses a {tool} at the {place}", 2), ("she cut the {food} with a {tool}", 1),
    ("the {royal} gave a {tool} to the worker", 1), ("a {animal} ran through the {place}", 2),
]
rng = np.random.default_rng(0)
weights = np.array([w for _, w in templates], float); weights /= weights.sum()
sentences = []
for _ in range(20000):
    t = templates[rng.choice(len(templates), p=weights)][0]
    sentences.append(t.format(**{g: rng.choice(ws) for g, ws in groups.items()}).split())
counts = Counter(w for s in sentences for w in s)
vocab = sorted(counts); stoi = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
print(f"{len(sentences)} sentences, vocab {V}; e.g.:", " | ".join(" ".join(s) for s in sentences[:3]))
#%% md
## Build (center, context) pairs with window size 2
#%%
pairs = [(stoi[s[i]], stoi[s[j]]) for s in sentences for i in range(len(s))
         for j in range(max(0, i - 2), min(len(s), i + 3)) if i != j]
pairs = torch.tensor(pairs, device=device)
noise = torch.tensor([counts[w] ** 0.75 for w in vocab], device=device); noise /= noise.sum()
print("training pairs:", len(pairs))
#%%
class SGNS(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.inp = nn.Embedding(V, d); self.out = nn.Embedding(V, d)
        nn.init.uniform_(self.inp.weight, -0.5 / d, 0.5 / d); nn.init.zeros_(self.out.weight)
    def forward(self, w, c, neg):
        u = self.inp(w)                                        # (B, d)
        pos = F.logsigmoid((u * self.out(c)).sum(-1))
        negs = F.logsigmoid(-(self.out(neg) @ u.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return -(pos + negs).mean()

model = SGNS(V, 32).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
K, B = 5, 1024
for step in range(3000):
    batch = pairs[torch.randint(0, len(pairs), (B,), device=device)]
    neg = torch.multinomial(noise, B * K, replacement=True).view(B, K)
    loss = model(batch[:, 0], batch[:, 1], neg)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 600 == 0: print(f"step {step} loss {loss.item():.3f}")
#%% md
## Nearest neighbours by cosine similarity
In this toy grammar, words in the same group are perfectly interchangeable, so similarities close to 1.0 are expected. Real corpora are messier, and neighbours there also capture finer relations.
#%%
E = F.normalize(model.inp.weight.detach(), dim=1).cpu()
def neighbours(w, k=5):
    sims = E @ E[stoi[w]]
    return [(vocab[i], round(sims[i].item(), 2)) for i in sims.argsort(descending=True)[1:k + 1]]
for w in ["cat", "cheese", "queen", "forest", "hammer"]:
    print(f"{w:8s} -> {neighbours(w)}")
#%% md
## The embedding space (PCA to 2-D)
Words cluster by their role, and the model was never told the categories.
#%%
from sklearn.decomposition import PCA
words = [w for ws in groups.values() for w in ws]
Z = PCA(2).fit_transform(E[[stoi[w] for w in words]].numpy())
colors = {g: c for g, c in zip(groups, plt.cm.tab10.colors)}
plt.figure(figsize=(8, 6))
for w, z in zip(words, Z):
    g = next(g for g, ws in groups.items() if w in ws)
    plt.scatter(*z, color=colors[g]); plt.annotate(w, z, fontsize=9)
for g, c in colors.items(): plt.scatter([], [], color=c, label=g)
plt.legend(); plt.title("word2vec embeddings"); plt.show()
#%% md
## Takeaways
- Words become dense vectors whose geometry encodes meaning. On large corpora this gives the famous analogies (king − man + woman ≈ queen).
- Every later language model starts with an embedding table like this one, now trained end-to-end inside the model.
