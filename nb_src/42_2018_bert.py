#%% md
# 2018 — BERT: Bidirectional Pre-Training with Masked Language Modelling

**Paper:** Jacob Devlin, Ming-Wei Chang, Kenton Lee & Kristina Toutanova, *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"* (2018).

GPT reads left-to-right. BERT uses the Transformer **encoder** so that every token sees **both sides**, and pre-trains it with a cloze task:

- **Masked LM:** choose 15% of tokens; replace 80% of those with `[MASK]`, 10% with a random token, and keep 10% unchanged; predict the originals
- (Next-sentence prediction was also used, but later work such as RoBERTa showed it was unnecessary, so we skip it.)

Then **fine-tune**: put a small classifier on the `[CLS]` token. *Pre-train once on unlabelled text, fine-tune cheaply on many tasks* became the NLP recipe of 2018–2020.

Our setup: a synthetic "review" language. We pre-train on 20k unlabelled sentences, then fine-tune on just **64 labelled** sentences, and compare against the same model trained from scratch.
#%%
{{SETUP}}
#%%
rng = random.Random(0)
lex = {
    "pos": ["great", "wonderful", "excellent", "brilliant", "lovely", "superb"],
    "neg": ["terrible", "awful", "boring", "dreadful", "poor", "weak"],
    "noun": ["movie", "book", "song", "show", "game", "meal"],
    "int": ["very", "really", "truly", "quite"],
    "who": ["my friend", "the critic", "everyone", "my sister", "the audience"],
}
def sentence():
    s = rng.choice(["pos", "neg"]); n = rng.choice(lex["noun"])
    t = rng.choice([
        f"the {n} was {rng.choice(lex['int'])} {rng.choice(lex[s])}",
        f"{rng.choice(lex['who'])} said the {n} was {rng.choice(lex[s])}",
        f"what a {rng.choice(lex[s])} {n}",
        f"i think this {n} is {rng.choice(lex[s])} and {rng.choice(lex[s])}",
        f"the {n} was not {rng.choice(lex['neg' if s == 'pos' else 'pos'])}",   # negation flips the word's polarity
    ])
    return t.split(), int(s == "pos")

PAD, CLS, MASK = "[PAD]", "[CLS]", "[MASK]"
words = sorted({w for _ in range(2000) for w in sentence()[0]} | {w for v in lex.values() for p in v for w in p.split()})
vocab = [PAD, CLS, MASK] + words; stoi = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
T = 12
def encode(batch):
    ids = [[stoi[CLS]] + [stoi[w] for w in s] for s, _ in batch]
    return torch.tensor([x + [0] * (T - len(x)) for x in ids], device=device), torch.tensor([y for _, y in batch], device=device)

unlabelled = [sentence() for _ in range(20000)]
labelled, test = [sentence() for _ in range(64)], [sentence() for _ in range(2000)]
print("vocab:", V, "| example:", " ".join(unlabelled[0][0]))
#%%
class BERT(nn.Module):
    def __init__(self, V, d=128, h=4, n=3):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, h, 4 * d, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.mlm_head = nn.Linear(d, V)
        self.cls_head = nn.Linear(d, 2)
    def forward(self, ids):
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        return self.norm(self.encoder(x, src_key_padding_mask=ids == 0))   # bidirectional: no causal mask

def mask_tokens(ids, p=0.15):
    labels = ids.clone()
    cand = ids > stoi[MASK]                                   # never mask PAD / CLS
    chosen = (torch.rand(ids.shape, device=ids.device) < p) & cand
    labels[~chosen] = -100                                    # loss only on chosen positions
    r = torch.rand(ids.shape, device=ids.device)
    ids = ids.clone()
    ids[chosen & (r < 0.8)] = stoi[MASK]                                             # 80%: [MASK]
    rand_pos = chosen & (r >= 0.8) & (r < 0.9)                                       # 10%: random token
    ids[rand_pos] = torch.randint(3, V, (int(rand_pos.sum()),), device=ids.device)
    return ids, labels                                                                # 10%: unchanged
#%% md
## Pre-training (masked LM)
#%%
bert = BERT(V).to(device)
opt = torch.optim.AdamW(bert.parameters(), lr=5e-4, weight_decay=0.01)
for step in range(2001):
    bert.train()
    ids, _ = encode(rng.sample(unlabelled, 128))
    inp, lab = mask_tokens(ids)
    loss = F.cross_entropy(bert.mlm_head(bert(inp)).reshape(-1, V), lab.reshape(-1), ignore_index=-100)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0: print(f"step {step:4d}  MLM loss {loss.item():.3f}")
#%%
@torch.no_grad()
def fill(text, k=4):
    bert.eval()
    toks = text.split(); ids, _ = encode([(toks, 0)])
    pos = toks.index(MASK) + 1
    probs = bert.mlm_head(bert(ids))[0, pos].softmax(-1)
    top = probs.topk(k)
    print(f"{text:45s} -> " + ", ".join(f"{vocab[i]} ({p:.2f})" for p, i in zip(top.values.tolist(), top.indices.tolist())))
fill("the movie was very [MASK]")
fill("my friend said the [MASK] was superb")
fill("i think this book is awful and [MASK]")
fill("what a [MASK] song")
#%% md
## Fine-tuning on 64 labelled examples vs. training from scratch
#%%
def finetune(model, steps=300):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    ids, y = encode(labelled)
    for _ in range(steps):
        model.train()
        loss = F.cross_entropy(model.cls_head(model(ids)[:, 0]), y)            # classify from [CLS]
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        ids_t, y_t = encode(test)
        return (model.cls_head(model(ids_t)[:, 0]).argmax(1) == y_t).float().mean().item()

seed_everything(1)
acc_scratch = finetune(BERT(V).to(device))
acc_pre = finetune(bert)
print(f"test accuracy — from scratch: {acc_scratch:.3f} | pre-trained BERT + fine-tune: {acc_pre:.3f}")
