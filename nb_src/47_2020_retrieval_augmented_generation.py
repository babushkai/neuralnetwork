#%% md
# 2020 — Retrieval-Augmented Generation (RAG)

**Papers:** Patrick Lewis et al., *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*, NeurIPS (2020). Same year: REALM (Guu et al.) and Dense Passage Retrieval (DPR, Karpukhin et al.). Later: RETRO (Borgeaud et al., 2022).

A language model stores facts in its weights. That has three problems: facts learned after training are missing, updating one fact means retraining, and you cannot tell where an answer came from.

**RAG** splits the job in two:
1. A **retriever** embeds the question and finds the most similar passages in an external document store. In DPR this is a **dual encoder**: $\text{score}(q, d) = E_q(q)^\top E_d(d)$, trained contrastively so that a question scores highest with the passage that answers it.
2. A **reader / generator** answers the question *given the retrieved passages* in its input.

Knowledge now lives in the store: add or edit a document and the system's answers change, with no retraining. Today almost every deployed LLM application (search assistants, enterprise chat over documents, coding agents reading a repository) uses some form of retrieval.

**Toy world:** 600 people with made-up names, each with a city, a favourite colour and a job, written as one-sentence documents ("Talimo lives in Lisbon."). Models are trained on 450 of them. The other 150 are **new knowledge**: their documents are added to the store only after training.
#%% md
![Diagram: Retrieval-augmented generation (Lewis et al., 2020; DPR: Karpukhin et al., 2020)](https://raw.githubusercontent.com/babushkai/neuralnetwork/main/assets/diagrams/47_2020_retrieval_augmented_generation.png)
#%%
{{SETUP}}
#%%
rng = random.Random(0)
syll = ["ka", "lo", "mi", "ta", "re", "su", "no", "vi", "da", "pe", "zu", "ro", "na", "ti", "ma", "be"]
names = set()
while len(names) < 600: names.add("".join(rng.choice(syll) for _ in range(rng.randint(2, 3))).capitalize())
names = sorted(names); rng.shuffle(names)
CITIES = ["Lisbon", "Oslo", "Lima", "Cairo", "Seoul", "Quito", "Perth", "Delhi", "Paris", "Accra", "Dublin", "Hanoi"]
COLORS = ["red", "blue", "green", "amber", "violet", "teal", "black", "white"]
JOBS = ["baker", "pilot", "nurse", "farmer", "judge", "potter", "sailor", "chemist", "tailor", "miner"]
ANSWERS = CITIES + COLORS + JOBS; A2I = {a: i for i, a in enumerate(ANSWERS)}
TEMPLATES = {"city": ("{n} lives in {v}.", ["Where does {n} live?", "In which city does {n} live?"]),
             "color": ("{n} likes the color {v}.", ["What color does {n} like?", "Which colour is {n}'s favourite?"]),
             "job": ("{n} works as a {v}.", ["What is {n}'s job?", "What does {n} do for a living?"])}
people = {n: {"city": rng.choice(CITIES), "color": rng.choice(COLORS), "job": rng.choice(JOBS)} for n in names}
def docs_for(ns): return [(n, a, TEMPLATES[a][0].format(n=n, v=people[n][a])) for n in ns for a in TEMPLATES]
def questions_for(ns): return [(n, a, q.format(n=n)) for n in ns for a in TEMPLATES for q in TEMPLATES[a][1]]
train_people, new_people = names[:450], names[450:]
print(docs_for(names[:1])); print(questions_for(names[:1])[:3])
#%% md
## Character-level encoders
Names are new strings at test time, so every model reads characters. One small Transformer encoder class is used for the retriever's two towers, the reader and the closed-book baseline.
#%%
VOCAB = sorted(set("".join(d for _, _, d in docs_for(names)) + "".join(q for _, _, q in questions_for(names)) + "|"))
C2I = {c: i + 1 for i, c in enumerate(VOCAB)}                                # 0 = padding
def enc(texts, L=80):
    out = torch.zeros(len(texts), L, dtype=torch.long)
    for i, t in enumerate(texts): ids = [C2I[c] for c in t[:L]]; out[i, :len(ids)] = torch.tensor(ids)
    return out.to(device)

class CharEncoder(nn.Module):
    def __init__(self, d_out, d=128, n_layers=2, L=80):
        super().__init__(); self.emb = nn.Embedding(len(VOCAB) + 1, d, padding_idx=0); self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.0, batch_first=True, norm_first=True), n_layers)
        self.out = nn.Linear(d, d_out)
    def forward(self, ids):
        pad = ids == 0; h = self.tf(self.emb(ids) + self.pos[:ids.shape[1]], src_key_padding_mask=pad)
        h = h.masked_fill(pad[..., None], 0).sum(1) / (~pad).sum(1, keepdim=True)   # mean over real characters
        return self.out(h)

def fit_classifier(model, texts, labels, steps=3000, lr=1e-3, B=128):
    seed_everything(0); model = model.to(device); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    X, Y = enc(texts), torch.tensor(labels, device=device)
    for s in range(steps):
        i = torch.randint(0, len(X), (B,), device=device); loss = F.cross_entropy(model(X[i]), Y[i]); opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()
def accuracy(model, texts, labels):
    with torch.no_grad(): return (model(enc(texts)).argmax(1).cpu() == torch.tensor(labels)).float().mean().item()
#%% md
## Baseline: closed-book model
The question goes in, the answer comes out; the facts must be stored in the weights. It is trained on every question (both phrasings) about the 450 known people.
#%%
qs_train, qs_new = questions_for(train_people), questions_for(new_people)
label = lambda qs: [A2I[people[n][a]] for n, a, _ in qs]
closed = fit_classifier(CharEncoder(len(ANSWERS)), [q for *_, q in qs_train], label(qs_train), steps=4000)
print(f"closed-book: questions about known people {accuracy(closed, [q for *_, q in qs_train], label(qs_train)):.3f} | "
      f"about new people {accuracy(closed, [q for *_, q in qs_new], label(qs_new)):.3f} (chance ≈ {1 / 10:.2f})")
#%% md
## The retriever: a dense dual encoder (DPR)
Two encoders map questions and documents into the same 128-d space. Training uses **in-batch negatives**: in a batch of (question, correct document) pairs, each question must score its own document above all the others (a softmax over the batch, the same InfoNCE loss as CLIP in notebook 51). Each batch holds all three documents of each sampled person, so the encoders must separate attributes of the same person as well as different people (**hard negatives**, as in DPR).
#%%
E_q, E_d = CharEncoder(128).to(device), CharEncoder(128).to(device)
doc_of = {(n, a): d for n, a, d in docs_for(names)}
ATTRS = list(TEMPLATES)
seed_everything(0); opt = torch.optim.AdamW(list(E_q.parameters()) + list(E_d.parameters()), lr=1e-3, weight_decay=0.01)
t0 = time.time()
for s in range(4000):
    ps = random.sample(train_people, 40)
    qs = [(n, a, random.choice(TEMPLATES[a][1]).format(n=n)) for n in ps for a in ATTRS]
    q = F.normalize(E_q(enc([x for *_, x in qs])), dim=-1); d = F.normalize(E_d(enc([doc_of[(n, a)] for n, a, _ in qs])), dim=-1)
    loss = F.cross_entropy(q @ d.T / 0.05, torch.arange(len(qs), device=device)); opt.zero_grad(); loss.backward(); opt.step()
E_q.eval(); E_d.eval(); print(f"retriever trained in {time.time() - t0:.0f}s")

def build_index(docs):
    with torch.no_grad(): return F.normalize(E_d(enc([d for *_, d in docs])), dim=-1)
def retrieve(questions, docs, index, k=1):
    with torch.no_grad(): s = F.normalize(E_q(enc(questions)), dim=-1) @ index.T
    return s.topk(k, dim=1).indices.cpu()

store = docs_for(names)                                                      # the store now includes the 150 new people
index = build_index(store)
for name, qs in [("known people", qs_train), ("new people (never seen in training)", qs_new)]:
    top = retrieve([q for *_, q in qs], store, index, k=5)
    hit = torch.tensor([[store[j][:2] == (n, a) for j in row] for row, (n, a, _) in zip(top.tolist(), qs)])
    print(f"retrieval over {len(store)} docs, {name:36s} recall@1 {hit[:, 0].float().mean():.3f}  recall@5 {hit.any(1).float().mean():.3f}")
#%% md
## The reader, and the full RAG pipeline
The reader gets `retrieved document | question` and predicts the answer. One trap: if it is always trained on true documents about known people, it can score perfectly by **memorising names** and ignoring the document, and then fail on new people. So it is trained on documents with **random values** ("Talimo lives in Quito." with answer Quito, regardless of where Talimo really lives): the only way to be right is to read the document.
#%%
seed_everything(0); reader = CharEncoder(len(ANSWERS)).to(device)
opt = torch.optim.AdamW(reader.parameters(), lr=1e-3, weight_decay=0.01); VALUES = {"city": CITIES, "color": COLORS, "job": JOBS}
for s in range(4000):
    batch, ys = [], []
    for _ in range(128):
        n, a = random.choice(train_people), random.choice(ATTRS); v = random.choice(VALUES[a])
        batch.append(TEMPLATES[a][0].format(n=n, v=v) + "|" + random.choice(TEMPLATES[a][1]).format(n=n)); ys.append(A2I[v])
    loss = F.cross_entropy(reader(enc(batch)), torch.tensor(ys, device=device)); opt.zero_grad(); loss.backward(); opt.step()
reader.eval()
def rag_answer(qs):
    top = retrieve([q for *_, q in qs], store, index, k=1)[:, 0]
    return [store[j][2] + "|" + q for j, (_, _, q) in zip(top.tolist(), qs)]
for name, qs in [("known people", qs_train), ("new people", qs_new)]:
    oracle = accuracy(reader, [doc_of[(n, a)] + "|" + q for n, a, q in qs], label(qs))
    print(f"{name:13s} closed-book {accuracy(closed, [q for *_, q in qs], label(qs)):.3f} | RAG {accuracy(reader, rag_answer(qs), label(qs)):.3f} | reader with perfect retrieval {oracle:.3f}")
#%% md
## Updating knowledge without retraining
Talk about a person who moves. Edit one document in the store and re-embed it; ask again.
#%%
n = new_people[0]; q = [(n, "city", f"Where does {n} live?")]
with torch.no_grad(): before = ANSWERS[reader(enc(rag_answer(q))).argmax().item()]
old_city = people[n]["city"]; people[n]["city"] = "Oslo" if old_city != "Oslo" else "Lima"
store = docs_for(names); index = build_index(store)
with torch.no_grad(): after = ANSWERS[reader(enc(rag_answer(q))).argmax().item()]
print(f"{q[0][2]}  store said {old_city} → RAG answered {before}.  Store edited to {people[n]['city']} → RAG answers {after}.  Closed-book answers {ANSWERS[closed(enc([q[0][2]])).argmax().item()]}.")
#%% md
## Takeaways
- **Parametric vs. non-parametric memory.** The closed-book model can only answer about people it was trained on; RAG answers about people it never saw, because the facts are looked up, not remembered.
- **Train the reader to read.** With only true documents, a reader can memorise names instead; training on counterfactual documents removes that shortcut.
- **The retriever is often the bottleneck.** End-to-end accuracy is capped by recall@1; production systems retrieve several passages (top-k), re-rank them, and let the generator read all of them.
- **Updates are edits, not training runs**, and every answer can cite the document it came from.
- Simplifications: the "generator" here is a classifier over a fixed answer set; RAG (2020) used BART to generate free text and marginalised over the top retrieved passages, and modern systems put retrieved text directly into an LLM's long context (or let an agent call a search tool).
