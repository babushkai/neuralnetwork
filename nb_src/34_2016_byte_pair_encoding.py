#%% md
# 2016 — Byte-Pair Encoding: Subword Tokenization

**Papers:** Rico Sennrich, Barry Haddow & Alexandra Birch, *"Neural Machine Translation of Rare Words with Subword Units"*, ACL (2016). The algorithm comes from data compression: Philip Gage, *"A New Algorithm for Data Compression"* (1994). Byte-level BPE: Radford et al., GPT-2 (2019).

Every language model in this repository so far reads **characters**. Real LLMs read **tokens**, and almost all of them use BPE or a close relative (WordPiece in BERT, Unigram/SentencePiece in T5 and Llama 1–2).

The problem it solves:
- **Word-level** vocabularies are huge and still miss rare words, names and typos (the "unknown word" problem in 2015 machine translation).
- **Character-level** sequences have no unknown words but are long, so the model spends its context and compute on spelling.

**BPE** starts from characters (or, in GPT-2 onward, the 256 possible **bytes**) and repeatedly **merges the most frequent adjacent pair** into a new symbol. Frequent words become one token, rare words split into pieces, and nothing is ever out of vocabulary.
#%%
{{SETUP}}
from collections import Counter
#%%
{{PROCEDURAL}}
#%% md
## Training BPE from scratch
Byte-level, GPT-2 style: split the text into "words" (a space attaches to the following word), turn each into its UTF-8 bytes, then merge.
#%%
import re
PRETOK = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+")

def train_bpe(text, n_merges):
    words = Counter(tuple(w.encode("utf-8")) for w in PRETOK.findall(text))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}                                                               # (a, b) -> new id, in order of creation
    for k in range(n_merges):
        pairs = Counter()
        for w, f in words.items():
            for a, b in zip(w, w[1:]): pairs[a, b] += f
        if not pairs: break
        (a, b), freq = pairs.most_common(1)[0]
        new = 256 + k; merges[a, b] = new; vocab[new] = vocab[a] + vocab[b]
        merged = Counter()
        for w, f in words.items():                                            # apply the merge everywhere
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b: out.append(new); i += 2
                else: out.append(w[i]); i += 1
            merged[tuple(out)] += f
        words = merged
    return merges, vocab

def encode_word(b, merges):
    ids = list(b)
    while len(ids) > 1:                                                       # apply the earliest-learned merge first
        pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, 1e18))
        if pair not in merges: break
        new, out, i = merges[pair], [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair: out.append(new); i += 2
            else: out.append(ids[i]); i += 1
        ids = out
    return ids

_caches = {}
def bpe_encode(s, merges):
    cache = _caches.setdefault(len(merges), {})                             # merge lists here are prefixes of one list
    ids = []
    for w in PRETOK.findall(s):
        if w not in cache: cache[w] = encode_word(w.encode("utf-8"), merges)
        ids += cache[w]
    return ids
def show(i):
    """A token as text if it is valid UTF-8 on its own, otherwise as its raw bytes."""
    try: return vocab[i].decode("utf-8")
    except UnicodeDecodeError: return "<" + vocab[i].hex(" ") + ">"
def bpe_decode(ids, vocab): return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

t0 = time.time(); merges, vocab = train_bpe(text[:split], 300)
print(f"300 merges learned in {time.time() - t0:.2f}s ({len(set(PRETOK.findall(text[:split])))} distinct pre-tokens: this corpus has a small vocabulary)")
print("first 12 merges:", [vocab[i].decode() for i in range(256, 268)])
print("merges 100–112: ", [vocab[i].decode() for i in range(356, 368)])
#%% md
The first merges are frequent letter pairs ("e", "th"-like pieces, " t"); later ones are whole words with their leading space, like " the" and " learns".
## Round trip, rare words, and anything at all
#%%
for s in ["The model learns the next token step by step.", "Transformerization of hyperparameters!", "naïve café 東京 🙂"]:
    ids = bpe_encode(s, merges)
    print(f"{len(s.encode()):3d} bytes → {len(ids):3d} tokens | {[show(i) for i in ids]}")
    assert bpe_decode(ids, vocab) == s                                      # lossless, even for unseen scripts
#%% md
Unseen words split into known pieces; characters never seen in training fall back to raw bytes (shown as `<hex>`; "ï" is two bytes, "東" three, the emoji four). There is no "unknown" token. Our corpus has only a few dozen distinct words, so after 300 merges every one of them is a single token; real tokenizers train on terabytes of diverse text and keep subword pieces for the long tail.
## Vocabulary size vs. sequence length
#%%
val_text = text[split:]
for n in [0, 25, 50, 100, 200, 300]:
    m = {k: v for k, v in merges.items() if v < 256 + n}
    print(f"{256 + n:4d} tokens in vocab → {len(val_text) / len(bpe_encode(val_text, m)):.2f} characters per token on held-out text")
#%% md
## Does it help a language model?
Train the same tiny GPT twice, on characters (notebook 41's setup) and on BPE tokens (556 = 256 bytes + 300 merges), with the same context of 64 *tokens* and the same number of steps. Losses are converted to **bits per character** so they are comparable.
#%%
{{TINYGPT}}
#%%
tok = torch.tensor(bpe_encode(text, merges), device=device)
tsplit = len(bpe_encode(text[:split], merges))
tok_train, tok_val = tok[:tsplit], tok[tsplit:]
chars_per_tok = split / tsplit
results = {}
for name, d_train, d_val, vocab_size, cpt in [("characters", train_data, val_data, V, 1.0), ("BPE (556)", tok_train, tok_val, 556, chars_per_tok)]:
    seed_everything(0); t0 = time.time()
    m = train_gpt(GPT(vocab_size, ctx=64), d_train, steps=800)
    loss = eval_loss(m, d_val)
    results[name] = m
    print(f"{name:11s} val loss {loss:.3f} nats/token = {loss / math.log(2) / cpt:.3f} bits/char | context = {64 * cpt:.0f} characters | {time.time() - t0:.0f}s")
#%%
seed_everything(1)
idx = torch.tensor([bpe_encode("The model", merges)], device=device)
with torch.no_grad():
    for _ in range(40):
        p = F.softmax(results["BPE (556)"](idx[:, -64:])[:, -1] / 0.8, -1); idx = torch.cat([idx, torch.multinomial(p, 1)], 1)
print(bpe_decode(idx[0].tolist(), vocab))
#%% md
## Takeaways
- **BPE = greedy compression.** Merge the most frequent pair, repeat. The merge list *is* the tokenizer; encoding replays merges in order.
- **Byte-level** BPE (GPT-2 onward) can encode any string losslessly, with no unknown token.
- Each token carries several characters, so the same context window covers several times more text and each prediction step covers more of it; per character, the model is also cheaper to run.
- **Costs:** tokenization is why LLMs struggle to count letters in a word and why some languages need many more tokens per sentence than English. Modern tokenizers use 100K–260K tokens (e.g. o200k in GPT-4o, 256K in Gemma) trained on multilingual data, and research on tokenizer-free byte models (e.g. BLT, 2024) continues.
