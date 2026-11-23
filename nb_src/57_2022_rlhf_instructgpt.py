#%% md
# 2022 — RLHF: Reinforcement Learning from Human Feedback (InstructGPT)

**Papers:** Long Ouyang et al., *"Training language models to follow instructions with human feedback"*, NeurIPS (2022) (InstructGPT, the method behind ChatGPT). Earlier: Christiano et al., *"Deep reinforcement learning from human preferences"* (2017); Stiennon et al., *"Learning to summarize from human feedback"* (2020). Reward over-optimization: Gao, Schulman & Hilton (2022).

A pre-trained LM imitates its training text; it is not trying to be helpful. InstructGPT aligned GPT-3 with what people want in three steps:

1. **SFT:** fine-tune on demonstrations written by people.
2. **Reward model (RM):** people compare two model outputs for the same prompt. Train a model $r_\phi$ to score them, with the Bradley–Terry loss $-\log\sigma\big(r_\phi(y_{win}) - r_\phi(y_{lose})\big)$.
3. **RL with PPO** (notebook 38): maximise the learned reward, with a penalty for drifting from the SFT model $\pi_{ref}$:
$$\max_\pi\ \mathbb E_{y\sim\pi}\big[r_\phi(y)\big] - \beta\,\mathrm{KL}\big(\pi\,\Vert\,\pi_{ref}\big)$$

A 1.3B InstructGPT model was preferred by labellers over the 175B GPT-3. The KL term matters because the RM is only an approximation of human judgement: optimise it too hard and the policy finds outputs the RM scores highly but people would not ("reward hacking", Goodhart's law).

**Toy version.** The "human" is a hidden labelling function: it prefers continuations that **mention the teacher or say "carefully"**, and strongly dislikes **gibberish** (words that are not real words of the corpus). Only pairwise comparisons from this labeller are given to the reward model. We can then measure what RLHF actually does to the true preference, not just to the learned reward.
#%% md
![Diagram: RLHF / InstructGPT (Ouyang et al., 2022)](https://raw.githubusercontent.com/babushkai/neuralnetwork/main/assets/diagrams/57_2022_rlhf_instructgpt.png)
#%%
{{SETUP}}
#%%
{{PROCEDURAL}}
#%%
{{TINYGPT}}
#%% md
## Step 1: the base / SFT model
Our "SFT" model is a small character-level GPT trained on the corpus. (In InstructGPT this step used human-written demonstrations.)
#%%
seed_everything(0); t0 = time.time()
sft = train_gpt(GPT(V, ctx=96), train_data, steps=1000)
print(f"base model val loss {eval_loss(sft, val_data):.3f} ({time.time() - t0:.0f}s)")
#%%
import re, copy
WORDS = set(re.sub(r"[^a-z ]", " ", text.lower()).split())
def true_reward(s):
    """The hidden labeller. +1 for 'teacher', +1 for 'carefully', -1 per gibberish word."""
    words = re.sub(r"[^a-z ]", " ", s.lower()).split()
    return float("teacher" in words) + float("carefully" in words) - sum(w not in WORDS for w in words)

PROMPTS = ["The model ", "A student ", "Each layer ", "The router ", "The optimizer "]
GEN = 48
@torch.no_grad()
def generate(model, n, temp=1.0, seed=None):
    if seed is not None: torch.manual_seed(seed)
    p = [PROMPTS[i % len(PROMPTS)] for i in range(n)]
    out = []
    for prompt in PROMPTS:                                                   # prompts differ in length: generate per prompt
        k = sum(1 for x in p if x == prompt)
        if k == 0: continue
        ids = torch.tensor([encode(prompt)] * k, device=device)
        for _ in range(GEN):
            logits = model(ids[:, -model.ctx:])[:, -1] / temp
            ids = torch.cat([ids, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        out += [(prompt, ids[i].tolist()) for i in range(k)]
    return out
samples = generate(sft, 10, seed=0)
for prompt, ids in samples[:4]: print(repr(decode(ids)), "→ true reward", true_reward(decode(ids)[len(prompt):]))
#%% md
## Step 2: the reward model from pairwise preferences
Sample pairs of continuations from the SFT model, let the hidden labeller pick the one with higher true reward (ties broken randomly, plus 10% label noise, since real labellers disagree), and train a reward model: the same Transformer body with a scalar head on the last token.
#%%
class RewardModel(nn.Module):
    def __init__(self, base):
        super().__init__(); self.body = copy.deepcopy(base); self.body.head = nn.Identity(); self.v = nn.Linear(base.head.in_features, 1)
    def forward(self, ids): return self.v(self.body(ids)[:, -1]).squeeze(-1)

def to_tensor(samples):
    L = max(len(ids) for _, ids in samples)
    return torch.tensor([[0] * (L - len(ids)) + ids for _, ids in samples], device=device)   # left-pad so the last token is real
seed_everything(0)
pool = generate(sft, 4000, seed=1)
rew = [true_reward(decode(ids)[len(p):]) for p, ids in pool]
pairs = []
for i in range(0, len(pool) - 1, 2):
    a, b = rew[i], rew[i + 1]; pref = (a > b) if a != b else (random.random() < 0.5)
    if random.random() < 0.1: pref = not pref
    pairs.append((i, i + 1) if pref else (i + 1, i))
ids_all = to_tensor(pool)
rm = RewardModel(sft).to(device); opt = torch.optim.AdamW(rm.parameters(), lr=3e-4, weight_decay=0.01)
tr_pairs, te_pairs = pairs[:1800], pairs[1800:]
for s in range(1500):
    b = [tr_pairs[j] for j in np.random.randint(0, len(tr_pairs), 64)]
    w, l = torch.tensor([x for x, _ in b], device=device), torch.tensor([y for _, y in b], device=device)
    loss = -F.logsigmoid(rm(ids_all[w]) - rm(ids_all[l])).mean(); opt.zero_grad(); loss.backward(); opt.step()
rm.eval()
with torch.no_grad():
    w, l = torch.tensor([x for x, _ in te_pairs], device=device), torch.tensor([y for _, y in te_pairs], device=device)
    print(f"reward model agrees with held-out (noisy) preferences: {(rm(ids_all[w]) > rm(ids_all[l])).float().mean().item():.3f}")
print(f"SFT samples: mean true reward {np.mean(rew):.3f}; with 'teacher' {np.mean([r >= 1 for r in rew]):.2f}; any gibberish word {np.mean([r < 0 for r in rew]):.3f}")
#%% md
## Step 3: PPO against the reward model, with a KL penalty
The policy starts as a copy of the SFT model. Each iteration samples continuations, scores them with the RM, subtracts $\beta$ times the per-sequence KL estimate $\sum_t \log\frac{\pi(y_t)}{\pi_{ref}(y_t)}$, and takes several clipped PPO steps. For brevity the advantage is the reward minus the batch mean (as in RLOO/GRPO-style methods) rather than InstructGPT's learned value function.
#%%
def seq_logprobs(model, ids, n_prompt):
    logp = F.log_softmax(model(ids[:, :-1]), -1).gather(2, ids[:, 1:, None]).squeeze(-1)
    return logp[:, n_prompt - 1:]                                             # only the generated tokens

def rlhf(beta, iters=80, B=64, epochs=4, clip=0.2, lr=2e-5, seed=0):
    seed_everything(seed); policy = copy.deepcopy(sft).train(); opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    hist = []
    for it in range(iters):
        prompt = PROMPTS[it % len(PROMPTS)]; n_p = len(prompt)
        with torch.no_grad():
            ids = torch.tensor([encode(prompt)] * B, device=device)
            for _ in range(GEN): ids = torch.cat([ids, torch.multinomial(F.softmax(policy(ids[:, -policy.ctx:])[:, -1], -1), 1)], 1)
            old_lp, ref_lp = seq_logprobs(policy, ids, n_p), seq_logprobs(sft, ids, n_p)
            kl = (old_lp - ref_lp).sum(1)
            r_rm = rm(ids); r = r_rm - beta * kl
            adv = (r - r.mean()) / (r.std() + 1e-8)
        for _ in range(epochs):
            lp = seq_logprobs(policy, ids, n_p); ratio = (lp - old_lp).exp()
            loss = -torch.min(ratio * adv[:, None], ratio.clamp(1 - clip, 1 + clip) * adv[:, None]).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        texts = [decode(x)[n_p:] for x in ids.tolist()]
        hist.append((r_rm.mean().item(), kl.mean().item(), np.mean([true_reward(t) for t in texts]), np.mean([true_reward(t) < 0 for t in texts]), len(set(texts)) / len(texts)))
    return policy.eval(), np.array(hist)

runs = {}
for beta in [0.0, 0.2, 1.0]:
    t0 = time.time(); runs[beta] = rlhf(beta); h = runs[beta][1]
    print(f"β = {beta:4.2f}: RM reward {h[:5, 0].mean():+.2f} → {h[-10:, 0].mean():+.2f} | KL to SFT {h[-10:, 1].mean():6.1f} nats | "
          f"TRUE reward {h[:5, 2].mean():+.2f} → {h[-10:, 2].mean():+.2f} | gibberish rate {h[-10:, 3].mean():.2f} | distinct samples {h[:5, 4].mean():.2f} → {h[-10:, 4].mean():.2f} | {time.time() - t0:.0f}s")
#%%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for beta, (_, h) in runs.items():
    sm = lambda v: np.convolve(v, np.ones(5) / 5, mode="valid")
    axes[0].plot(sm(h[:, 0]), label=f"β = {beta}"); axes[1].plot(sm(h[:, 2]), label=f"β = {beta}"); axes[2].plot(sm(h[:, 1]), label=f"β = {beta}")
axes[0].set(title="learned reward (what PPO optimises)", xlabel="PPO iteration"); axes[1].set(title="true reward (the hidden labeller)", xlabel="PPO iteration")
axes[2].set(title="KL(policy ‖ SFT) per sequence", xlabel="PPO iteration", yscale="symlog"); axes[0].legend()
plt.tight_layout(); plt.show()
for beta, (pol, _) in runs.items():
    print(f"--- β = {beta}")
    for prompt, ids in generate(pol, 3, seed=5): print("   ", repr(decode(ids)))
#%% md
## Takeaways
- **Preferences → reward model → RL.** Comparisons are easier to collect than demonstrations, and the RM turns them into a dense training signal for any output the policy produces.
- **The KL penalty is not decoration.** The RM is an imperfect proxy (it saw only SFT-like text and noisy labels; it agrees with held-out preferences only modestly). In the run above, all three policies push the learned reward from about +1 to +7…+9, but the true reward tells a different story. With $\beta=0$ the policy drifts furthest (KL ≈ 15 nats) and produces the most gibberish (28% of samples, e.g. "the los"), which the RM never learned to punish: reward hacking. A moderate $\beta=0.2$ gives the best true reward with little gibberish. A large $\beta=1$ keeps samples diverse and close to the SFT model but gains less. That trade-off is why $\beta$ is tuned in practice.
- **Mode collapse is a real side effect:** with weak or moderate KL, about half the samples become duplicates of a few high-reward sentences, a small-scale version of the reduced output diversity reported for RLHF-tuned models.
- **What came next:** DPO (notebook 58) removes the explicit RM and the RL loop by solving for the optimal policy in closed form; RL with verifiable rewards (GRPO, notebook 69) replaces the learned RM with a checker wherever answers can be verified, which removes this kind of reward hacking for maths and code.
