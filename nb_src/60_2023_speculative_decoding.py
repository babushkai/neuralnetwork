#%% md
# 2023 — Speculative Decoding: Faster Generation, Same Distribution

**Papers:** Yaniv Leviathan, Matan Kalman & Yossi Matias, *"Fast Inference from Transformers via Speculative Decoding"* (2022/ICML 2023); Charlie Chen et al., *"Accelerating Large Language Model Decoding with Speculative Sampling"* (DeepMind, 2023). Successors: Medusa (2024), EAGLE (2024), and **multi-token-prediction heads** used as drafters (DeepSeek-V3, Qwen3.5; notebook 68).

Generating with a big LLM is slow because it is **one token per forward pass**, and each pass is limited by memory bandwidth, not arithmetic. Checking $k$ tokens in one pass costs about the same as generating one.

**Speculative sampling:**
1. a small, fast **draft** model $q$ proposes $\gamma$ tokens
2. the big **target** model $p$ scores all of them in **one** forward pass
3. accept each draft token $x$ with probability $\min\!\big(1, \frac{p(x)}{q(x)}\big)$; at the first rejection, resample from the residual $\operatorname{norm}(\max(0, p - q))$ and stop
4. if all $\gamma$ are accepted, sample one bonus token from $p$

This is **mathematically exact**: the output has precisely the target model's distribution, not an approximation. The speed-up depends on how often the draft agrees with the target.
#%%
{{SETUP}}
#%%
{{PROCEDURAL}}
#%%
{{TINYGPT}}

seed_everything(0); t0 = time.time()
target = train_gpt(GPT(V, d=192, n_layers=6, heads=6), train_data, steps=800)
seed_everything(0)
draft = train_gpt(GPT(V, d=48, n_layers=1, heads=2), train_data, steps=800)
n_params = lambda m: sum(p.numel() for p in m.parameters())
print(f"target: {n_params(target) / 1e6:.2f}M params, val loss {eval_loss(target, val_data):.3f}")
print(f"draft : {n_params(draft) / 1e6:.3f}M params ({n_params(target) / n_params(draft):.0f}x smaller), val loss {eval_loss(draft, val_data):.3f}")
print(f"(trained in {time.time() - t0:.0f}s)")
#%%
@torch.no_grad()
def probs(model, seq, temperature):
    return F.softmax(model(seq[:, -model.ctx:])[0] / temperature, -1)          # next-token distributions at every position

@torch.no_grad()
def generate_naive(prompt, n, temperature=1.0):
    seq = torch.tensor([encode(prompt)], device=device); calls = 0
    while seq.shape[1] < len(prompt) + n:
        p = probs(target, seq, temperature)[-1]; calls += 1
        seq = torch.cat([seq, torch.multinomial(p, 1)[None]], 1)
    return seq[0, len(prompt):len(prompt) + n].tolist(), calls

@torch.no_grad()
def generate_speculative(prompt, n, gamma=4, temperature=1.0):
    seq = torch.tensor([encode(prompt)], device=device); target_calls, accepted = 0, 0
    while seq.shape[1] < len(prompt) + n:
        # 1) draft proposes gamma tokens autoregressively (cheap)
        draft_seq, q_dists = seq, []
        for _ in range(gamma):
            q = probs(draft, draft_seq, temperature)[-1]; q_dists.append(q)
            draft_seq = torch.cat([draft_seq, torch.multinomial(q, 1)[None]], 1)
        proposal = draft_seq[0, seq.shape[1]:]
        # 2) target scores the prefix + all gamma proposals in ONE forward pass
        p_all = probs(target, draft_seq, temperature)[-(gamma + 1):]             # gamma + 1 distributions: after the prefix and after each proposal
        target_calls += 1
        # 3) accept / reject left to right
        new = []
        for i, x in enumerate(proposal.tolist()):
            p, q = p_all[i], q_dists[i]
            if torch.rand(()) < torch.clamp(p[x] / q[x], max=1.0):
                new.append(x); accepted += 1
            else:
                residual = torch.clamp(p - q, min=0); new.append(torch.multinomial(residual / residual.sum(), 1).item())
                break
        else:
            new.append(torch.multinomial(p_all[gamma], 1).item())                 # 4) all accepted: free bonus token
        seq = torch.cat([seq, torch.tensor([new], device=device)], 1)
    return seq[0, len(prompt):len(prompt) + n].tolist(), target_calls, accepted

prompt = "The model "
out, calls, acc = generate_speculative(prompt, 300, gamma=4, temperature=0.8)
print(prompt + decode(out))
print(f"\n300 tokens with {calls} target forward passes ({300 / calls:.2f} tokens per target call; naive decoding needs 300)")
#%% md
## Exactness check
Compare the distribution of the first generated token under both methods with the target model's exact probabilities. Both should match within sampling noise. (The draft is noticeably worse than the target, so a biased method would show up.)
#%%
prompt = "The attention head "
exact = probs(target, torch.tensor([encode(prompt)], device=device), 1.0)[-1].cpu()
N = 3000
naive_counts, spec_counts = torch.zeros(V), torch.zeros(V)
for _ in range(N):
    naive_counts[generate_naive(prompt, 1)[0][0]] += 1
    spec_counts[generate_speculative(prompt, 1, gamma=3)[0][0]] += 1
draft_exact = probs(draft, torch.tensor([encode(prompt)], device=device), 1.0)[-1].cpu()
tv = lambda a, b: 0.5 * (a - b).abs().sum().item()
print(f"total-variation distance from the exact target distribution ({N} samples each):")
print(f"  naive sampling       {tv(naive_counts / N, exact):.3f}")
print(f"  speculative sampling {tv(spec_counts / N, exact):.3f}")
print(f"  (the draft model's own distribution is {tv(draft_exact, exact):.3f} away, so a method that leaked draft bias would show)")
#%% md
## Speed-up vs. draft length γ
Expected tokens per target call is $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$, where $\alpha$ is the per-token acceptance rate. Longer drafts help only while the draft keeps agreeing with the target.
#%%
rows = []
for gamma in [1, 2, 4, 6, 8]:
    _, calls, acc = generate_speculative("Each layer ", 400, gamma=gamma, temperature=0.8)
    rows.append((gamma, 400 / calls, acc / (calls * gamma)))
    print(f"gamma={gamma}: {400 / calls:.2f} tokens per target call | draft acceptance rate {acc / (calls * gamma):.2f}")
plt.plot([r[0] for r in rows], [r[1] for r in rows], marker="o"); plt.xlabel("draft length γ"); plt.ylabel("tokens per target forward pass")
plt.title("Fewer expensive target calls, identical output distribution"); plt.show()
