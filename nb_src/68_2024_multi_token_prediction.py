#%% md
# 2024 — Multi-Token Prediction (MTP)

**Papers:** Gloeckle et al., *"Better & Faster Large Language Models via Multi-token Prediction"* (Meta, 2024); DeepSeek-AI, *"DeepSeek-V3 Technical Report"* (2024), whose sequential MTP design is kept in DeepSeek-V4 (2026, MTP depth 1, loss weight 0.3 → 0.1); Qwen3-Next and Qwen3.5 also train with MTP.

Next-token prediction gives one learning signal per position. **MTP** adds small extra modules that predict tokens **further ahead**. In DeepSeek-V3's sequential design, MTP module $k$ combines the previous depth's hidden state with the embedding of the token $k$ steps ahead and predicts the token after it:

$$h'^{\,k}_i = M_k\big[\operatorname{RMSNorm}(h^{k-1}_i);\ \operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))\big],\qquad h^k_i = \operatorname{TRM}_k(h'^{\,k}_i),\qquad P^k_{i+k+1} = \operatorname{Head}(h^k_i)$$

The embedding and output head are **shared** with the main model, and the total loss is $\mathcal L_{main} + \lambda\,\mathcal L_{MTP}$. Two benefits:
1. **Denser training signal:** the model must plan representations that are useful several tokens ahead.
2. **Free drafting at inference:** the MTP module guesses the token after next, so it can serve as the draft model for speculative decoding (notebook 60) without a separate model. DeepSeek reports 85–90% acceptance for the second token.
#%%
{{SETUP}}
#%%
{{PROCEDURAL}}
#%%
{{TINYGPT}}

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6): super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x): return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class GPTWithMTP(GPT):
    """GPT + one DeepSeek-V3-style MTP module (depth 1) that predicts token i+2 from h_i and Emb(t_{i+1})."""
    def __init__(self, V, d=128, n_layers=4, heads=4, ctx=128):
        super().__init__(V, d, n_layers, heads, ctx)
        self.mtp_norm_h, self.mtp_norm_e = RMSNorm(d), RMSNorm(d)
        self.mtp_proj = nn.Linear(2 * d, d, bias=False)                                  # M_1
        self.mtp_block = Block(d, heads)                                                 # TRM_1 (causal)
    def hidden(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks: x = b(x)
        return x
    def forward_mtp(self, idx):
        h = self.hidden(idx)                                                             # (B, T, d)
        main_logits = self.head(self.ln(h))                                              # predicts t_{i+1}
        e_next = self.tok(idx[:, 1:])                                                    # Emb(t_{i+1}): shared embedding
        h1 = self.mtp_proj(torch.cat([self.mtp_norm_h(h[:, :-1]), self.mtp_norm_e(e_next)], -1))
        h1 = self.mtp_block(h1 + self.pos(torch.arange(h1.shape[1], device=idx.device)))
        mtp_logits = self.head(self.ln(h1))                                              # predicts t_{i+2}; shared head
        return main_logits, mtp_logits

def train(use_mtp, steps=800, lam=0.3, B=32):
    seed_everything(0)
    m = (GPTWithMTP(V) if use_mtp else GPT(V)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
    for s in range(steps):
        ix = torch.randint(0, len(train_data) - m.ctx - 2, (B,), device=device)
        x = train_data[ix[:, None] + torch.arange(m.ctx, device=train_data.device)]
        y1 = train_data[ix[:, None] + torch.arange(1, m.ctx + 1, device=train_data.device)]
        if use_mtp:
            main, mtp = m.forward_mtp(x)
            y2 = train_data[ix[:, None] + torch.arange(2, m.ctx + 1, device=train_data.device)]            # targets two steps ahead
            loss = F.cross_entropy(main.reshape(-1, V), y1.reshape(-1)) + lam * F.cross_entropy(mtp.reshape(-1, V), y2.reshape(-1))
        else:
            loss = F.cross_entropy(m(x).reshape(-1, V), y1.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    return m.eval()

t0 = time.time()
base, mtp_model = train(False), train(True)
print(f"trained both in {time.time() - t0:.0f}s")
print(f"next-token validation loss | plain GPT: {eval_loss(base, val_data):.4f} | GPT trained with MTP: {eval_loss(mtp_model, val_data):.4f}")
print("(the MTP module adds parameters only for training and drafting; the main next-token path is the same architecture)")
print("At this toy scale both models reach the same loss floor of the corpus; the paper-scale gain from the denser signal does not show here.")
#%% md
## Using the MTP module as a free draft model
At each position the main head predicts $t_{i+1}$; feeding that guess back into the MTP module predicts $t_{i+2}$. How often does that guess match what the main model itself would predict one step later (greedy)? That is the acceptance rate for self-speculative decoding.
#%%
@torch.no_grad()
def mtp_acceptance(m, n_batches=10, B=16):
    """At position i the main head guesses g1 = t_{i+1} and the MTP module guesses g2 = t_{i+2}.
    Verifying g2 means running the main model on prefix + g1. Wherever g1 equals the real next token,
    that is just the main model's own prediction at position i+1 on the real text, so we can check
    every such position in one batched pass."""
    g = torch.Generator().manual_seed(1); agree = total = 0
    for _ in range(n_batches):
        ix = torch.randint(0, len(val_data) - m.ctx - 2, (B,), generator=g).to(device)
        x = val_data[ix[:, None] + torch.arange(m.ctx, device=val_data.device)]
        h = m.hidden(x); g1 = m.head(m.ln(h)).argmax(-1)                                  # main model's greedy guesses
        h1 = m.mtp_proj(torch.cat([m.mtp_norm_h(h), m.mtp_norm_e(m.tok(g1))], -1))
        g2 = m.head(m.ln(m.mtp_block(h1 + m.pos(torch.arange(h1.shape[1], device=device))))).argmax(-1)
        ok = g1[:, :-1] == x[:, 1:]                                                      # positions where the prefix+g1 is the real text
        agree += ((g2[:, :-1] == g1[:, 1:]) & ok).sum().item(); total += ok.sum().item()
    return agree / total
rate = mtp_acceptance(mtp_model)
print(f"MTP draft of the 2nd token agrees with the main model's own greedy choice {rate:.0%} of the time")
print(f"=> self-speculative decoding would emit about {1 + rate:.2f} tokens per forward pass (greedy)")
