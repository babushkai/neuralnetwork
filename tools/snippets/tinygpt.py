class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj, self.h = nn.Linear(d, 3 * d), nn.Linear(d, d), heads
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    """A small GPT-2-style character model (pre-norm blocks, learned positions)."""
    def __init__(self, V, d=128, n_layers=4, heads=4, ctx=128):
        super().__init__()
        self.ctx = ctx
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(n_layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)
    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks: x = b(x)
        return self.head(self.ln(x))

def train_gpt(model, train_data, steps=800, B=32, lr=2e-3):
    model = model.to(train_data.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    for s in range(steps):
        ix = torch.randint(0, len(train_data) - model.ctx - 1, (B,), device=train_data.device)
        x = torch.stack([train_data[i:i + model.ctx] for i in ix]); y = torch.stack([train_data[i + 1:i + model.ctx + 1] for i in ix])
        loss = F.cross_entropy(model(x).reshape(-1, model.head.out_features), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return model.eval()

@torch.no_grad()
def eval_loss(model, data, n=20, B=32):
    g = torch.Generator(device="cpu").manual_seed(123); losses = []
    for _ in range(n):
        ix = torch.randint(0, len(data) - model.ctx - 1, (B,), generator=g).to(data.device)
        x = torch.stack([data[i:i + model.ctx] for i in ix]); y = torch.stack([data[i + 1:i + model.ctx + 1] for i in ix])
        losses.append(F.cross_entropy(model(x).reshape(-1, model.head.out_features), y.reshape(-1)).item())
    return float(np.mean(losses))
