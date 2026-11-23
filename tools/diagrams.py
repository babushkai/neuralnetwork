r"""Architecture diagrams, one per notebook: writes assets/diagrams/<notebook stem>.png.

Usage: python tools/diagrams.py            (all)
       python tools/diagrams.py 13 29      (some)

Each diagram is drawn from the notebook's own implementation (layer sizes and choices as in the code).
Syntax of rows and edges: see tools/diagram_engine.py. " // " is a line break.
"""
import pathlib, sys
from diagram_engine import render

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = {}
def d(num, title, rows, edges=None, caption=""):
    D[num] = (title, rows, edges, caption)

# ---------------------------------------------------------------- 1943–1997
d("01", "McCulloch–Pitts neuron (1943)",
  [[r"$x_1$ #x1", r"$x_2$ #x2", r"$x_3$ (inhibitory) #x3"], [r"$\Sigma$ of excitatory inputs #s"],
   [r"fires if $\Sigma \geq \theta$ and no inhibitory input is active #t !gate"], [r"$y \in \{0, 1\}$ #y"]],
  "x1>s; x2>s; x3.)>t:veto; s>t; t>y", "Binary inputs, a fixed threshold θ, no learning: logic gates are wired by hand")
d("02", "Perceptron (Rosenblatt, 1958)",
  [[r"$x_1$ #x1", r"$x_2$ #x2", "… !note #dots", r"$x_n$ #xn", "1 (bias) #one"], [r"$z = w \cdot x + b$ #z"],
   [r"$\hat y = \mathrm{sign}(z)$ #g !gate", "label y #lab !in"], [r"if $\hat y \neq y$:  $w \leftarrow w + \eta\, y\, x$ #u !loss"]],
  "x1>z; x2>z; xn>z; one>z; z>g; g>u; lab>u; u.(>z:update w, b",
  "Learns only from mistakes; converges whenever the classes are linearly separable")
d("03", "ADALINE and the LMS rule (Widrow & Hoff, 1960)",
  [[r"$x_1$ #x1", r"$x_2$ #x2", "… !note #dots", r"$x_n$ #xn"], [r"$z = w \cdot x + b$ (linear output) #z"],
   [r"error $e = y - z$ #e !loss", r"$\mathrm{sign}(z)$ #q !gate"], [r"LMS: $w \leftarrow w + \eta\, e\, x$ #u !loss", r"$\hat y \in \{-1, +1\}$ #y !out"]],
  "x1>z; x2>z; xn>z; z>e; z>q; q>y; e>u; u.(>z:gradient step",
  "Unlike the perceptron, the error is taken before the threshold: gradient descent on squared error")
d("04", "XOR needs a hidden layer (Minsky & Papert, 1969)",
  [[r"$x_1$ #x1", r"$x_2$ #x2"], [r"$h_1$ (≈ OR) #h1", r"$h_2$ (≈ NAND) #h2"], [r"$y$ (≈ AND of $h_1, h_2$) = XOR #y"]],
  None, "One threshold unit draws one line and cannot separate XOR. Two hidden units can (one solution shown)")
d("05", "Neocognitron (Fukushima, 1980)",
  [["input image #x"], [r"$U_{S1}$: 16 shared 5×5 templates (S-cells), learned by competitive learning #s1"],
   [r"$U_{C1}$: C-cells pool neighbouring S-cells → tolerance to shifts #c1 !norm"],
   [r"$U_{S2}$: 64 templates over the $C_1$ maps #s2"], [r"$U_{C2}$: pool again #c2 !norm"], ["linear readout → digit class #y"]],
  None, "Alternating feature-detecting (S) and pooling (C) layers: the template for convolution + pooling, without backprop")
d("06", "Hopfield network (1982)",
  [[r"stored patterns $\xi^1 … \xi^P$ #p", "corrupted pattern s(0) #x"],
   [r"$W = \frac{1}{N}\sum_\mu \xi^\mu \xi^{\mu\top}$, $w_{ii} = 0$ (Hebbian, symmetric) #w !mem",
    r"update $s_i \leftarrow \mathrm{sign}(\sum_j w_{ij} s_j)$, one unit at a time #u !gate"],
   ["recalled pattern (a fixed point) #y"]],
  "x>u; p>w; w>u; u@:until stable; u>y", r"Energy $E = -\frac{1}{2}s^\top W s$ never increases, so recall settles in a stored minimum")
d("07", "Self-organizing map (Kohonen, 1982)",
  [["input x #x"], [r"best-matching unit $c = \mathrm{argmin}_j \|x - w_j\|$ #c !gate"],
   [r"$w_j \leftarrow w_j + \eta\, h(j, c)\,(x - w_j)$,  h: Gaussian neighbourhood on the grid #u !loss"],
   [r"2-D grid of units with weight vectors $w_j$ #m !mem"]],
  "x>c; m(>c:compare with every unit; c>u; u>m", "Neighbouring units learn similar inputs, so the grid becomes a topology-preserving map")
d("08", "Restricted Boltzmann machine (Hinton, 2002; Boltzmann machines, 1985)",
  [["visible units v (pixels) #v"], ["hidden units h (features) #h"],
   [r"CD-1: $\Delta W \propto \langle v h^\top \rangle_{data} - \langle v h^\top \rangle_{recon}$ #cd !loss"]],
  r"v>h:$p(h{=}1|v) = \sigma(W^\top v + c)$; h(>v:$p(v{=}1|h) = \sigma(W h + b)$; h>cd",
  "Bipartite: no connections within a layer, so each layer is sampled in parallel given the other")
d("09", "Backpropagation (Rumelhart, Hinton & Williams, 1986)",
  [["x #x"], [r"$h = \sigma(W_1 x + b_1)$ #h"], [r"$\hat y = \mathrm{softmax}(W_2 h + b_2)$ #y"], [r"loss $L(\hat y, y)$ #L !loss"]],
  r"x>h; h>y; y>L; L.(>y:$\delta_2 = \partial L/\partial z_2$; y.(>h:$\delta_1 = W_2^\top \delta_2 \odot \sigma'$",
  "Forward pass up, gradients down by the chain rule (checked against finite differences in the notebook)")
d("10", "LeNet-5 (LeCun et al., 1989/1998)",
  [["image 32×32 #x"], ["C1: conv 5×5, 6 maps → 28×28 #c1"], ["S2: average-pool 2×2 → 14×14 #s2 !norm"],
   ["C3: conv 5×5, 16 maps → 10×10 #c3"], ["S4: average-pool 2×2 → 5×5 #s4 !norm"], ["C5: conv 5×5, 120 maps → 1×1 #c5"],
   ["F6: fully connected, 84 #f6"], ["10 classes #y"]],
  None, "Local receptive fields, shared weights and subsampling, all trained end to end by backprop")
d("11", "Elman network (1990)",
  [[r"$x_t$ #x"], [r"$h_t = \tanh(W x_t + U h_{t-1} + b)$ #h"], [r"$y_t$ #y"]],
  r"x>h; h@:$h_{t-1}$; h>y", "The hidden state is fed back as context; trained by backpropagation through time")
d("12", "REINFORCE (Williams, 1992)",
  [[r"state $s_t$ #s"], [r"policy network $\pi_\theta(a \mid s)$ #pi"], [r"sample action $a_t \sim \pi_\theta$ #a !gate"],
   ["environment (CartPole) #env !mem"], [r"return $G_t = \sum_k \gamma^k r_{t+k}$ #G", r"baseline $b(s_t)$ #b !norm"],
   [r"$\nabla_\theta J \approx \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,(G_t - b(s_t))$ #g !loss"]],
  "s>pi; pi>a; a>env; env(>s:next state; env>G:rewards; G>g; b>g; g.(>pi:gradient ascent",
  "Unbiased for any baseline; a good baseline cuts the variance")
d("13", "LSTM (Hochreiter & Schmidhuber, 1997)",
  [[r"$x_t$ #x", r"$h_{t-1}$ #hp !mem"],
   [r"forget $f_t = \sigma(\cdot)$ #f !gate", r"input $i_t = \sigma(\cdot)$ #i !gate", r"candidate $g_t = \tanh(\cdot)$ #g", r"output $o_t = \sigma(\cdot)$ #o !gate"],
   [r"$c_t = f_t \odot c_{t-1} + i_t \odot g_t$ #c !mem"], [r"$h_t = o_t \odot \tanh(c_t)$ #h"]],
  r"auto0-1; f>c; i>c; g>c; c@:$c_{t-1}$; c>h; o>h",
  "The additive cell update carries gradients across long gaps (forget gate: Gers et al., 2000)")

# ---------------------------------------------------------------- 2001–2016
d("14", "Echo state network (Jaeger, 2001)",
  [[r"input $u_t$ #u"], [r"reservoir: fixed random sparse $W$, spectral radius $\rho$ // $x_t = \tanh(W_{in} u_t + W x_{t-1})$ #r !mem"],
   [r"linear readout $W_{out}$ (ridge regression: the only trained part) #o"], [r"prediction $y_t$ #y"]],
  "u>r; r@:fixed; r>o; o>y; y.(>u:free-running: prediction fed back as input",
  "Echo state property: with ρ below about 1 the reservoir forgets its initial state")
d("15", "Deep autoencoder with greedy pre-training (Hinton & Salakhutdinov, 2006)",
  [["x: 64 pixels #x"], ["encoder layers, each pre-trained alone as a shallow autoencoder #e"], ["2-D code #z !mem"],
   ["decoder: the mirror of the encoder #d"], [r"reconstruction $\hat x$ #y"], [r"loss $\|x - \hat x\|^2$, then fine-tune end to end #L !loss"]],
  None, "Greedy layer-wise pre-training, then backprop through the whole stack; compared with PCA")
d("16", "Weight initialization (Glorot & Bengio, 2010; He et al., 2015)",
  [["x (standardized) #x"], [r"Linear $W_1$, $W \sim N(0, \sigma^2)$ #l1"], ["ReLU or tanh #a1 !gate"], ["⋮  (20–50 layers) #dots !note"],
   [r"Linear $W_L$ #lL"], ["output #y"]],
  None, r"Xavier: $\sigma^2 = 2/(n_{in} + n_{out})$ for tanh.  He: $\sigma^2 = 2/n_{in}$ for ReLU. // Both keep the variance of activations and gradients constant from layer to layer")
d("17", "AlexNet, scaled down (Krizhevsky, Sutskever & Hinton, 2012)",
  [["augmented image 32×32 #x"], ["conv 5×5, 32 → ReLU → max-pool #c1"], ["conv 3×3, 64 → ReLU → max-pool #c2"],
   ["conv 3×3, 96 → ReLU #c3"], ["conv 3×3, 64 → ReLU → max-pool #c4"], ["dropout → FC 256 → ReLU #f1"],
   ["dropout → FC 256 → ReLU #f2"], ["FC → 10 classes #y"]],
  None, "The original's recipe at toy scale: ReLU, max-pooling, dropout, data augmentation, a GPU")
d("18", "word2vec: skip-gram with negative sampling (Mikolov et al., 2013)",
  [["centre word w #w"], [r"input vector $v_w$ (matrix V) #v !mem", r"output vectors $u$ (matrix U) #u !mem"],
   [r"$\sigma(u_c \cdot v_w)$ for a true context word c #pos !gate", r"$\sigma(-u_n \cdot v_w)$ for k random negative words #neg !gate"],
   [r"maximize $\log \sigma(u_c \cdot v_w) + \sum_n \log \sigma(-u_n \cdot v_w)$ #L !loss"]],
  "w>v; v>pos; v>neg; u>pos; u>neg; pos>L; neg>L", "The rows of V become the word embeddings")
d("19", "Variational autoencoder (Kingma & Welling, 2013)",
  [["x #x"], [r"encoder $q_\phi(z \mid x)$ #enc"], [r"$\mu$ #mu", r"$\log \sigma^2$ #ls"],
   [r"$z = \mu + \sigma \odot \epsilon$ #z !gate", r"$\epsilon \sim N(0, I)$ #eps !in"], [r"decoder $p_\theta(x \mid z)$ #dec"],
   [r"loss $= -\log p_\theta(x \mid z) + \mathrm{KL}(q_\phi(z \mid x)\, \|\, N(0, I))$ #L !loss"]],
  "x>enc; enc>mu; enc>ls; mu>z; ls>z; eps>z; z>dec; dec>L:reconstruction; ls)>L:KL",
  "Reparameterization puts the randomness in ε, so gradients flow through z to the encoder")
d("20", "Deep Q-Network (Mnih et al., 2013/2015)",
  [["state s #s"], [r"Q-network $Q_\theta(s, \cdot)$ #q"], ["ε-greedy action a #a !gate"], ["environment (CartPole) #env !mem"],
   [r"replay buffer of $(s, a, r, s')$ #buf !mem", r"target network $Q_{\theta^-}$ #tgt !norm"],
   [r"loss $(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a))^2$ on random minibatches #L !loss"]],
  "s>q; q>a; a>env; env(>s:next state; env>buf:store; buf>L:sample; tgt>L; L.(>q:update θ; q.)>tgt:copy θ every C steps",
  "Replay breaks the correlation between consecutive samples; the target network holds the regression target still")
d("21", "Dropout (Hinton et al., 2012; Srivastava et al., 2014)",
  [["x #x"], ["Linear → ReLU #h1"], ["dropout: zero each unit with probability p (training only) #d1 !gate"], ["Linear → ReLU #h2"],
   ["dropout #d2 !gate"], ["Linear → classes #y"]],
  None, "At test time keep every unit and scale (weight scaling), or average many random masks (Monte Carlo dropout)")
d("22", "Generative adversarial network (Goodfellow et al., 2014)",
  [[r"noise $z \sim N(0, I)$ #z"], ["generator G #g"], ["fake sample G(z) #f !mem", "real sample x #x !in"],
   ["discriminator D → P(real) #d !gate"],
   [r"D: maximize $\log D(x) + \log(1 - D(G(z)))$;   G: maximize $\log D(G(z))$ #L !loss"]],
  "z>g; g>f; f>d; x>d; d>L; L.(>g:train G to fool D; L.)>d:train D",
  "The two networks train against each other. Failure mode shown in the notebook: mode collapse")
d("23", "GRU and sequence-to-sequence (Cho et al.; Sutskever et al., 2014)",
  [[r"source tokens $x_1 … x_S$ #x"], ["GRU encoder #e", r"GRU: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t$ // gates $z_t, r_t = \sigma(\cdot)$ #n !note"],
   ["final hidden state c: one fixed-size vector #c !mem"], ["GRU decoder, started from c #dd"], [r"target tokens $y_1 … y_T$ #y"]],
  r"x>e; e@; e>c; c>dd; dd@:previous output $y_{t-1}$; dd>y",
  "The whole source must squeeze through one vector c: the bottleneck that attention removed")
d("24", "Bahdanau attention (Bahdanau, Cho & Bengio, 2014)",
  [["source tokens #x"], [r"bidirectional GRU encoder → $h_1 … h_S$ #h !mem"],
   [r"scores $e_{t,s} = v^\top \tanh(W s_{t-1} + U h_s)$ #e !gate", r"decoder state $s_{t-1}$ #s"],
   [r"weights $\alpha_{t,s} = \mathrm{softmax}_s(e_{t,s})$ #a !gate"], [r"context $c_t = \sum_s \alpha_{t,s} h_s$ #c"],
   [r"decoder GRU: $s_t = f(s_{t-1}, y_{t-1}, c_t)$ #dd"], [r"$y_t$ #y"]],
  r"x>h; h>e; s>e; e>a; a>c; h(>c:values $h_s$; c>dd; s)>dd; dd>y",
  "A new weighted summary of the source at every output step; the weights are learned alignments")
d("25", "Adam (Kingma & Ba, 2014) and AdamW (Loshchilov & Hutter, 2017)",
  [[r"gradient $g_t$ #g"], [r"momentum $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ #m !mem", r"scale $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ #v !mem"],
   [r"bias correction $\hat m_t = m_t / (1 - \beta_1^t)$,  $\hat v_t = v_t / (1 - \beta_2^t)$ #bc"],
   [r"$\theta \leftarrow \theta - \eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon) - \eta \lambda \theta$ #u"]],
  "g>m; g>v; m>bc; v>bc; bc>u",
  "Momentum (from SGD with momentum) plus per-parameter scaling (from RMSProp). // AdamW: the decay term −ηλθ is applied directly, outside the adaptive scaling")
d("26", "VGG and Inception (2014): the Inception module",
  [["previous layer #x"], ["1×1 conv #b1", "1×1 conv (bottleneck) #r3 !norm", "1×1 conv (bottleneck) #r5 !norm", "3×3 max-pool #p !norm"],
   ["_ #s1", "3×3 conv #c3", "5×5 conv #c5", "1×1 conv #pp"], ["concatenate along channels #cat"]],
  "x>b1; x>r3; x>r5; x>p; r3>c3; r5>c5; p>pp; b1>cat; c3>cat; c5>cat; pp>cat",
  "Inception: parallel branches, with 1×1 bottlenecks to cut compute. // VGG: two stacked 3×3 convolutions see 5×5 with fewer parameters")
d("27", "Neural Turing Machine (Graves, Wayne & Danihelka, 2014)",
  [[r"input $x_t$ #x"], ["controller (LSTM) #c"],
   ["heads: key k, strength β, gate g, shift s, sharpening γ // → weights $w_t$ (content + location addressing) #h !gate", r"output $y_t$ #y !out"],
   [r"read: $r_t = \sum_i w_t(i)\, M_t(i)$ #rd", r"write: $M_t = M_{t-1} \odot (1 - w_t e_t^\top) + w_t a_t^\top$ #wr"],
   [r"memory matrix $M_t$ (N slots) #M !mem"]],
  r"x>c; c@; c>h; c>y; h>rd; h>wr; wr>M; M>rd; rd(>c:$r_t$ at the next step",
  "Every read and write is a soft, differentiable weighting over all N slots")
d("28", "Batch normalization (Ioffe & Szegedy, 2015)",
  [["mini-batch activations x #x"], [r"$\mu_B$, $\sigma_B^2$: mean and variance over the batch #st !norm"],
   [r"$\hat x = (x - \mu_B) / \sqrt{\sigma_B^2 + \epsilon}$ #n !norm"], [r"$y = \gamma \hat x + \beta$ (learned scale and shift) #y"]],
  "x>st; st>n; x)>n; n>y", "Usually conv/linear → BN → ReLU. At test time, running averages replace the batch statistics")
d("29", "ResNet basic block (He et al., 2015)",
  [[r"$x_l$ #x"], ["3×3 conv → BN → ReLU #c1"], ["3×3 conv → BN #c2"], ["+ #add !circle"], ["ReLU #r !gate"], [r"$x_{l+1}$ #o"]],
  "x>c1; c1>c2; c2>add; x)>add:identity shortcut; add>r; r>o",
  "The block learns a residual F(x); the identity path carries signal and gradient through any depth")
d("30", "U-Net (Ronneberger, Fischer & Brox, 2015)",
  [["_ #a0", "bottleneck: double conv, 8c #b", "_ #a1"], ["encoder 3: pool → double conv, 4c #e3", "_ #a2", "decoder 3: up-conv → concat → double conv #d3"],
   ["encoder 2: pool → double conv, 2c #e2", "_ #a3", "decoder 2 #d2"], ["encoder 1: double conv, c #e1", "_ #a4", "decoder 1 #d1"],
   ["input image #x !in", "_ #a5", "1×1 conv → per-pixel classes #y !out"]],
  "x>e1; e1>e2:max-pool; e2>e3; e3>b; b>d3:up-conv; d3>d2; d2>d1; d1>y; e1>d1:copy; e2>d2:copy; e3>d3:copy",
  "Skip connections hand fine spatial detail from each encoder level straight to the decoder")
d("31", "Knowledge distillation (Hinton, Vinyals & Dean, 2015)",
  [["input x #x"], ["teacher (large, frozen) #t !norm", "student (small) #s"],
   [r"softmax($z_t / T$) #pt !mem", r"softmax($z_s / T$) #ps", r"softmax($z_s$) #ps1"],
   [r"$T^2\, \mathrm{KL}$(teacher ‖ student) #kd !loss", "cross-entropy with labels y #ce !loss"],
   [r"loss $= \alpha \cdot \mathrm{KD} + (1 - \alpha) \cdot \mathrm{CE}$ #L !loss"]],
  "x>t; x>s; t>pt; s>ps; s>ps1; pt>kd; ps>kd; ps1>ce; kd>L; ce>L; L.)>s:update the student only",
  "A temperature T > 1 exposes how similar the teacher finds the wrong classes")
d("32", "WaveNet (van den Oord et al., 2016)",
  [[r"past samples $x_{<t}$ (μ-law, 256 levels) #x"], ["causal convolution #cc"], ["residual block, dilation 1 #r1"],
   ["residual block, dilation 2 #r2"], ["residual blocks, dilation 4, 8, … #r3"],
   ["sum of skip outputs → ReLU → 1×1 → ReLU → 1×1 #sk"], [r"softmax over 256 levels for $x_t$ #y"]],
  "x>cc; cc>r1; r1>r2; r2>r3; r1)>sk:skip; r2)>sk; r3>sk; sk>y",
  r"Each block: dilated causal conv → $\tanh(\cdot) \odot \sigma(\cdot)$ → 1×1 → residual and skip outputs. // Doubling dilations grow the receptive field exponentially with depth")
d("33", "Graph convolutional network (Kipf & Welling, 2016)",
  [["node features X, graph A #x"], [r"$H = \mathrm{ReLU}(\hat A X W_0)$ #h"], [r"$Z = \mathrm{softmax}(\hat A H W_1)$ #z"],
   ["cross-entropy on the few labelled nodes only #L !loss"]],
  None, r"$\hat A = \tilde D^{-1/2}(A + I)\tilde D^{-1/2}$: each layer averages over neighbours. // Too many layers over-smooth: every node ends up with the same features")
d("34", "Byte-pair encoding (Sennrich et al., 2016; byte-level: GPT-2, 2019)",
  [["training text as bytes (256 base tokens) #b"], ["count adjacent token pairs #c !gate"], ["merge the most frequent pair into a new token #m"],
   ["vocabulary + ordered list of merges #v !mem"], ["new text → bytes → apply the merges in order → token ids #e"]],
  "b>c; c>m; m(>c:repeat until the vocabulary is full; m>v; v>e",
  "Frequent words become single tokens; rare words split into pieces; nothing is out of vocabulary")
d("35", "RealNVP affine coupling (Dinh, Sohl-Dickstein & Bengio, 2016)",
  [["x #x"], [r"split into $x_a$, $x_b$ #sp !norm"], [r"$s, t = \mathrm{NN}(x_a)$ #nn", r"$x_a$ unchanged #xa !norm"],
   [r"$y_b = x_b \odot e^{s} + t$ #yb !gate"], [r"$y = (x_a, y_b)$; swap the halves and repeat #y"]],
  r"x>sp; sp>nn; sp>xa; nn>yb; sp(>yb:$x_b$; xa)>y; yb>y",
  r"Invertible, with a triangular Jacobian: $\log|\det J| = \sum s$, so the likelihood is exact")
d("36", "YOLO (Redmon et al., 2016), small version",
  [["image 64×64 #x"], ["3 × [conv 3×3 → BN → LeakyReLU → max-pool] (32, 64, 128 channels) #b"], ["2 × conv 3×3, 128 #c"],
   ["1×1 conv → 8×8 grid; each cell: objectness, x, y, √w, √h, 3 class scores #h"],
   ["score threshold → per-class non-maximum suppression #nms !gate"], ["boxes #y"]],
  None, "Detection as a single regression: every grid cell predicts a box in one forward pass")
d("37", "Transformer (Vaswani et al., 2017)",
  [["source embeddings + positions #se", "target embeddings + positions (shifted right) #te"],
   ["multi-head self-attention → add & norm #esa", "masked multi-head self-attention → add & norm #dsa"],
   ["feed-forward → add & norm #eff", "_ #s1"],
   ["encoder output (after N layers) #eo !mem", "cross-attention: Q from the decoder, K and V from the encoder → add & norm #ca"],
   ["_ #s2", "feed-forward → add & norm (N decoder layers) #dff"], ["linear → softmax: next target token #y"]],
  "se>esa; esa>eff; eff>eo; te>dsa; dsa>ca; eo>ca:K, V; ca>dff; dff>y",
  "Every sublayer has a residual connection followed by LayerNorm (post-norm, as in the paper)")
d("38", "Proximal Policy Optimization (Schulman et al., 2017)",
  [["state s #s"], [r"policy $\pi_\theta$ #pi", r"value $V_\phi$ #v"], [r"rollouts with $\pi_{old}$ #ro !mem"],
   [r"GAE advantages $\hat A_t$ #gae"], [r"ratio $r_t = \pi_\theta(a_t \mid s_t) / \pi_{old}(a_t \mid s_t)$ #r !gate"],
   [r"$L = \mathbb{E}[\min(r_t \hat A_t,\ \mathrm{clip}(r_t, 1 - \epsilon, 1 + \epsilon)\, \hat A_t)]$ + value loss #L !loss"]],
  r"s>pi; s>v; pi>ro; v)>gae:$V(s)$; ro>gae; gae>r; r>L; L.(>pi:several epochs on the same rollouts",
  "The clip stops the policy from moving far from the one that collected the data")
d("39", "VQ-VAE (van den Oord, Vinyals & Kavukcuoglu, 2017)",
  [["image 16×16 #x"], [r"conv encoder → grid of vectors $z_e$ #e"],
   [r"quantize: $z_q = e_k$, $k = \mathrm{argmin}_j \|z_e - e_j\|$ #q !gate", r"codebook $e_1 … e_K$ #cb !mem"],
   ["conv decoder #dd", "stage 2: Transformer prior over code indices (class-conditional) #p"], [r"reconstruction $\hat x$ #y", "sampled codes → decoder → new images #g"]],
  "x>e; e>q; cb>q; q>dd; q>p:code indices; dd>y; p>g",
  r"Straight-through gradient from $z_q$ to $z_e$; codebook and commitment losses train the codebook")
d("40", "AlphaGo Zero / AlphaZero (Silver et al., 2017)",
  [["board s (tic-tac-toe) #s"], ["policy–value network: MLP → policy p, value v #n"],
   ["MCTS: select by PUCT, expand with p, back up v #m !gate"], ["visit counts π: a stronger policy #pi !mem"],
   ["self-play: play moves sampled from π #g"], [r"train: $(z - v)^2 - \pi^\top \log p$ #L !loss"]],
  "s>n; n>m; m@:simulations; m>pi; pi>g; g>L:outcome z, targets π; L.(>n:update",
  "Search improves the network's policy, and the network makes the search stronger, all from self-play")

# ---------------------------------------------------------------- 2017–2022
d("41", "GPT (Radford et al., 2018): decoder-only Transformer",
  [[r"tokens $x_1 … x_t$ #x"], ["token + position embeddings #emb"], ["LayerNorm → masked multi-head self-attention #att"], ["+ #a1 !circle"],
   ["LayerNorm → MLP (GELU) #mlp"], ["+ #a2 !circle"], [r"final LayerNorm → linear → softmax: next token $x_{t+1}$ #y"]],
  "x>emb; emb>att; att>a1; emb)>a1; a1>mlp; mlp>a2; a1)>a2; a2>y:× N blocks",
  "A causal mask lets each position attend only to earlier ones. Pre-norm, as in GPT-2")
d("42", "BERT (Devlin et al., 2018)",
  [["[CLS] the [MASK] sat [SEP] … #x"], ["token + position embeddings #e"], ["Transformer encoder × N (bidirectional self-attention) #enc"],
   ["MLM head: predict the masked tokens #mlm !out", "[CLS] → classifier (fine-tuning) #cls !out"]],
  None, "Pre-train by filling in masked tokens (15%), then fine-tune the same encoder on a labelled task")
d("43", "Graph attention networks: GAT (2018) and GATv2 (2021)",
  [[r"node features $h_i$ and neighbours $j \in N(i)$ #x"], [r"$W h_i$, $W h_j$ #w"],
   [r"GAT: $e_{ij} = \mathrm{LeakyReLU}(a^\top [W h_i \| W h_j])$ // GATv2: $e_{ij} = a^\top \mathrm{LeakyReLU}(W [h_i \| h_j])$ #e !gate"],
   [r"$\alpha_{ij} = \mathrm{softmax}_j(e_{ij})$ over the neighbours #a !gate"], [r"$h_i' = \sigma(\sum_j \alpha_{ij} W h_j)$, several heads #h"]],
  None, "Unlike GCN's fixed averaging, each node learns how much to listen to each neighbour")
d("44", "Neural ordinary differential equations (Chen et al., 2018)",
  [[r"$h(t_0)$ #h0"], [r"ODE solver: $dh/dt = f_\theta(h, t)$, an MLP #s"], [r"$h(t_1)$ #h1"], ["loss L #L !loss"]],
  r"h0>s; s>h1; h1>L; L.(>s:adjoint: solve $da/dt = -a^\top \partial f / \partial h$ backwards",
  "Continuous depth; the adjoint gives gradients with memory independent of the number of solver steps")
d("45", "Scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022)",
  [["compute budget C ≈ 6 N D #c"], ["IsoFLOP: train models of many sizes N at this C #sw"],
   [r"final loss vs. N: a valley with its minimum at $N^*(C)$ #p !gate"],
   [r"fit power laws $N^* \propto C^a$, $D^* \propto C^b$ and $L(N, D) = E + A/N^\alpha + B/D^\beta$ #f"]],
  "c>sw; sw>p; p>f; f(>c:repeat for several C",
  "Chinchilla: grow parameters and training tokens in equal proportion (about 20 tokens per parameter)")
d("46", "GPT-3 and in-context learning (Brown et al., 2020)",
  [[r"prompt $x_1, y_1, x_2, y_2, …, x_k$ for a new random $w$, $y = w^\top x$ #p"], [r"tokens $[x, 0]$ and $[0, y]$ → linear embedding + positions #e"],
   ["causal Transformer, 4 layers #t"], [r"prediction $\hat y_i$ read out at each x token #y"]],
  None, "The weights are fixed at test time: the 'learning' happens inside the forward pass. // Compared with least squares on the same examples")
d("47", "Retrieval-augmented generation (Lewis et al., 2020; DPR: Karpukhin et al., 2020)",
  [["question q #q", "documents (an editable store) #dd !mem"], [r"question encoder $E_q$ #eq", r"document encoder $E_d$ #ed"],
   [r"top-k passages by $E_q(q) \cdot E_d(d)$ #k !gate"], ["reader: Transformer over [question; retrieved passages] #r"], ["answer #a"]],
  "q>eq; dd>ed; eq>k; ed>k; k>r; q(>r; r>a",
  "Knowledge lives in the store: edit a document and the answer changes, with no retraining")
d("48", "SimCLR (Chen et al., 2020)",
  [["image x #x"], ["random augmentation t #t1 !norm", "random augmentation t′ #t2 !norm"], ["encoder f #f1", "encoder f (same weights) #f2"],
   ["projection head g #g1", "projection head g #g2"], [r"$z_i$ #z1", r"$z_j$ #z2"],
   [r"NT-Xent: pull $z_i$, $z_j$ together, push the other images in the batch away #L !loss"]],
  "x>t1; x>t2; t1>f1; t2>f2; f1>g1; f2>g2; g1>z1; g2>z2; z1>L; z2>L", "After training, discard g and use f's features")
d("49", "Vision Transformer (Dosovitskiy et al., 2020)",
  [["image 16×16 → 16 patches of 4×4 #x"], ["linear patch embedding + [CLS] token + positions #e"], ["Transformer encoder × N #t"],
   ["[CLS] output → MLP head #h"], ["class #y"]],
  None, "No convolutions: an image is a sequence of patch tokens")
d("50", "Denoising diffusion probabilistic models (Ho, Jain & Abbeel, 2020)",
  [[r"clean sample $x_0$ #x0", r"$t \sim U\{1 … T\}$, $\epsilon \sim N(0, I)$ #n"],
   [r"$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$ #f !norm"], [r"noise predictor $\epsilon_\theta(x_t, t)$ #net"],
   [r"loss $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ #L !loss"]],
  r"x0>f; n>f; f>net; net>L; n)>L:$\epsilon$",
  r"Sampling runs the chain in reverse: start from $x_T \sim N(0, I)$ and denoise T times")
d("51", "CLIP (Radford et al., 2021)",
  [["images #i", "captions #c"], ["image encoder #ie", "text encoder #te"], [r"embeddings $I_1 … I_B$ #ib", r"$T_1 … T_B$ #tb"],
   [r"B × B similarities $I_i \cdot T_j / \tau$ #s !gate"], ["cross-entropy both ways: matching pairs on the diagonal #L !loss"]],
  "i>ie; c>te; ie>ib; te>tb; ib>s; tb>s; s>L", "Zero-shot: classify an image by its similarity to 'a photo of a {class}' prompts")
d("52", "LoRA (Hu et al., 2021)",
  [["x #x"], [r"frozen pretrained $W$ (d × k) #w !norm", "A (r × k), random init #a"], ["_ #sp", "B (d × r), init 0 #b"], ["+ #p !circle"],
   [r"$h = W x + (\alpha / r)\, B A x$ #h"]],
  "x>w; x>a; a>b; w>p; b>p; p>h", "Only A and B are trained (r ≪ d). After training BA merges into W: no extra inference cost")
d("53", "Masked autoencoders (He et al., 2021)",
  [["image → 16 patches #x"], ["mask a random 75% #m !gate"], ["ViT encoder on the 4 visible patches only #e"],
   ["insert mask tokens + positions #mt !norm"], ["small Transformer decoder #dd"], ["predict the pixels of the masked patches (MSE on those only) #y !loss"]],
  None, "The decoder is discarded after pre-training; the encoder then sees every patch")
d("54", "FlashAttention (Dao et al., 2022)",
  [["Q, K, V in GPU main memory (large, slow) #h !mem"], [r"load a block $Q_i$ into on-chip SRAM #q"],
   [r"for each block $K_j, V_j$: $S = Q_i K_j^\top$ in SRAM #s !gate"],
   ["online softmax: keep a running max and sum, // rescale the partial output O #o !gate"], [r"write $O_i$ to main memory once #w"]],
  "h>q; q>s; s>o; o(>s:next K, V block; o>w",
  "Exact attention without materializing the T × T matrix: O(T) memory and far fewer slow-memory reads")
d("55", "Flow matching / rectified flow (Lipman et al.; Liu et al., 2022)",
  [[r"noise $x_0 \sim N(0, I)$ #x0", r"data $x_1$ #x1"], [r"$x_t = (1 - t)\, x_0 + t\, x_1$,  $t \sim U[0, 1]$ #xt"],
   [r"velocity network $v_\theta(x_t, t)$ #v"], [r"loss $\|v_\theta(x_t, t) - (x_1 - x_0)\|^2$ #L !loss"]],
  None, "Sample by integrating dx/dt = v from noise to data. // Reflow: retrain on (noise, own sample) pairs, so paths straighten and few steps suffice")
d("56", "Latent diffusion and classifier-free guidance (Rombach et al.; Ho & Salimans, 2022)",
  [["image x #x"], ["autoencoder encoder (trained first, then frozen) #e !norm"], ["small latent z #z !mem"],
   [r"denoiser $\epsilon_\theta(z_t, t, c)$ in latent space #dd", r"class c, replaced by 'null' 10% of the time #c !in"],
   [r"guided: $\hat\epsilon = \epsilon_\theta(z_t, \emptyset) + w\, (\epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \emptyset))$ #g !gate"],
   ["autoencoder decoder → image #y"]],
  "x>e; e>z; z>dd; c>dd; dd>g; g>y", "Diffusion runs on the small latent; guidance strength w trades diversity for fidelity")
d("57", "RLHF / InstructGPT (Ouyang et al., 2022)",
  [["prompts #p"], [r"frozen reference $\pi_{ref}$ #ref !norm", r"policy $\pi_\theta$ #pi"], ["_ #sp", "sampled completions y #y !mem"],
   [r"KL penalty $\log \pi_\theta / \pi_{ref}$ #kl !gate", r"reward model $r_\phi(x, y)$, trained on preference pairs #rm"],
   [r"PPO maximizes $r_\phi(x, y) - \beta\, \mathrm{KL}(\pi_\theta \| \pi_{ref})$ #L !loss"]],
  "p>ref; p>pi; pi>y; y>rm; y>kl; ref>kl; kl>L; rm>L; L.)>pi:update",
  "Without the KL term the policy learns to game the reward model (shown in the notebook)")
d("58", "Direct preference optimization (Rafailov et al., 2023)",
  [[r"prompt x, preferred $y_w$, rejected $y_l$ #dd"], [r"policy $\pi_\theta$ #pi", r"frozen reference $\pi_{ref}$ #ref !norm"],
   [r"implicit rewards $\hat r(y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{ref}(y \mid x)}$ for $y_w$ and $y_l$ #r !gate"],
   [r"$L = -\log \sigma(\hat r(y_w) - \hat r(y_l))$ #L !loss"]],
  None, "The RLHF objective solved in closed form: no reward model, no sampling, no RL")

# ---------------------------------------------------------------- 2023–2025
d("59", "Mamba: selective state spaces (Gu & Dao, 2023)",
  [[r"$x_t$ #x"], [r"input-dependent $\Delta_t, B_t, C_t$ = Linear($x_t$) #sel !gate"], [r"$h_t = e^{\Delta_t A} h_{t-1} + \Delta_t B_t x_t$ #h !mem"],
   [r"$y_t = C_t h_t$ #y"]],
  r"x>sel; sel>h; x)>h; h@; h>y; sel(>y:$C_t$",
  "Selection makes the recurrence depend on the input; training uses a parallel scan. // Full block: in-projection → short conv → selective SSM, gated by a SiLU branch → out-projection")
d("60", "Speculative decoding (Leviathan et al.; Chen et al., 2023)",
  [["prefix #p"], ["small draft model q proposes γ tokens, one by one #dd"], ["target model p scores all γ + 1 positions in one pass #t"],
   ["accept each draft token x with probability min(1, p(x) / q(x)) #a !gate"],
   ["at the first rejection, resample from norm(max(0, p − q)); // if all are accepted, sample one more token from p #r"]],
  "p>dd; dd>t; t>a; a>r; r(>p:append and repeat", "The output distribution is exactly the target model's")
d("61", "Diffusion Transformer (Peebles & Xie, 2023)",
  [[r"noisy image $x_t$ → patches #x", "timestep t, class c #tc"], ["patch embedding + positions #e", r"MLP → scale, shift, gate (γ, β, α) per block #m !gate"],
   ["DiT block × N with adaLN-Zero #b"], ["linear → predicted noise for each patch #o"]],
  "x>e; tc>m; e>b; m>b; b>o",
  "adaLN-Zero: t and c set each block's LayerNorm scale and shift; the residual gate α starts at 0, // so every block starts as the identity. Sampled with classifier-free guidance")
d("62", "Quantization (LLM.int8, GPTQ and others, 2022–2023)",
  [["fp32 weights W #w"], ["scale per output channel (or group): s = max|W| / 127 #s"], [r"$W_q = \mathrm{round}(W / s)$: int8 or int4 #q !gate"],
   [r"$W \approx s \cdot W_q$ at matmul time #dd"]],
  None, "Outlier features: a few activation channels need higher precision (LLM.int8). // GPTQ: quantize one column at a time, correcting the rest with second-order information")
d("63", "Vision-language model, LLaVA-style (Liu et al., 2023)",
  [["image #img", "question tokens #t"], ["vision encoder (CNN) #ve", "token embeddings #te"], ["projector → 8 visual tokens #pr", "_ #sp"],
   ["[visual tokens; text tokens] #seq !mem"], ["language model (causal Transformer) #lm"], ["answer #a"]],
  "img>ve; ve>pr; pr>seq; t>te; te>seq; seq>lm; lm>a", "The image enters the language model as a few extra tokens in its input")
d("64", "Mixture of experts (Shazeer et al., 2017; Switch, 2021; DeepSeek-V3, 2024)",
  [["token x #x"], [r"router: $g = \mathrm{softmax}(W_r x)$, keep the top k #r !gate"],
   ["expert 1 (MLP) #e1", "expert 2 #e2", "expert 3 #e3", "… #dots !note", "expert N #en"], [r"$y = \sum_{i \in \mathrm{top}k} g_i\, E_i(x)$ #y"]],
  "x>r; r>e1; r.>e2; r>e3; r.>en; e1>y; e3>y",
  "Only k of N experts run per token (k = 2 shown). Load balancing: an auxiliary loss (Switch) // or a per-expert bias on the router scores (DeepSeek-V3)")
d("65", "Sparse autoencoders for interpretability (2023–2024)",
  [[r"model activations $x \in \mathbb{R}^d$ #x"], [r"encoder: $f = \mathrm{TopK}(W_e (x - b_d) + b_e)$, with m ≫ d latents #f !gate"],
   [r"decoder: $\hat x = W_d f + b_d$ #dd"], [r"loss $\|x - \hat x\|^2$ (TopK enforces sparsity) #L !loss"]],
  None, "Each latent should be one interpretable feature that the model stores in superposition")
d("66", "Kolmogorov–Arnold network (Liu et al., 2024)",
  [[r"$x_1$ #x1", r"$x_2$ #x2"], [r"$\phi_{1,1}$ #a !gate", r"$\phi_{1,2}$ #b !gate", r"$\phi_{2,1}$ #c !gate", r"$\phi_{2,2}$ #dd !gate"],
   [r"$\Sigma$ #s1 !circle", r"$\Sigma$ #s2 !circle"], ["next KAN layer → output #y"]],
  "x1>a; x1>c; x2>b; x2>dd; a>s1; b>s1; c>s2; dd>s2; s1>y; s2>y",
  r"MLP: learned weights on edges, fixed activations on nodes. KAN: a learned spline $\phi$ on every edge, plain sums on nodes")
d("67", "Multi-head latent attention (DeepSeek-V2, 2024)",
  [[r"hidden state $h_t$ #h"], [r"down-project: $c_t = W^{DKV} h_t$ (small, cached) #c !mem", r"queries $W^Q h_t$ (+ RoPE part) #q"],
   [r"up-project per head: $k = W^{UK} c_t$, $v = W^{UV} c_t$ #up", r"decoupled RoPE key $k^R_t$ (shared, cached) #kr !mem"],
   ["attention per head #att"], [r"$W^O$ → output #o"]],
  "h>c; h>q; h>kr; c>up; up>att; kr>att; q)>att; att>o",
  r"The cache holds only $c_t$ and $k^R_t$. At inference $W^{UK}$ is absorbed into the query projection")
d("68", "Multi-token prediction (Gloeckle et al., 2024; DeepSeek-V3)",
  [[r"tokens $t_1 … t_n$ #x"], ["main Transformer #m"],
   [r"head → predict $t_{i+1}$ #h1", r"MTP module: concat(RMSNorm(h), RMSNorm(emb($t_{i+1}$))) → linear → block #mtp"],
   [r"loss on $t_{i+1}$ #l1 !loss", r"shared head → predict $t_{i+2}$ #h2"], ["_ #sp", r"loss on $t_{i+2}$ (weight λ) #l2 !loss"]],
  "x>m; m>h1; m>mtp; h1>l1; mtp>h2; h2>l2",
  "A denser training signal. At inference the MTP module drafts the token after next (self-speculative decoding)")
d("69", "GRPO and RL with verifiable rewards (DeepSeekMath, 2024; DeepSeek-R1, 2025)",
  [["prompt: a problem with a checkable answer #p"], ["policy samples a group of G answers #g"], [r"verifier: reward $r_i = 1$ if correct, else 0 #v !gate"],
   [r"group-relative advantage $A_i = (r_i - \mathrm{mean}(r)) / \mathrm{std}(r)$ #a"],
   [r"clipped policy-gradient update $- \beta\, \mathrm{KL}(\pi_\theta \| \pi_{ref})$ #L !loss"]],
  "p>g; g>v; v>a; a>L; L.(>g:update the policy", "The group mean replaces PPO's learned value network as the baseline")
d("70", "Gated DeltaNet (Yang, Kautz & Hatamizadeh, 2025)",
  [[r"$x_t$ #x"], [r"$q_t, k_t, v_t$ (short conv; L2-normalized q, k) #qkv", r"decay $\alpha_t$, write strength $\beta_t$ #g !gate"],
   [r"$S_t = \alpha_t (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$ #S !mem"], [r"$o_t = S_t^\top q_t$ → norm → output gate #o"]],
  r"x>qkv; x>g; qkv>S; g>S; S@; S>o; qkv(>o:$q_t$",
  "A fixed-size memory edited by the delta rule; training uses a chunkwise parallel algorithm")
d("71", "Gated attention (Qwen, 2025)",
  [["x #x"], ["softmax attention (per head) #att", r"gate $\sigma(x W_g)$ #g !gate"], [r"$\odot$ #m !circle"], [r"$W_O$ → output #o"]],
  None, "A sigmoid gate on each head's output adds non-linearity and sparsity and removes attention sinks")
d("72", "Tiny Recursive Model (Jolicoeur-Martineau, 2025)",
  [["answer y (latent) #y !mem", "puzzle x #x", "reasoning z (latent) #z !mem"], [r"n times: $z \leftarrow f(x + y + z)$ #zz"],
   [r"then: $y \leftarrow f(y + z)$ #yy"], ["output head → answer #o"]],
  "x>zz; y>zz; z>zz; zz@:n; zz>yy; y(>yy; yy>o; yy.(>y:repeat T times, then for each supervision step",
  "One tiny 2-layer network f does every step. Only the last recursion is backpropagated; // deep supervision carries (y, z), detached, into the next step")
d("73", "Masked diffusion language model (LLaDA, Nie et al., 2025)",
  [[r"text $x_0$ #x"], ["mask each token with probability t, t ~ U(0, 1) #m !gate"], ["bidirectional Transformer (no causal mask) #t"],
   ["predict every masked token; loss = cross-entropy on masked tokens / t #L !loss"]],
  None, "Sampling starts fully masked and unmasks the most confident tokens step by step, // so the model can fill in text anywhere, not only at the end")

# ---------------------------------------------------------------- 2026
d("74", "The modern LLM block (2023–2026)",
  [["x #x"], ["RMSNorm #n1 !norm"], ["grouped-query attention, QK-norm, RoPE on q and k #att"], ["+ #a1 !circle"], ["RMSNorm #n2 !norm"],
   [r"SwiGLU: $W_2(\mathrm{SiLU}(W_1 x) \odot W_3 x)$ #ffn"], ["+ #a2 !circle"], ["output #o"]],
  "x>n1; n1>att; att>a1; x)>a1; a1>n2; n2>ffn; ffn>a2; a1)>a2; a2>o",
  "vs. GPT-2: RMSNorm for LayerNorm, RoPE for learned positions, GQA, QK-norm, SwiGLU for GELU, no biases. // Hidden matrices trained with Muon")
d("75", "Learned sparse attention (DeepSeek-V3.2, 2025) and hybrid stacks",
  [["tokens #x"], [r"lightning indexer: $I_{t,s} = \sum_j w_{t,j}\, \mathrm{ReLU}(q^I_{t,j} \cdot k^I_s)$ #i !gate"], ["top-k keys for each query #k !gate"],
   ["main attention over the k selected keys only #a"], ["output #o"]],
  "x>i; i>k; k>a; x)>a:keys, values; a>o",
  "The indexer is first trained to match dense attention (KL), then everything trains sparse. // Hybrid stacks: most layers use a sliding window, a few attend globally")
d("76", "mHC: manifold-constrained hyper-connections (DeepSeek, 2026)",
  [[r"n = 4 residual streams $X_l$ #X !mem"], [r"$H^{pre}$: weighted read of the streams #pre !gate", r"$H^{res}$: n × n mix, doubly stochastic (Sinkhorn–Knopp) #res !gate"],
   ["layer F (attention or MLP) #F", "_ #sp"], [r"$H^{post}$: write the output to the streams #post !gate"],
   [r"$X_{l+1} = H^{res} X_l + H^{post\top} F(H^{pre} X_l)$ #o !mem"]],
  "X>pre; X>res; pre>F; F>post; post>o; res)>o",
  "The H matrices are computed from the input. Doubly stochastic mixing keeps the gain near 1 at any depth")
d("77", "Engram: conditional memory (DeepSeek, 2026)",
  [["token ids #t"], ["canonical ids → suffix 2-grams and 3-grams #g"],
   [r"K hash heads → embedding tables → concatenated $e_t$ #e !mem", r"hidden state $h_t$ #h !in"],
   [r"gate $\alpha_t = \sigma(\mathrm{RMSNorm}(h_t)^\top \mathrm{RMSNorm}(W_K e_t) / \sqrt{d})$ #a !gate"],
   [r"$\tilde v_t = \alpha_t W_V e_t$ → causal depthwise conv (SiLU) + residual #v"], ["added to the residual stream in early layers #o"]],
  "t>g; g>e; e>a; h>a; a>v; e(>v:$e_t$; v>o",
  "Static local patterns are looked up instead of recomputed: O(1) per token, and the tables can live in host memory")
d("78", "DeepSeek-V4: compressed sparse attention (CSA) and heavily compressed attention (HCA)",
  [["key/value entries for every token #kv !mem"], ["CSA: pool every 4 tokens (overlapping, per-channel softmax weights) #csa", "HCA: pool every 128 tokens #hca"],
   ["lightning indexer → top-k compressed blocks #idx !gate", "_ #sp"],
   ["CSA layer: attend to the selected blocks + sliding window + sink #a1", "HCA layer: attend to all blocks + sliding window + sink #a2"], ["output #o"]],
  "kv>csa; kv>hca; csa>idx; idx>a1; hca>a2; a1>o; a2>o",
  "Layers alternate between CSA and HCA. Compression shrinks the KV cache; selection shrinks the attention compute")
d("79", "Early-2026 capstone: a Qwen3.5 / DeepSeek-V4-style hybrid",
  [["tokens #x"], ["embeddings #e"], ["Gated DeltaNet → MoE #g1"], ["Gated DeltaNet → MoE #g2"], ["Gated DeltaNet → MoE #g3"],
   ["gated attention (RoPE, QK-norm, GQA) → MoE #ga"], ["head: next token  +  MTP module: the token after next #h"]],
  "x>e; e>g1; g1>g2; g2>g3; g3>ga; ga>h:× L/4",
  "MoE: fine-grained experts + a shared expert, bias-based load balancing. // Pre-norm RMSNorm, SwiGLU experts; Muon for hidden matrices, AdamW for the rest")
d("80", "Kimi Delta Attention (Moonshot AI, 2025–2026)",
  [[r"$x_t$ #x"], ["q, k, v: short conv → Swish (L2-normalized q, k) #qkv", r"$\beta_t = \sigma(W_\beta x_t)$ #b !gate", r"$\alpha_t \in [0, 1]^{d_k}$, one per channel (low-rank) #a !gate"],
   [r"$S_t = (I - \beta_t k_t k_t^\top)\, \mathrm{Diag}(\alpha_t)\, S_{t-1} + \beta_t k_t v_t^\top$ #S !mem"], [r"$o_t = S_t^\top q_t$ → RMSNorm → output gate #o"]],
  "x>qkv; x>b; x>a; qkv>S; b>S; a>S; S@; S>o",
  "vs. Gated DeltaNet: a forget rate per channel instead of one per head. // Kimi K3: 69 KDA layers + 24 global attention layers, no RoPE")
d("81", "Mamba-3 (Lahoti et al., 2026)",
  [[r"$x_t$ #x"], [r"$\Delta_t, B_t, C_t, \lambda_t, \theta_t$ = Linear($x_t$) #p !gate"],
   [r"rotate $B_t, C_t$ by $-\phi_t$,  $\phi_t = \sum_{s \leq t} \Delta_s \theta_s$  (complex state as data-dependent RoPE) #r"],
   [r"$h_t = \alpha_t h_{t-1} + \beta_t B_{t-1} x_{t-1} + \gamma_t B_t x_t$  (exponential-trapezoidal) #h !mem"], [r"$y_t = C_t^\top h_t$ #y"]],
  "x>p; p>r; r>h; x)>h; h@; h>y",
  "λ = 1 gives back Mamba-2. The rotation lets the state track parity, which real decays cannot. // MIMO: rank-r inputs and outputs per step")
d("82", "Attention residuals (Kimi, 2026)",
  [[r"$v_0$ = embedding #v0", r"$v_1 = f_1(h_1)$ #v1", r"$v_2 = f_2(h_2)$ #v2", "… #dots !note", r"$v_{l-1}$ #vl"],
   [r"pseudo-query $w_l$ (initialized to 0) against $\mathrm{RMSNorm}(v_i)$ #k !gate"], [r"$\alpha_{i \to l} = \mathrm{softmax}_i(w_l^\top \mathrm{RMSNorm}(v_i))$ #a !gate"],
   [r"$h_l = \sum_i \alpha_{i \to l}\, v_i$ → layer $f_l$ #h"]],
  "v0>k; v1>k; v2>k; vl>k; k>a; a>h",
  r"Replaces the fixed sum $h_l = h_1 + \sum_{i<l} f_i(h_i)$ with softmax attention over depth. // Block AttnRes attends over about 8 block summaries instead of every layer")
d("83", "LatentMoE (NVIDIA Nemotron 3, 2026)",
  [[r"token $x \in \mathbb{R}^d$ #x"], [r"router on x (full width d) → top-K′ #r !gate", r"$W_\downarrow$: d → ℓ #down"],
   [r"N′ = N·d/ℓ experts, each working in $\mathbb{R}^\ell$ #e"], [r"$W_\uparrow$: ℓ → d #up", "shared expert (width d) #sh"], ["sum → y #y"]],
  "x>r; x>down; down>e; r>e:gates; e>up; up>y; x)>sh; sh>y",
  "Only ℓ-dimensional vectors cross devices. Here ℓ = d/4: 4× the experts and 4× top-k at the same communication")
d("84", "Gated DeltaNet-2 (NVIDIA, 2026)",
  [[r"$x_t$ #x"], ["q, k, v #qkv", r"erase gate $b_t \in [0, 1]^{d_k}$ #b !gate", r"write gate $w_t \in [0, 1]^{d_v}$ #w !gate", r"decay $\alpha_t$, per channel #a !gate"],
   [r"$S_t = (I - k_t (b_t \odot k_t)^\top)\, \mathrm{Diag}(\alpha_t)\, S_{t-1} + k_t (w_t \odot v_t)^\top$ #S !mem"], [r"$o_t = S_t^\top q_t$ #o"]],
  "x>qkv; x>b; x>w; x>a; qkv>S; b>S; w>S; a>S; S@; S>o",
  r"$b_t = w_t = \beta_t \mathbf{1}$ gives KDA; a scalar $\alpha_t$ as well gives Gated DeltaNet")
d("85", "Gated Residual (Qwen3.8-Next, 2026)",
  [[r"$n_r = 4$ residual branches $R_1 … R_4$ #R !mem"], [r"per-branch RMSNorm (own gain $\gamma_i$) → $\hat R_i$ #n !norm"],
   [r"read gate $G = \sigma(W_u\, \mathrm{SiLU}(W_d\, \mathrm{vec}(\hat R) / n_r))$ #G !gate", r"write gate $s = 2\sigma(W_w\, \mathrm{vec}(\hat R) / n_r) \in \mathbb{R}^{n_r}$ #s !gate"],
   [r"$x = \frac{1}{n_r} \sum_i G_i \odot \hat R_i$ #xx"], ["layer y = F(x), no separate pre-norm #F"], [r"$R_i' = R_i + s_i\, y$ #o !mem"]],
  r"R>n; n>G; n>s; G>xx; n(>xx:$\hat R_i$; xx>F; F>o; s)>o; R(>o:residual",
  "The read gate is per branch and per channel, through a low-rank bottleneck (r = d/8); // the write gate is one scalar per branch")
d("86", "Cheaper sparse-attention indexers (Qwen3.8 QSA; DeepSeek-V4.1 CSA2, 2026)",
  [[r"indexer keys $k_j$ for every past token #k"], [r"average-pool blocks of r = 4 → RMSNorm → $\hat k_b$ #pool !norm"],
   [r"score blocks: $I_{ib} = \sum_h \mathrm{ReLU}(\langle q^h_i, \hat k_b \rangle)$ #s !gate"], ["top ⌈K/r⌉ blocks + the current block → token set #t !gate"],
   ["sparse attention in this layer #a", "following layers reuse these indices // (or re-index) #r"]],
  "k>pool; pool>s; s>t; t>a; t>r",
  "Indexer cost falls from O(T²) to O(T²/r) per layer, and to nothing in layers that reuse the indices")
d("87", "Causal encoder–decoder (DeepSeek-V4.1-Flash, 2026)",
  [["tokens #x"], ["causal encoder: layers 1 … L/2 #enc"], [r"$H_{L/2}$ for every token (cached) #H !mem"],
   [r"global K, V: $C_l = H_{L/2} W^{KV}_l$ #kv", "local sliding window over the decoder's own states #loc !mem"],
   ["decoder: layers L/2 + 1 … L #dec"], ["next token #y"]],
  "x>enc; enc>H; H>kv; kv>dec; loc>dec; H(>dec:current token; dec>y",
  "Prefill runs only the encoder; the last 128 prompt tokens are replayed through the decoder to fill its local cache")
d("88", "Looped transformer (Universal Transformer, 2018 → 2026 models)",
  [["tokens → embedding e #e"], ["+ e (input injection from the second pass on) #add !gate"], [r"shared block(s), weights $\theta$ #b"], ["after R passes: head #y"]],
  "e>add; add>b; b(>add:R passes; b>y",
  "Parameters shrink R× but FLOPs and KV cache do not. More passes at test time buy more compute")
d("89", "Constant learning rate + checkpoint averaging (2026)",
  [["the same model and data #r"], ["cosine decay #c !norm", "WSD: warm-up, stable, short decay #w !norm", "constant LR #k !norm"],
   ["final weights #cf", "final weights #wf", "average of recent checkpoints (EMA or window) #a !gate"], ["validation loss, compared #v"]],
  "r>c; r>w; r>k; c>cf; w>wf; k>a; cf>v; wf>v; a>v",
  "Averaging cancels the noise a high learning rate leaves, without fixing the run length in advance")
d("90", "Late-2026 capstone",
  [["tokens #x"], ["embeddings #e"], ["Gated DeltaNet-2 → LatentMoE #g1"], ["Gated DeltaNet-2 → LatentMoE #g2"], ["Gated DeltaNet-2 → LatentMoE #g3"],
   ["gated sparse attention with a block-compressed indexer → LatentMoE #ga"], ["head: next token  +  MTP module: the token after next #h"]],
  "x>e; e>g1; g1>g2; g2>g3; g3>ga; ga>h:× L/4",
  "Residual path: Block Attention Residuals over depth. LatentMoE with a shared expert and bias-based balancing. // Muon + AdamW, constant learning rate + EMA of the weights")


def stem(num):
    return next((ROOT / "nb_src").glob(f"{num}_*.py")).stem

if __name__ == "__main__":
    out = ROOT / "assets" / "diagrams"; out.mkdir(parents=True, exist_ok=True)
    for num in sys.argv[1:] or sorted(D):
        title, rows, edges, caption = D[num]
        render(rows, edges, f"{num} · {title}", caption, out / f"{stem(num)}.png")
        print("wrote", f"assets/diagrams/{stem(num)}.png")
