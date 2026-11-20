# Neural Networks, 1943 → 2026

One runnable Jupyter notebook per milestone in neural-network history, from the McCulloch–Pitts neuron to the 2025–2026 frontier. Each notebook explains the idea (paper, key equations, why it mattered), implements it **from scratch**, trains it on a small task, and shows the result.

- **Runs anywhere:** every notebook runs on a laptop CPU. A GPU is optional and used automatically when present: CUDA, then Apple Silicon (MPS), then CPU.
- **Offline:** no dataset downloads. Notebooks use scikit-learn's bundled 8×8 digits, synthetic data, or small text corpora embedded in the notebook.
- **Self-contained:** each notebook stands alone; just open it and run all cells.

## Timeline

| # | Year | Notebook | Idea | Task |
|---|---|---|---|---|
| 01 | 1943 | [McCulloch–Pitts neuron](notebooks/01_1943_mcculloch_pitts_neuron.ipynb) | threshold logic units | logic gates, XOR network, latch |
| 02 | 1958 | [Perceptron](notebooks/02_1958_perceptron.ipynb) | first learning rule, convergence theorem | 2-D blobs, iris |
| 03 | 1960 | [ADALINE / LMS](notebooks/03_1960_adaline.ipynb) | gradient descent on squared error | overlapping classes |
| 04 | 1969 | [The XOR problem](notebooks/04_1969_xor_problem.ipynb) | limits of single layers (Minsky & Papert) | XOR, feature lifting |
| 05 | 1982 | [Hopfield network](notebooks/05_1982_hopfield_network.ipynb) | energy-based associative memory | pattern recall, capacity |
| 06 | 1982 | [Self-organizing map](notebooks/06_1982_self_organizing_map.ipynb) | competitive, topology-preserving learning | colours, digits |
| 07 | 1985 | [Boltzmann machine / RBM](notebooks/07_1985_boltzmann_machine_rbm.ipynb) | stochastic generative nets, contrastive divergence | binarised digits |
| 08 | 1986 | [Backpropagation](notebooks/08_1986_backpropagation_mlp.ipynb) | MLP + chain rule in NumPy, gradient check | XOR, two moons |
| 09 | 1989 | [LeNet CNN](notebooks/09_1989_lenet_cnn.ipynb) | convolution, weight sharing, pooling | digits |
| 10 | 1990 | [Elman RNN](notebooks/10_1990_elman_rnn.ipynb) | recurrence, BPTT | char-level LM, word boundaries |
| 11 | 1997 | [LSTM](notebooks/11_1997_lstm.ipynb) | gated cell state vs vanishing gradients | adding problem |
| 12 | 2006 | [Deep autoencoder](notebooks/12_2006_deep_autoencoder.ipynb) | greedy layer-wise pre-training | 2-D codes vs PCA |
| 13 | 2012 | [AlexNet](notebooks/13_2012_alexnet.ipynb) | ReLU, dropout, augmentation, GPUs | digits, ablations |
| 14 | 2013 | [word2vec](notebooks/14_2013_word2vec.ipynb) | skip-gram with negative sampling | synthetic corpus |
| 15 | 2013 | [VAE](notebooks/15_2013_vae.ipynb) | ELBO, reparameterisation trick | digit generation |
| 16 | 2014 | [GAN](notebooks/16_2014_gan.ipynb) | adversarial training, mode collapse | ring of Gaussians |
| 17 | 2014 | [GRU + seq2seq](notebooks/17_2014_gru_seq2seq.ipynb) | encoder–decoder, teacher forcing | sorting digits |
| 18 | 2014 | [Bahdanau attention](notebooks/18_2014_bahdanau_attention.ipynb) | additive attention, alignments | date normalisation |
| 19 | 2015 | [Batch normalization](notebooks/19_2015_batchnorm.ipynb) | normalising activations | 10-layer sigmoid MLP |
| 20 | 2015 | [ResNet](notebooks/20_2015_resnet.ipynb) | residual connections, degradation problem | plain vs residual, 6–42 layers |
| 21 | 2015 | [U-Net](notebooks/21_2015_unet.ipynb) | encoder–decoder with skips for segmentation | synthetic shapes |
| 22 | 2016 | [Graph convolutional network](notebooks/22_2016_graph_convolutional_network.ipynb) | message passing, over-smoothing | semi-supervised node classification |
| 23 | 2017 | [Transformer](notebooks/23_2017_transformer.ipynb) | multi-head attention, encoder–decoder | reverse + increment |
| 24 | 2018 | [GPT](notebooks/24_2018_gpt.ipynb) | decoder-only next-token prediction | char-level LM |
| 25 | 2018 | [BERT](notebooks/25_2018_bert.ipynb) | masked LM pre-training + fine-tuning | synthetic sentiment |
| 26 | 2020 | [Vision Transformer](notebooks/26_2020_vision_transformer.ipynb) | patches as tokens | digits |
| 27 | 2020 | [DDPM diffusion](notebooks/27_2020_ddpm_diffusion.ipynb) | denoising diffusion | Swiss roll, digits |
| 28 | 2021 | [CLIP](notebooks/28_2021_clip.ipynb) | contrastive image–text learning | zero-shot digit classification |
| 29 | 2021 | [LoRA](notebooks/29_2021_lora.ipynb) | low-rank adaptation, weight merging | domain-shifted digits |
| 30 | 2022 | [Flow matching](notebooks/30_2022_flow_matching.ipynb) | velocity regression, rectified flow / reflow | checkerboard |
| 31 | 2023 | [Mamba](notebooks/31_2023_mamba_selective_ssm.ipynb) | selective state spaces, parallel scan | selective copying |
| 32 | 2017→2024 | [Mixture of Experts](notebooks/32_2023_mixture_of_experts.ipynb) | top-k routing, load balancing | expert specialisation |
| 33 | 2024 | [KAN](notebooks/33_2024_kan.ipynb) | learnable spline activations on edges | symbolic regression |
| 34 | 2024 | [Multi-head Latent Attention](notebooks/34_2024_multi_head_latent_attention.ipynb) | compressed KV cache, decoupled RoPE, absorption | tiny LM + cache checks |
| 35 | 2025 | [GRPO](notebooks/35_2025_grpo_reasoning_rl.ipynb) | RL with verifiable rewards, group baselines | digit-sum puzzles |
| 36 | 2025 | [Gated DeltaNet](notebooks/36_2025_gated_deltanet.ipynb) | delta-rule linear attention with gating | recall with overwrites |
| 37 | 2025 | [Tiny Recursive Model](notebooks/37_2025_tiny_recursive_model.ipynb) | recursion + deep supervision | 4×4 Sudoku |
| 38 | 2026 | [Modern LLM recipe](notebooks/38_2026_modern_llm_recipe.ipynb) | RMSNorm, RoPE, GQA, QK-norm, SwiGLU, Muon | GPT-2 block vs modern block |
| 39 | 2026 | [Sparse & hybrid attention](notebooks/39_2026_sparse_and_hybrid_attention.ipynb) | DeepSeek Sparse Attention (lightning indexer, top-k), local/global hybrids | long-context retrieval |

Notebooks 01–08 use pure NumPy. From 09 onward they use PyTorch.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # or: uv venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

### Choosing the device
Each PyTorch notebook picks `cuda` → `mps` → `cpu` automatically. To force one:

```bash
NN_DEVICE=cpu jupyter lab notebooks/
```

Run times measured on an Apple M1 Pro: with the GPU (MPS), each notebook takes a few seconds to ~8 minutes, about 45 minutes for all 39. CPU-only is similar for most notebooks. The exception is `31_2023_mamba_selective_ssm`, whose hand-written selective scan takes ~17 minutes on CPU. The other slower ones (2–8 min) are the Gated DeltaNet, Tiny Recursive Model, LLM-recipe and sparse-attention notebooks.

## Repository layout

```
notebooks/      the notebooks (committed with outputs so they can be read on GitHub)
nb_src/         source of each notebook in a light cell-marker format ("#%% md" / "#%%")
tools/
  build.py        nb_src/*.py  ->  notebooks/*.ipynb (expands shared snippets)
  snippets/       shared code: device setup, digits loader, small text corpus
  run_all.sh      rebuild and execute every notebook in place
  show_outputs.py print the text outputs of executed notebooks
```

To edit a notebook, either change the `.ipynb` directly, or edit `nb_src/<name>.py` and run `python tools/build.py <name>` (this overwrites the `.ipynb`). `tools/run_all.sh` rebuilds everything and re-executes it.

## Notes

- These are **teaching-scale** implementations: tiny models, small data, short training runs. They show the mechanism, not state-of-the-art numbers. Where a toy result does not show the effect seen at scale, the notebook says so.
- The 2025–2026 entries cover ideas that were established in published work and production models by 2026: GRPO/RLVR, Gated DeltaNet hybrids, recursive reasoning models, the modern dense-block recipe with the Muon optimizer, and learned sparse attention with hybrid local/global stacks (the long-context approach of the 2026 open-weight frontier: DeepSeek-V4, GLM-5, Kimi, MiMo, ...).
