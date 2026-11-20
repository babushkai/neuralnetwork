# Neural Networks, 1943 → 2026

One runnable Jupyter notebook per milestone in neural-network history, from the McCulloch–Pitts neuron to the designs inside 2026 frontier models (DeepSeek-V4, Qwen3.5). Each notebook explains the idea (paper, key equations, why it mattered), implements it **from scratch**, trains it on a small task, and shows the result.

- **Runs anywhere:** every notebook runs on a laptop CPU. A GPU is optional and used automatically when present: CUDA, then Apple Silicon (MPS), then CPU.
- **Offline:** no dataset downloads. Notebooks use scikit-learn's bundled 8×8 digits, synthetic data, or small text corpora generated inside the notebook.
- **Self-contained:** each notebook stands alone; just open it and run all cells.

## Timeline

### Foundations (1943–1997)
| # | Year | Notebook | Idea |
|---|---|---|---|
| 01 | 1943 | [McCulloch–Pitts neuron](notebooks/01_1943_mcculloch_pitts_neuron.ipynb) | threshold logic units |
| 02 | 1958 | [Perceptron](notebooks/02_1958_perceptron.ipynb) | the first learning rule, convergence theorem |
| 03 | 1960 | [ADALINE / LMS](notebooks/03_1960_adaline.ipynb) | gradient descent on squared error |
| 04 | 1969 | [The XOR problem](notebooks/04_1969_xor_problem.ipynb) | limits of single layers |
| 05 | 1982 | [Hopfield network](notebooks/05_1982_hopfield_network.ipynb) | energy-based associative memory |
| 06 | 1982 | [Self-organizing map](notebooks/06_1982_self_organizing_map.ipynb) | competitive, topology-preserving learning |
| 07 | 1985 | [Boltzmann machine / RBM](notebooks/07_1985_boltzmann_machine_rbm.ipynb) | stochastic generative nets, contrastive divergence |
| 08 | 1986 | [Backpropagation](notebooks/08_1986_backpropagation_mlp.ipynb) | the MLP and the chain rule, in NumPy |
| 09 | 1989 | [LeNet CNN](notebooks/09_1989_lenet_cnn.ipynb) | convolution, weight sharing, pooling |
| 10 | 1990 | [Elman RNN](notebooks/10_1990_elman_rnn.ipynb) | recurrence, BPTT |
| 11 | 1997 | [LSTM](notebooks/11_1997_lstm.ipynb) | gated memory vs vanishing gradients |

### The deep learning revival (2006–2016)
| # | Year | Notebook | Idea |
|---|---|---|---|
| 12 | 2006 | [Deep autoencoder](notebooks/12_2006_deep_autoencoder.ipynb) | greedy layer-wise pre-training |
| 13 | 2012 | [AlexNet](notebooks/13_2012_alexnet.ipynb) | ReLU, dropout, augmentation, GPUs |
| 14 | 2013 | [word2vec](notebooks/14_2013_word2vec.ipynb) | skip-gram with negative sampling |
| 15 | 2013 | [VAE](notebooks/15_2013_vae.ipynb) | ELBO, reparameterisation trick |
| 16 | 2013 | [Deep Q-Network](notebooks/16_2013_deep_q_network.ipynb) | deep RL: experience replay, target networks (CartPole) |
| 17 | 2014 | [GAN](notebooks/17_2014_gan.ipynb) | adversarial training, mode collapse |
| 18 | 2014 | [GRU + seq2seq](notebooks/18_2014_gru_seq2seq.ipynb) | encoder–decoder, the fixed-vector bottleneck |
| 19 | 2014 | [Bahdanau attention](notebooks/19_2014_bahdanau_attention.ipynb) | additive attention, alignments |
| 20 | 2014 | [Adam & optimizers](notebooks/20_2014_adam_optimizers.ipynb) | SGD → momentum → RMSProp → Adam → AdamW |
| 21 | 2015 | [Batch normalization](notebooks/21_2015_batchnorm.ipynb) | normalising activations |
| 22 | 2015 | [ResNet](notebooks/22_2015_resnet.ipynb) | residual connections, the degradation problem |
| 23 | 2015 | [U-Net](notebooks/23_2015_unet.ipynb) | encoder–decoder with skips, segmentation |
| 24 | 2015 | [Knowledge distillation](notebooks/24_2015_knowledge_distillation.ipynb) | dark knowledge, teacher → student |
| 25 | 2016 | [WaveNet](notebooks/25_2016_wavenet.ipynb) | dilated causal convolutions, raw audio |
| 26 | 2016 | [Graph convolutional network](notebooks/26_2016_graph_convolutional_network.ipynb) | message passing, over-smoothing |

### Transformers and scale (2017–2022)
| # | Year | Notebook | Idea |
|---|---|---|---|
| 27 | 2017 | [Transformer](notebooks/27_2017_transformer.ipynb) | multi-head attention, encoder–decoder |
| 28 | 2018 | [GPT](notebooks/28_2018_gpt.ipynb) | decoder-only next-token prediction |
| 29 | 2018 | [BERT](notebooks/29_2018_bert.ipynb) | masked LM pre-training + fine-tuning |
| 30 | 2020 | [Scaling laws](notebooks/30_2020_scaling_laws.ipynb) | power laws, Chinchilla IsoFLOP curves |
| 31 | 2020 | [SimCLR](notebooks/31_2020_simclr_self_supervised.ipynb) | self-supervised contrastive learning |
| 32 | 2020 | [Vision Transformer](notebooks/32_2020_vision_transformer.ipynb) | image patches as tokens |
| 33 | 2020 | [DDPM diffusion](notebooks/33_2020_ddpm_diffusion.ipynb) | denoising diffusion |
| 34 | 2021 | [CLIP](notebooks/34_2021_clip.ipynb) | contrastive image–text, zero-shot |
| 35 | 2021 | [LoRA](notebooks/35_2021_lora.ipynb) | low-rank adaptation |
| 36 | 2022 | [FlashAttention](notebooks/36_2022_flash_attention.ipynb) | tiled exact attention, online softmax |
| 37 | 2022 | [Flow matching](notebooks/37_2022_flow_matching.ipynb) | rectified flow, reflow |

### The LLM era (2023–2025)
| # | Year | Notebook | Idea |
|---|---|---|---|
| 38 | 2023 | [RLHF → DPO](notebooks/38_2023_dpo_preference_optimization.ipynb) | preference optimisation without RL |
| 39 | 2023 | [Mamba](notebooks/39_2023_mamba_selective_ssm.ipynb) | selective state spaces, parallel scan |
| 40 | 2023 | [Speculative decoding](notebooks/40_2023_speculative_decoding.ipynb) | draft + verify, exact sampling |
| 41 | 2023 | [Diffusion Transformer](notebooks/41_2023_diffusion_transformer.ipynb) | DiT, adaLN-Zero, classifier-free guidance |
| 42 | 2023 | [Quantization](notebooks/42_2023_quantization.ipynb) | int8/int4, outliers, GPTQ |
| 43 | 2023 | [Vision-language model](notebooks/43_2023_vision_language_model.ipynb) | LLaVA: encoder → projector → LLM |
| 44 | 2017→2024 | [Mixture of Experts](notebooks/44_2023_mixture_of_experts.ipynb) | top-k routing, load balancing |
| 45 | 2024 | [KAN](notebooks/45_2024_kan.ipynb) | learnable spline activations on edges |
| 46 | 2024 | [Multi-head Latent Attention](notebooks/46_2024_multi_head_latent_attention.ipynb) | compressed KV cache, weight absorption |
| 47 | 2024 | [Multi-token prediction](notebooks/47_2024_multi_token_prediction.ipynb) | DeepSeek-V3 MTP, self-speculative drafting |
| 48 | 2025 | [GRPO](notebooks/48_2025_grpo_reasoning_rl.ipynb) | RL with verifiable rewards |
| 49 | 2025 | [Gated DeltaNet](notebooks/49_2025_gated_deltanet.ipynb) | delta-rule linear attention with gating |
| 50 | 2025 | [Gated attention](notebooks/50_2025_gated_attention.ipynb) | sigmoid output gate, attention sinks |
| 51 | 2025 | [Tiny Recursive Model](notebooks/51_2025_tiny_recursive_model.ipynb) | recursion + deep supervision |
| 52 | 2025 | [Diffusion language model](notebooks/52_2025_diffusion_language_model.ipynb) | LLaDA masked diffusion, infilling |

### 2026: inside today's frontier models
| # | Year | Notebook | Idea |
|---|---|---|---|
| 53 | 2026 | [Modern LLM recipe](notebooks/53_2026_modern_llm_recipe.ipynb) | RMSNorm, RoPE, GQA, QK-norm, SwiGLU, Muon |
| 54 | 2026 | [Sparse & hybrid attention](notebooks/54_2026_sparse_and_hybrid_attention.ipynb) | DeepSeek Sparse Attention (lightning indexer), local/global hybrids |
| 55 | 2026 | [mHC](notebooks/55_2026_manifold_constrained_hyper_connections.ipynb) | manifold-constrained hyper-connections, Sinkhorn (DeepSeek) |
| 56 | 2026 | [Engram](notebooks/56_2026_engram_conditional_memory.ipynb) | conditional memory via hashed N-gram lookup (DeepSeek) |
| 57 | 2026 | [Compressed sparse attention](notebooks/57_2026_compressed_sparse_attention.ipynb) | DeepSeek-V4 CSA + HCA, attention sinks, 1M context |
| 58 | 2026 | [Frontier capstone](notebooks/58_2026_frontier_model_capstone.ipynb) | Qwen3.5/V4-style: hybrid DeltaNet + gated attention, MoE, MTP, Muon |

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

Run times on an Apple M1 Pro (MPS): most notebooks take seconds to a few minutes; the longest (Mamba, Gated DeltaNet, the 2026 attention notebooks, the capstone) take 5–10 minutes. Without a GPU most notebooks run at a similar speed; the hand-written recurrent scans are the slowest (Mamba ~17 min on CPU). The DQN notebook deliberately runs on CPU, which is faster for its tiny networks.

## Repository layout

```
notebooks/      the notebooks (committed with outputs so they can be read on GitHub)
nb_src/         source of each notebook in a light cell-marker format ("#%% md" / "#%%")
tools/
  build.py        nb_src/*.py  ->  notebooks/*.ipynb (expands shared snippets)
  snippets/       shared code: device setup, digits loader, text corpora, a tiny GPT
  run_all.sh      rebuild and execute every notebook in place
  show_outputs.py print the text outputs of executed notebooks
```

To edit a notebook, either change the `.ipynb` directly, or edit `nb_src/<name>.py` and run `python tools/build.py <name>` (this overwrites the `.ipynb`). `tools/run_all.sh` rebuilds everything and re-executes it.

## Notes

- These are **teaching-scale** implementations: tiny models, small data, short training runs. They show the mechanism, not state-of-the-art numbers. Where a toy result does not show the effect seen at scale, the notebook says so.
- The 2026 notebooks follow the published designs: the DeepSeek-V4 technical report (arXiv 2606.19348), the mHC (arXiv 2512.24880) and Engram (arXiv 2601.07372) papers, DeepSeek-V3.2's sparse attention (arXiv 2512.02556), Qwen's gated attention (arXiv 2505.06708), and the Qwen3.5 model card. Where a detail was not specified, the notebook says which choice it made.
