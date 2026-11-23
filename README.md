# Neural Networks, 1943–2026

[![check](https://github.com/babushkai/neuralnetwork/actions/workflows/check.yml/badge.svg)](https://github.com/babushkai/neuralnetwork/actions/workflows/check.yml)

A runnable history of neural networks in 90 Jupyter notebooks, from the McCulloch–Pitts neuron to the architectures of late-2026 language models. Each notebook implements one milestone from scratch, runs a small experiment that tests the paper's central claim, and reports what the experiment shows, including when it does not reproduce at toy scale. Everything runs on a laptop.

<p align="center"><img src="assets/hero.png" width="860" alt="Plots produced by the notebooks: Hopfield recall, an echo state network predicting a chaotic series, a Neural Turing Machine's memory accesses, ResNet gradient flow, YOLO detections, in-context learning compared with least squares, latent diffusion samples with classifier-free guidance, and attention over depth."></p>

## Contents

- [The notebooks](#the-notebooks): [1943–1997](#foundations-19431997) · [2001–2016](#the-deep-learning-revival-20012016) · [2017–2022](#transformers-and-scale-20172022) · [2023–2025](#the-llm-era-20232025) · [2026](#2026)
- [How the notebooks are written](#how-the-notebooks-are-written)
- [Reading paths](#reading-paths)
- [Running the notebooks](#running-the-notebooks)
- [Repository layout](#repository-layout)
- [Sources and accuracy](#sources-and-accuracy)

## The notebooks

Each title links to the notebook on GitHub, where most render with their outputs. *Colab* opens the same notebook in Google Colab.

### Foundations, 1943–1997

| # | Year | Notebook | What it builds and measures |
|---|---|---|---|
| 01 | 1943 | [McCulloch–Pitts neuron](notebooks/01_1943_mcculloch_pitts_neuron.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/01_1943_mcculloch_pitts_neuron.ipynb) | Threshold logic units; logic gates built by hand |
| 02 | 1958 | [Perceptron](notebooks/02_1958_perceptron.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/02_1958_perceptron.ipynb) | The first learning rule and its convergence on separable data |
| 03 | 1960 | [ADALINE and the LMS rule](notebooks/03_1960_adaline.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/03_1960_adaline.ipynb) | Gradient descent on squared error |
| 04 | 1969 | [*Perceptrons* and XOR](notebooks/04_1969_xor_problem.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/04_1969_xor_problem.ipynb) | Why one layer cannot learn XOR, and why a hidden layer can |
| 05 | 1980 | [Neocognitron](notebooks/05_1980_neocognitron.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/05_1980_neocognitron.ipynb) | S/C layers trained by competitive learning; recognition under shifts vs. pixel models |
| 06 | 1982 | [Hopfield network](notebooks/06_1982_hopfield_network.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/06_1982_hopfield_network.ipynb) | Energy-based associative memory and its capacity |
| 07 | 1982 | [Self-organizing map](notebooks/07_1982_self_organizing_map.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/07_1982_self_organizing_map.ipynb) | Competitive, topology-preserving learning |
| 08 | 1985 | [Boltzmann machines and RBMs](notebooks/08_1985_boltzmann_machine_rbm.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/08_1985_boltzmann_machine_rbm.ipynb) | Stochastic generative networks, contrastive divergence |
| 09 | 1986 | [Backpropagation](notebooks/09_1986_backpropagation_mlp.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/09_1986_backpropagation_mlp.ipynb) | The multi-layer perceptron and the chain rule, in NumPy, with a gradient check |
| 10 | 1989 | [LeNet](notebooks/10_1989_lenet_cnn.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/10_1989_lenet_cnn.ipynb) | Convolution, weight sharing and pooling, trained end to end |
| 11 | 1990 | [Elman RNN](notebooks/11_1990_elman_rnn.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/11_1990_elman_rnn.ipynb) | Recurrence and backpropagation through time |
| 12 | 1992 | [REINFORCE](notebooks/12_1992_reinforce_policy_gradient.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/12_1992_reinforce_policy_gradient.ipynb) | The policy gradient; unbiasedness and variance of baselines; CartPole |
| 13 | 1997 | [LSTM](notebooks/13_1997_lstm.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/13_1997_lstm.ipynb) | Gated memory against vanishing gradients |

### The deep learning revival, 2001–2016

| # | Year | Notebook | What it builds and measures |
|---|---|---|---|
| 14 | 2001 | [Echo state networks](notebooks/14_2001_echo_state_network.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/14_2001_echo_state_network.ipynb) | A fixed random reservoir and a linear readout predict a chaotic series; the echo state property vs. spectral radius |
| 15 | 2006 | [Deep autoencoders](notebooks/15_2006_deep_autoencoder.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/15_2006_deep_autoencoder.ipynb) | Greedy layer-wise pre-training |
| 16 | 2010 | [Weight initialization](notebooks/16_2010_weight_initialization.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/16_2010_weight_initialization.ipynb) | Signal and gradient through 50 layers; Xavier vs. He; which 20-layer nets train |
| 17 | 2012 | [AlexNet](notebooks/17_2012_alexnet.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/17_2012_alexnet.ipynb) | ReLU, dropout, augmentation, GPUs |
| 18 | 2013 | [word2vec](notebooks/18_2013_word2vec.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/18_2013_word2vec.ipynb) | Skip-gram with negative sampling |
| 19 | 2013 | [Variational autoencoder](notebooks/19_2013_vae.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/19_2013_vae.ipynb) | The ELBO and the reparameterisation trick |
| 20 | 2013 | [Deep Q-Network](notebooks/20_2013_deep_q_network.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/20_2013_deep_q_network.ipynb) | Experience replay and target networks, ablated on CartPole |
| 21 | 2014 | [Dropout](notebooks/21_2014_dropout.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/21_2014_dropout.ipynb) | Memorisation of noisy labels with and without dropout; weight scaling vs. Monte Carlo averaging |
| 22 | 2014 | [GAN](notebooks/22_2014_gan.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/22_2014_gan.ipynb) | Adversarial training and mode collapse |
| 23 | 2014 | [GRU and seq2seq](notebooks/23_2014_gru_seq2seq.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/23_2014_gru_seq2seq.ipynb) | Encoder–decoder and the fixed-vector bottleneck |
| 24 | 2014 | [Bahdanau attention](notebooks/24_2014_bahdanau_attention.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/24_2014_bahdanau_attention.ipynb) | Additive attention and learned alignments |
| 25 | 2014 | [Adam and optimizers](notebooks/25_2014_adam_optimizers.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/25_2014_adam_optimizers.ipynb) | SGD, momentum, RMSProp, Adam, AdamW |
| 26 | 2014 | [VGG and Inception](notebooks/26_2014_vgg_inception.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/26_2014_vgg_inception.ipynb) | Measured receptive fields of stacked 3×3 convolutions; 1×1 bottlenecks; parameter and compute counts |
| 27 | 2014 | [Neural Turing Machine](notebooks/27_2014_neural_turing_machine.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/27_2014_neural_turing_machine.ipynb) | Differentiable read/write memory on the copy task; length generalisation vs. LSTM |
| 28 | 2015 | [Batch normalization](notebooks/28_2015_batchnorm.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/28_2015_batchnorm.ipynb) | Normalising activations during training |
| 29 | 2015 | [ResNet](notebooks/29_2015_resnet.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/29_2015_resnet.ipynb) | Plain vs. residual networks at increasing depth |
| 30 | 2015 | [U-Net](notebooks/30_2015_unet.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/30_2015_unet.ipynb) | Encoder–decoder with skip connections for segmentation |
| 31 | 2015 | [Knowledge distillation](notebooks/31_2015_knowledge_distillation.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/31_2015_knowledge_distillation.ipynb) | Soft targets from a teacher to a smaller student |
| 32 | 2016 | [WaveNet](notebooks/32_2016_wavenet.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/32_2016_wavenet.ipynb) | Dilated causal convolutions on raw audio |
| 33 | 2016 | [Graph convolutional network](notebooks/33_2016_graph_convolutional_network.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/33_2016_graph_convolutional_network.ipynb) | Message passing; semi-supervised node classification; over-smoothing |
| 34 | 2016 | [Byte-pair encoding](notebooks/34_2016_byte_pair_encoding.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/34_2016_byte_pair_encoding.ipynb) | A byte-level BPE tokenizer from scratch; characters vs. subwords for the same language model |
| 35 | 2016 | [Normalizing flows (RealNVP)](notebooks/35_2016_normalizing_flows_realnvp.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/35_2016_normalizing_flows_realnvp.ipynb) | Affine coupling layers; exact likelihood; log-determinant checked against autograd |
| 36 | 2016 | [YOLO](notebooks/36_2016_yolo_object_detection.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/36_2016_yolo_object_detection.ipynb) | Single-pass detection, IoU, non-maximum suppression, and mAP on synthetic scenes |

### Transformers and scale, 2017–2022

| # | Year | Notebook | What it builds and measures |
|---|---|---|---|
| 37 | 2017 | [Transformer](notebooks/37_2017_transformer.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/37_2017_transformer.ipynb) | Multi-head attention in an encoder–decoder |
| 38 | 2017 | [PPO](notebooks/38_2017_ppo.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/38_2017_ppo.ipynb) | Clipped surrogate and GAE; sample efficiency and policy drift vs. unclipped reuse and vanilla policy gradient |
| 39 | 2017 | [VQ-VAE](notebooks/39_2017_vq_vae.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/39_2017_vq_vae.ipynb) | Images as discrete tokens; a Transformer prior generates new ones |
| 40 | 2016–17 | [AlphaGo and AlphaZero](notebooks/40_2017_alphazero.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/40_2017_alphazero.ipynb) | MCTS with a policy–value network, learned by self-play; checked against perfect play |
| 41 | 2018 | [GPT](notebooks/41_2018_gpt.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/41_2018_gpt.ipynb) | Decoder-only next-token prediction |
| 42 | 2018 | [BERT](notebooks/42_2018_bert.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/42_2018_bert.ipynb) | Masked language model pre-training and fine-tuning |
| 43 | 2018 | [Graph attention network](notebooks/43_2018_graph_attention_network.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/43_2018_graph_attention_network.ipynb) | GAT and GATv2 against GCN on graphs with noisy edges; where the attention goes |
| 44 | 2018 | [Neural ODEs](notebooks/44_2018_neural_ode.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/44_2018_neural_ode.ipynb) | The adjoint method from scratch, checked against backprop; learning and extrapolating dynamics |
| 45 | 2020 | [Scaling laws](notebooks/45_2020_scaling_laws.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/45_2020_scaling_laws.ipynb) | Power laws and Chinchilla-style IsoFLOP curves |
| 46 | 2020 | [GPT-3 and in-context learning](notebooks/46_2020_gpt3_in_context_learning.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/46_2020_gpt3_in_context_learning.ipynb) | A Transformer that learns linear functions from its prompt, compared with least squares |
| 47 | 2020 | [Retrieval-augmented generation](notebooks/47_2020_retrieval_augmented_generation.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/47_2020_retrieval_augmented_generation.ipynb) | Dense dual-encoder retrieval and a reader; answering about facts added after training |
| 48 | 2020 | [SimCLR](notebooks/48_2020_simclr_self_supervised.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/48_2020_simclr_self_supervised.ipynb) | Self-supervised contrastive learning |
| 49 | 2020 | [Vision Transformer](notebooks/49_2020_vision_transformer.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/49_2020_vision_transformer.ipynb) | Image patches as tokens |
| 50 | 2020 | [DDPM](notebooks/50_2020_ddpm_diffusion.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/50_2020_ddpm_diffusion.ipynb) | Denoising diffusion |
| 51 | 2021 | [CLIP](notebooks/51_2021_clip.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/51_2021_clip.ipynb) | Contrastive image–text training and zero-shot classification |
| 52 | 2021 | [LoRA](notebooks/52_2021_lora.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/52_2021_lora.ipynb) | Low-rank adaptation |
| 53 | 2021 | [Masked autoencoders](notebooks/53_2021_masked_autoencoder.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/53_2021_masked_autoencoder.ipynb) | Encoder sees 25% of patches; linear probes with 50 labels; mask-ratio sweep |
| 54 | 2022 | [FlashAttention](notebooks/54_2022_flash_attention.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/54_2022_flash_attention.ipynb) | Tiled exact attention with an online softmax |
| 55 | 2022 | [Flow matching](notebooks/55_2022_flow_matching.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/55_2022_flow_matching.ipynb) | Rectified flow and reflow |
| 56 | 2022 | [Latent diffusion](notebooks/56_2022_latent_diffusion.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/56_2022_latent_diffusion.ipynb) | The same diffusion model in pixel and latent space; classifier-free guidance strength vs. diversity |
| 57 | 2022 | [RLHF (InstructGPT)](notebooks/57_2022_rlhf_instructgpt.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/57_2022_rlhf_instructgpt.ipynb) | Reward model from pairwise preferences, PPO with a KL penalty; learned vs. true reward |

### The LLM era, 2023–2025

| # | Year | Notebook | What it builds and measures |
|---|---|---|---|
| 58 | 2023 | [DPO](notebooks/58_2023_dpo_preference_optimization.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/58_2023_dpo_preference_optimization.ipynb) | Preference optimisation without a reward model or RL |
| 59 | 2023 | [Mamba](notebooks/59_2023_mamba_selective_ssm.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/59_2023_mamba_selective_ssm.ipynb) | Selective state spaces and the parallel scan |
| 60 | 2023 | [Speculative decoding](notebooks/60_2023_speculative_decoding.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/60_2023_speculative_decoding.ipynb) | Draft and verify, with exact sampling |
| 61 | 2023 | [Diffusion Transformer](notebooks/61_2023_diffusion_transformer.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/61_2023_diffusion_transformer.ipynb) | DiT, adaLN-Zero, classifier-free guidance |
| 62 | 2023 | [Quantization](notebooks/62_2023_quantization.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/62_2023_quantization.ipynb) | int8/int4, outlier features, GPTQ |
| 63 | 2023 | [Vision-language model](notebooks/63_2023_vision_language_model.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/63_2023_vision_language_model.ipynb) | LLaVA-style: vision encoder, projector, language model |
| 64 | 2017–24 | [Mixture of experts](notebooks/64_2023_mixture_of_experts.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/64_2023_mixture_of_experts.ipynb) | Top-k routing and load balancing |
| 65 | 2023 | [Sparse autoencoders](notebooks/65_2023_sparse_autoencoders.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/65_2023_sparse_autoencoders.ipynb) | Recovering features stored in superposition; SAE latents of a small language model |
| 66 | 2024 | [KAN](notebooks/66_2024_kan.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/66_2024_kan.ipynb) | Learnable spline activations on edges |
| 67 | 2024 | [Multi-head latent attention](notebooks/67_2024_multi_head_latent_attention.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/67_2024_multi_head_latent_attention.ipynb) | Compressed KV cache and weight absorption |
| 68 | 2024 | [Multi-token prediction](notebooks/68_2024_multi_token_prediction.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/68_2024_multi_token_prediction.ipynb) | DeepSeek-V3-style MTP and self-speculative drafting |
| 69 | 2025 | [GRPO](notebooks/69_2025_grpo_reasoning_rl.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/69_2025_grpo_reasoning_rl.ipynb) | Reinforcement learning with verifiable rewards |
| 70 | 2025 | [Gated DeltaNet](notebooks/70_2025_gated_deltanet.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/70_2025_gated_deltanet.ipynb) | Delta-rule linear attention with gating; chunkwise algorithm |
| 71 | 2025 | [Gated attention](notebooks/71_2025_gated_attention.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/71_2025_gated_attention.ipynb) | A sigmoid output gate and attention sinks |
| 72 | 2025 | [Tiny Recursive Model](notebooks/72_2025_tiny_recursive_model.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/72_2025_tiny_recursive_model.ipynb) | Recursion and deep supervision on puzzles |
| 73 | 2025 | [Diffusion language model](notebooks/73_2025_diffusion_language_model.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/73_2025_diffusion_language_model.ipynb) | Masked diffusion (LLaDA) and infilling |

### 2026

These notebooks follow designs published in 2026. Several are new enough that independent replications do not yet exist; each notebook states its sources and any detail it had to choose itself.

| # | Notebook | What it builds and measures | Used in |
|---|---|---|---|
| 74 | [The modern LLM recipe](notebooks/74_2026_modern_llm_recipe.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/74_2026_modern_llm_recipe.ipynb) | RMSNorm, RoPE, GQA, QK-norm and SwiGLU in one block; GPT-2 block vs. modern block, AdamW vs. Muon | most open LLMs |
| 75 | [Sparse and hybrid attention](notebooks/75_2026_sparse_and_hybrid_attention.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/75_2026_sparse_and_hybrid_attention.ipynb) | A learned top-k indexer; local/global layer hybrids | DeepSeek-V3.2 |
| 76 | [mHC](notebooks/76_2026_manifold_constrained_hyper_connections.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/76_2026_manifold_constrained_hyper_connections.ipynb) | Hyper-connections constrained by Sinkhorn normalisation | DeepSeek-V4 |
| 77 | [Engram](notebooks/77_2026_engram_conditional_memory.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/77_2026_engram_conditional_memory.ipynb) | Conditional memory via hashed N-gram lookup | DeepSeek |
| 78 | [Compressed sparse attention](notebooks/78_2026_compressed_sparse_attention.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/78_2026_compressed_sparse_attention.ipynb) | CSA and HCA KV compression; attention sinks | DeepSeek-V4 |
| 79 | [Early-2026 capstone](notebooks/79_2026_frontier_model_capstone.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/79_2026_frontier_model_capstone.ipynb) | Hybrid Gated DeltaNet + gated attention, MoE, MTP, Muon | Qwen3.5-style |
| 80 | [Kimi Delta Attention](notebooks/80_2026_kimi_delta_attention.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/80_2026_kimi_delta_attention.ipynb) | Channel-wise gated delta rule; chunkwise algorithm | Kimi Linear, K3 |
| 81 | [Mamba-3](notebooks/81_2026_mamba3.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/81_2026_mamba3.ipynb) | Trapezoidal discretisation, complex state as data-dependent RoPE, MIMO; parity | SSM hybrids |
| 82 | [Attention Residuals](notebooks/82_2026_attention_residuals.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/82_2026_attention_residuals.ipynb) | Softmax attention over depth instead of a fixed residual sum | Kimi |
| 83 | [LatentMoE](notebooks/83_2026_latent_moe.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/83_2026_latent_moe.ipynb) | Experts in a compressed space: 4× the experts at equal communication | Nemotron 3 |
| 84 | [Gated DeltaNet-2](notebooks/84_2026_gated_deltanet_2.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/84_2026_gated_deltanet_2.ipynb) | Separate channel-wise erase and write gates | NVIDIA |
| 85 | [Gated Residual](notebooks/85_2026_gated_residual.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/85_2026_gated_residual.ipynb) | Widened residual stream with read and write gates; stability at high learning rate | Qwen3.8-Next |
| 86 | [Cheaper sparse-attention indexers](notebooks/86_2026_cheaper_sparse_attention_indexers.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/86_2026_cheaper_sparse_attention_indexers.ipynb) | Block-compressed indexers and index reuse across layers | Qwen3.8, DeepSeek-V4.1 |
| 87 | [Causal encoder–decoder](notebooks/87_2026_causal_encoder_decoder.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/87_2026_causal_encoder_decoder.ipynb) | Encoder-only prefill; decoder KV from the encoder state; replay exactness | DeepSeek-V4.1-Flash |
| 88 | [Looped Transformers](notebooks/88_2026_looped_transformers.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/88_2026_looped_transformers.ipynb) | Recurrent depth and test-time loops on multi-hop lookup | several 2026 models |
| 89 | [Constant LR and checkpoint averaging](notebooks/89_2026_constant_lr_checkpoint_averaging.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/89_2026_constant_lr_checkpoint_averaging.ipynb) | Cosine vs. WSD vs. no decay with EMA or window averaging | 2026 pre-training runs |
| 90 | [Late-2026 capstone](notebooks/90_2026_late_2026_frontier_capstone.ipynb) · [Colab](https://colab.research.google.com/github/babushkai/neuralnetwork/blob/main/notebooks/90_2026_late_2026_frontier_capstone.ipynb) | Gated DeltaNet-2, sparse attention, attention residuals, LatentMoE, MTP and Muon in one model | all of the above |

No single notebook is "the state of the art". The open frontier of late 2026 is a combination: mostly linear-memory layers, a few sparse global-attention layers, learned residual mixing, and very sparse mixtures of experts (notebook 90). Closed labs do not publish their architectures, so only open designs are covered.

## How the notebooks are written

Every notebook follows the same structure:

1. **The paper and the problem it solved.** Primary citations first, then the idea in a few equations.
2. **A diagram of the design**, drawn to match the notebook's own implementation (layer sizes, gates, where the loss attaches).
3. **A from-scratch implementation.** NumPy for the early notebooks and plain PyTorch afterwards. No pretrained weights, model hubs, or library layers that hide the mechanism being taught.
4. **One experiment that isolates the claim.** For example: plain vs. residual networks at increasing depth; a reward model that is optimised with and without a KL penalty; a delta-rule memory against plain linear attention on associative recall.
5. **Takeaways and limits.** What the result shows, what it does not, and what the full-scale system did differently.

Conventions:

- **No downloads.** Data is scikit-learn's bundled 8×8 digits, synthetic tasks, or text generated inside the notebook. The notebooks run offline.
- **Standalone.** Each notebook runs top to bottom on its own. Shared code (device selection, data, a small GPT) is inlined at build time, not imported.
- **Toy scale, stated as such.** Models have thousands to a few million parameters and train for seconds to minutes. They demonstrate mechanisms, not benchmark numbers. Where a small model cannot show an effect reported at scale, the notebook says so.

## Reading paths

The notebooks are numbered chronologically, but each one stands alone. Some routes through them:

- **Language models:** 09 → 11 → 13 → 23 → 24 → 37 → 41 → 34 → 45 → 46 → 74 → 90
- **Vision:** 05 → 10 → 17 → 26 → 28 → 29 → 30 → 36 → 49 → 53
- **Generative models:** 08 → 19 → 22 → 35 → 39 → 50 → 55 → 56 → 61 → 73
- **Reinforcement learning and alignment:** 12 → 20 → 38 → 40 → 57 → 58 → 69
- **Memory and sequence models:** 06 → 13 → 14 → 27 → 59 → 70 → 80 → 81 → 84
- **Training and efficiency:** 16 → 21 → 25 → 28 → 31 → 52 → 54 → 60 → 62 → 64 → 89
- **Understanding trained networks:** 18 → 46 → 65

## Running the notebooks

```bash
git clone https://github.com/babushkai/neuralnetwork && cd neuralnetwork
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

Requirements: PyTorch 2.2+, NumPy, Matplotlib, scikit-learn (see `requirements.txt`). Each notebook selects a device automatically (CUDA, then Apple MPS, then CPU); set `NN_DEVICE=cpu` (or `cuda`, `mps`) to override. Notebooks 01–04 and 06–09 use only NumPy. A few with tiny models stepping one action at a time (reinforcement learning, the Neural Turing Machine, Neural ODEs) deliberately run on CPU.

**Run time.** On an Apple M1 Pro, most notebooks finish in under two minutes and the heaviest in 5–15 minutes. On CPU only, Mamba (59), WaveNet (32), mHC (76), compressed sparse attention (78) and the early-2026 capstone (79) take 10–20 minutes and Kimi Delta Attention (80) about 35; the other 2026 notebooks have not been timed on CPU.

**Saved outputs.** Every notebook except 84–89 is committed with the outputs of a full run, so it reads like an article on GitHub. Those six 2026 notebooks are the heaviest and have not been executed in this repository yet; run them locally or in Colab (roughly 5–20 minutes each on a recent GPU or Apple silicon). Reports of failures are welcome.

## Repository layout

```
notebooks/    the 90 notebooks (.ipynb)
assets/diagrams/  one architecture diagram per notebook (PNG), shown at the top of each notebook
nb_src/       their sources, in a plain-text "#%% md" / "#%%" cell format that diffs cleanly
tools/
  build.py         nb_src/<name>.py → notebooks/<name>.ipynb; expands shared snippets; keeps outputs of unchanged cells
  snippets/        shared code inlined at build time: device setup, digits loader, text corpora, a small GPT
  run_all.sh       rebuild and execute notebooks in place
  check.py         consistency checks run in CI: sources match notebooks, code parses, outputs saved, README complete
  make_figure.py   rebuild assets/hero.png from notebook outputs
  diagrams.py      the 90 architecture diagrams, as data; draws assets/diagrams/ (renderer: diagram_engine.py)
  show_outputs.py  print the text outputs of executed notebooks
```

To change a notebook, edit `nb_src/<name>.py`, run `python tools/build.py <name>`, then execute the notebook. `python tools/check.py` verifies the result. CI also executes a fast subset of notebooks on CPU. Pull requests that fix errors, sharpen an experiment, or add a milestone with a public paper are welcome; please include the executed notebook.

## Sources and accuracy

Each notebook cites its primary sources at the top: the original paper, and for later systems the technical report or model card. The 2026 notebooks rely on recent technical reports, including:

- **DeepSeek:** V3.2 sparse attention (arXiv 2512.02556), mHC (2512.24880), Engram (2601.07372), V4 (2606.19348), V4.1-Flash (2609.19969)
- **Qwen:** gated attention (2505.06708), the Qwen3.5 model card, the Qwen3.8-Next architecture report (2608.30320)
- **Moonshot (Kimi):** Kimi Linear (2510.26692), Attention Residuals (2603.15031)
- **NVIDIA:** Nemotron 3 (2604.12374, 2606.15007), Gated DeltaNet-2 (2605.22791)
- **Mamba-3** (2603.15569)

Where a source leaves a detail unspecified, the notebook says which choice it made. Where a claim rests on secondary reporting rather than a paper or model card, the notebook says so. If you find an error, please open an issue with a reference.

## License

[MIT](LICENSE)
