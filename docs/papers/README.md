# Papers behind microllm

Sources for the techniques this project implements, with where each one lives in the code.
Attributions were cross-checked against the code with OpenAI's Codex, an AI coding assistant,
on 2026-10-07; see "Credits and AI use" in the main README. Line numbers are as of that date.

## The three pillar papers

| Paper | Link | Where it shows up |
|---|---|---|
| Vaswani et al. 2017, Attention Is All You Need | [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) | Cited in `inside-one-training-step.html:8`, which follows it closely (sinusoidal positions, post-LN, ReLU). The viewer's Transformer (`src/model.rs:400-449`) is the GPT-style descendant: learned positions, pre-LN, GELU, decoder only. |
| Kaplan et al. 2020, Scaling Laws for Neural Language Models | [arXiv 2001.08361](https://arxiv.org/abs/2001.08361) | The viewer's Scaling panel (`viewer/index.html:400`, fit near `:2681`) fits bits/char ∝ N^(−α), the form of Kaplan's L(N). A local analogue: total parameters at a fixed budget, not non-embedding parameters at convergence. |
| Devlin et al. 2018, BERT | [arXiv 1810.04805](https://arxiv.org/abs/1810.04805) | Not used. microllm is its opposite: a causal next-token model, not a bidirectional masked LM, with character or BPE tokens rather than WordPiece. |

## Implemented techniques

| Technique | Code | Source | Link |
|---|---|---|---|
| Muon: Newton–Schulz quintic (3.4445, −4.775, 2.0315), Nesterov momentum | `src/optim.rs:17-64, 126-147` | Keller Jordan 2024 (blog, no paper) | [kellerjordan.github.io](https://kellerjordan.github.io/posts/muon/) |
| Muon update scale 0.2·√max(rows, cols), matching AdamW's RMS | `src/optim.rs:30-34` | Liu et al. 2025, Muon is Scalable for LLM Training | [arXiv 2502.16982](https://arxiv.org/abs/2502.16982) |
| AdamW (decoupled weight decay) | `src/optim.rs:87-124, 163-170` | Loshchilov & Hutter 2017 | [arXiv 1711.05101](https://arxiv.org/abs/1711.05101) |
| Adam moments and bias correction | `src/optim.rs:111-124` | Kingma & Ba 2014 | [arXiv 1412.6980](https://arxiv.org/abs/1412.6980) |
| Cosine decay (no restarts) | `src/train.rs:107-123` | Loshchilov & Hutter, SGDR | [arXiv 1608.03983](https://arxiv.org/abs/1608.03983) |
| Linear warmup | `src/train.rs:107-116` | Goyal et al. 2017 (popularizer, not first origin) | [arXiv 1706.02677](https://arxiv.org/abs/1706.02677) |
| Warmup–stable–decay schedule | `src/train.rs:126-138` | Hu et al. 2024, MiniCPM | [arXiv 2404.06395](https://arxiv.org/abs/2404.06395) |
| GELU (tanh approximation) | `src/autograd.rs:564-570` | Hendrycks & Gimpel 2016 | [arXiv 1606.08415](https://arxiv.org/abs/1606.08415) |
| Layer normalization | `src/model.rs:360-364` | Ba, Kiros & Hinton 2016 | [arXiv 1607.06450](https://arxiv.org/abs/1607.06450) |
| Pre-LN blocks, final LN, learned positions, init std 0.02, residual outputs scaled by 1/√(2L) | `src/model.rs:230-347, 400-449` | Radford et al. 2019, GPT-2 | [cdn.openai.com](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| Pre-LN analysis (why it trains stably without warmup tricks) | — | Xiong et al. 2020 (analysis; GPT-2 came first) | [arXiv 2002.04745](https://arxiv.org/abs/2002.04745) |
| Residual connections | `src/model.rs:437-445` | He et al. 2015 | [arXiv 1512.03385](https://arxiv.org/abs/1512.03385) |
| Subword BPE over Unicode characters (not GPT-2 byte-level) | `src/bpe.rs` | Sennrich et al. 2015 | [arXiv 1508.07909](https://arxiv.org/abs/1508.07909) |
| TinyStories corpus (optional; the default corpus is Tiny Shakespeare) | `data/`, `src/main.rs:103` | Eldan & Li 2023 | [arXiv 2305.07759](https://arxiv.org/abs/2305.07759) |
| MLP language model | `src/model.rs:159-217` | Bengio et al. 2003 | [jmlr.org](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) |
| Fan-in initialization; saturation diagnostics background | `src/model.rs:171-186`, `src/metrics.rs:17-83` | Glorot & Bengio 2010 (background only; the 5/3 tanh gain is PyTorch's `calculate_gain` convention) | [proceedings.mlr.press](https://proceedings.mlr.press/v9/glorot10a.html) |
| `relu2` variant (squared ReLU) | `src/model.rs:239-240`, `src/autograd.rs:573-576` | So et al. 2021, Primer | [arXiv 2109.08668](https://arxiv.org/abs/2109.08668) |
| `qk-norm` variant (mean/variance normalization of q and k with a gain; 1/√d kept) | `src/model.rs:241-243, 417-421` | Dehghani et al. 2023, ViT-22B | [arXiv 2302.05442](https://arxiv.org/abs/2302.05442) |
| `zero-init` variant (zero last matrix of each residual branch) | `src/model.rs:244-246, 325-347` | Fixup-inspired (Zhang et al. 2019); not full Fixup | [arXiv 1901.09321](https://arxiv.org/abs/1901.09321) |
| Head ablation | `src/model.rs:429-435`, `src/score.rs:85-101` | Michel et al. 2019 | [arXiv 1905.10650](https://arxiv.org/abs/1905.10650) |
| Induction-head pattern measurement | `src/metrics.rs:160-203` | Olsson et al. 2022 | [arXiv 2209.11895](https://arxiv.org/abs/2209.11895) |
| Attention entropy and distance diagnostics | `src/metrics.rs:86-128` | Related: Clark et al. 2019 (the exact formulas are local) | [arXiv 1906.04341](https://arxiv.org/abs/1906.04341) |
| Logit lens | `viewer/index.html:368-370` | nostalgebraist 2020 (LessWrong, no paper) | [lesswrong.com](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) |
| Loss landscape: the loss over the plane of the training path's two principal directions (PCA of θᵢ − θ_final), with the path projected onto it | `src/landscape.rs`, the viewer's "valley" panel on Overview | Li et al. 2018, Visualizing the Loss Landscape of Neural Nets, sec. 7 (their filter normalization is for random directions and is not used) | [arXiv 1712.09913](https://arxiv.org/abs/1712.09913) |

The three variants and Muon were picked from Keller Jordan's
[modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) speedrun records (MIT license); the table credits each
technique's original paper. The TinyStories dataset is licensed under
[CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/) ([dataset card](https://huggingface.co/datasets/roneneldan/TinyStories)).

Classical sources, cited by DOI on the pages:

| Technique | Code | Source |
|---|---|---|
| Bits as surprise, bits per character, the bigram model | `src/score.rs`, `src/model.rs` | Shannon 1948, [doi:10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x) |
| Plain SGD | `src/optim.rs:82-85` | Robbins & Monro 1951, [doi:10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586) |
| Nesterov momentum as Muon uses it | `src/optim.rs:126-147` | Sutskever et al. 2013, [PMLR v28](https://proceedings.mlr.press/v28/sutskever13.html) |
| Reverse-mode autodiff (the tape) | `src/autograd.rs` | Linnainmaa 1976, [doi:10.1007/BF01931367](https://doi.org/10.1007/BF01931367) |
| Backpropagation | `src/autograd.rs` | Rumelhart, Hinton & Williams 1986, [doi:10.1038/323533a0](https://doi.org/10.1038/323533a0) |
| SplitMix64 random numbers | `src/rng.rs` | Steele, Lea & Flood 2014, [doi:10.1145/2660193.2660195](https://doi.org/10.1145/2660193.2660195) |
| Normal draws for initialization | `src/rng.rs` | Box & Muller 1958, [doi:10.1214/aoms/1177706645](https://doi.org/10.1214/aoms/1177706645) |
| PCA for the embedding map and the loss landscape's plane | `viewer/index.html`, `src/landscape.rs` | Pearson 1901, [doi:10.1080/14786440109462720](https://doi.org/10.1080/14786440109462720) |
| The 5/3 tanh gain | `src/model.rs:171` | [PyTorch `calculate_gain`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain) |
| Tiny Shakespeare (default text) | `src/main.rs:103` | [Karpathy, char-rnn](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) (MIT license) |
| ReLU (lesson page only) | `inside-one-training-step.html` | [Nair & Hinton 2010](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf) |

Standard methods with no single source: softmax and cross-entropy, sampling from the predicted
distribution, temperature (the Playground's negative temperatures are a local extension), the
key/value cache, and five-point gradient checking. The viewer's mulberry32 seed generator
follows [bryc's public-domain JavaScript version](https://github.com/bryc/code/blob/master/jshash/PRNGs.md#mulberry32). The TinyStories cleaning rules in
`data/prepare_tinystories.py` are this project's own choices.

## Where the pages cite these

- `viewer/index.html`: a Sources panel at the bottom of Overview lists the sources behind the
  loaded run (its model, optimizer, schedule, variants, tokenizer and text), with a full list
  under it. Short "Source:" lines sit under each explanation. The registry is `SOURCES`, next to
  `dataName`.
- `inside-one-training-step.html`: a sources line in the footer.

## Related, but not implemented here

| Paper | Link | Why it is here |
|---|---|---|
| Henry et al. 2020, Query-Key Normalization | [arXiv 2010.04245](https://arxiv.org/abs/2010.04245) | L2-normalized q/k with a learned scale; microllm's `qk-norm` follows ViT-22B instead. |
| Fan et al. 2018, top-k sampling | [arXiv 1805.04833](https://arxiv.org/abs/1805.04833) | The viewer's `topK` only sorts values for display; sampling uses the full distribution. |
| Holtzman et al. 2019, nucleus (top-p) sampling | [arXiv 1904.09751](https://arxiv.org/abs/1904.09751) | Not implemented; relevant to the temperature experiments (degeneration at low temperature). |
