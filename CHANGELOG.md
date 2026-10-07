# Changelog

Versions follow [Semantic Versioning](https://semver.org). The version is in `Cargo.toml`,
and each release is a `vX.Y.Z` tag.

## 0.1.0

First public version.

- Bigram, MLP and transformer language models with their own automatic differentiation, in
  plain Rust with no dependencies; SGD, AdamW and Muon; character and BPE tokenizers;
  optional NVIDIA GPU training.
- The studio (`microllm` with no arguments, or `START_HERE`): a local page that checks the
  computer, estimates a run's needs and starts or stops training.
- The viewer: every number of a run, live or replayed, from the loss and samples to attention,
  the tokenizer, a playground, run comparison, scaling and the loss landscape.
- `inside-one-training-step.html`: one training step of a tiny transformer, by hand.
- Six finished runs: five on Tiny Shakespeare that tell the story from a bigram to a
  transformer, plus a TinyStories showcase model.
