# Contributing

microllm runs code on people's computers, so changes come only from invited collaborators.
Pull requests from other accounts are turned off on this repository.

- **Found a bug or have an idea?** Open an issue. Say what you did, what happened, your
  operating system, and the `train.log` of the run if there is one.
- **Found a security problem?** Do not open an issue; follow [SECURITY.md](SECURITY.md).
- **Want to help with code?** Open an issue first to discuss it. If a collaborator invitation
  follows, changes go through a pull request that the maintainer reviews before merging.

Changes must keep microllm free of third-party crates, keep both servers on `127.0.0.1`, and
pass `cargo test --release` and `cargo run --release -- gradcheck`.
