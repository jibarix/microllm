#!/bin/sh
# Double-click (Mac) or run (Linux) to open microllm in your browser. Keep this window open while you use it.
cd "$(dirname "$0")" || exit 1
if [ -x ./microllm ]; then exec ./microllm; fi
if [ -x target/release/microllm ]; then exec target/release/microllm; fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "microllm is not built yet, and Rust is not installed."
  echo "Install Rust from https://rustup.rs , then open START_HERE.command again."
  echo "The first start builds microllm, which takes about a minute."
  printf "Press Enter to close. "; read -r _
  exit 1
fi
echo "Building microllm (about a minute, the first time only)..."
cargo run --release || { printf "Press Enter to close. "; read -r _; }
