"""Build a clean character-level TinyStories corpus for microllm.

usage: python -I data/prepare_tinystories.py RAW_DIR OUT_FILE TRAIN_CHARS VAL_CHARS

RAW_DIR holds TinyStoriesV2-GPT4-train.txt and TinyStoriesV2-GPT4-valid.txt (stories
separated by <|endoftext|>). The output is the training stories followed by the
validation stories, each story ending in a blank line, so `--val-frac` = VAL/(TRAIN+VAL)
holds out exactly the validation stories (to within one story).

Cleaning, in order (rules chosen for this project, not taken from a published recipe):
  1. normalize: CRLF, curly quotes, dashes, ellipses, non-breaking spaces -> ASCII;
     spaces collapsed, blank lines inside a story removed;
  2. reject (not repair) a story with any other character outside printable ASCII;
  3. reject very short or very long stories and degenerate ones (few distinct words);
  4. exact dedup on a lowercased, whitespace-free key, across both files;
  5. held-out leakage: a validation story is dropped when half or more of its word
     8-grams appear in one training story.
Train stories come from the train file and validation stories from the valid file, both
in file order, so the subset is deterministic.

TinyStories is by Ronen Eldan and Yuanzhi Li (arXiv:2305.07759), licensed under
CDLA-Sharing-1.0: https://huggingface.co/datasets/roneneldan/TinyStories
"""
import hashlib
import re
import sys
from collections import Counter

SEP = "<|endoftext|>"
FIX = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "–": "-", "—": " - ", "―": " - ", "−": "-", "‐": "-", "‑": "-",
    "…": "...", " ": " ", " ": " ", " ": " ", " ": " ",
    "​": "", "﻿": "", "\t": " ",
}
FIX_RE = re.compile("|".join(map(re.escape, FIX)))
OK = set(chr(c) for c in range(32, 127)) | {"\n"}
MIN_CHARS, MAX_CHARS = 200, 4000
NGRAM = 8


def normalize(s):
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = FIX_RE.sub(lambda m: FIX[m.group(0)], s)
    lines = [re.sub(r" {2,}", " ", l).strip() for l in s.split("\n")]
    return "\n".join(l for l in lines if l)


def stories(path):
    buf = []
    with open(path, encoding="utf-8", errors="strict") as f:
        for line in f:
            if SEP in line:
                head, _, tail = line.partition(SEP)
                buf.append(head)
                yield "".join(buf)
                buf = [tail]
            else:
                buf.append(line)
    if "".join(buf).strip():
        yield "".join(buf)


def key(s):
    return hashlib.blake2b(re.sub(r"\s+", "", s.lower()).encode(), digest_size=16).digest()


def grams(s):
    w = re.findall(r"[a-z']+", s.lower())
    return {hash(" ".join(w[i:i + NGRAM])) for i in range(len(w) - NGRAM + 1)}


def take(path, budget, seen, why):
    out, total = [], 0
    for raw in stories(path):
        s = normalize(raw)
        if not s:
            why["empty"] += 1
            continue
        if not set(s) <= OK:
            why["non-ascii"] += 1
            continue
        if not MIN_CHARS <= len(s) <= MAX_CHARS:
            why["length"] += 1
            continue
        words = s.lower().split()
        if len(set(words)) < 0.3 * len(words):
            why["repetitive"] += 1
            continue
        k = key(s)
        if k in seen:
            why["duplicate"] += 1
            continue
        seen.add(k)
        out.append(s)
        total += len(s) + 2
        if total >= budget:
            break
    return out


def main():
    raw, out_path, train_chars, val_chars = sys.argv[1], sys.argv[2], int(float(sys.argv[3])), int(float(sys.argv[4]))
    seen, why_t, why_v = set(), Counter(), Counter()
    # validation first in the seen-set so a story in both files stays held out
    val = take(f"{raw}/TinyStoriesV2-GPT4-valid.txt", val_chars, seen, why_v)
    train = take(f"{raw}/TinyStoriesV2-GPT4-train.txt", train_chars, seen, why_t)
    # leakage: drop validation stories mostly contained in a single training story
    owner = {}
    vgrams = [grams(s) for s in val]
    for j, g in enumerate(vgrams):
        for h in g:
            owner.setdefault(h, []).append(j)
    leaked = set()
    for s in train:
        hits = Counter()
        for h in grams(s):
            for j in owner.get(h, ()):
                hits[j] += 1
        leaked.update(j for j, c in hits.items() if c >= 0.5 * len(vgrams[j]))
    val = [s for j, s in enumerate(val) if j not in leaked]
    text_t = "".join(s + "\n\n" for s in train)
    text_v = "".join(s + "\n\n" for s in val)
    with open(out_path, "w", encoding="ascii", newline="\n") as f:
        f.write(text_t + text_v)
    n = len(text_t) + len(text_v)
    print(f"train: {len(train)} stories, {len(text_t)} chars; rejected {dict(why_t)}")
    print(f"val:   {len(val)} stories, {len(text_v)} chars; rejected {dict(why_v)}, leaked {len(leaked)}")
    print(f"alphabet: {len(set(text_t + text_v))} characters; --val-frac {len(text_v) / n:.6f}")


if __name__ == "__main__":
    main()
