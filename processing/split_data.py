import json
import random
import hashlib
from pathlib import Path

INPUT_FILE = Path("data/raw/ollama_samples.jsonl")
OUTPUT_DIR = Path("data/processed")
SEED       = 42
MIN_WORDS  = 50
MAX_WORDS  = 600


def load(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def clean(records):
    kept = []
    for r in records:
        words = len(r.get("text", "").split())
        if MIN_WORDS <= words <= MAX_WORDS:
            kept.append(r)
    return kept


def dedup(records):
    seen, unique = set(), []
    for r in records:
        h = hashlib.md5(r["text"].strip().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(r)
    return unique


def main():
    print(f"Loading {INPUT_FILE}...")
    records = load(INPUT_FILE)
    print(f"  {len(records)} records loaded")

    records = clean(records)
    print(f"  {len(records)} after cleaning")

    records = dedup(records)
    print(f"  {len(records)} after dedup")

    random.seed(SEED)
    random.shuffle(records)

    n       = len(records)
    n_val   = int(n * 0.10)
    n_test  = int(n * 0.10)
    n_train = n - n_val - n_test

    train = records[:n_train]
    val   = records[n_train:n_train + n_val]
    test  = records[n_train + n_val:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = OUTPUT_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(split)} samples -> {path}")

    print("Done.")


if __name__ == "__main__":
    main()