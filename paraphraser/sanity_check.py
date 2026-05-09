"""
sanity_check.py

Diagnostic script for inspecting what the paraphraser is actually doing.
Run from project root:
    python -m paraphraser.sanity_check
    python -m paraphraser.sanity_check --checkpoint checkpoints/paraphraser/epoch_3
    python -m paraphraser.sanity_check --n_samples 20 --n_candidates 8

Answers three questions:
  1. BREVITY COLLAPSE - is the model just shortening text?
  2. SEMANTIC DRIFT  - are high-detector candidates actually saying the same thing?
  3. REWARD COMPOSITION - is detector signal driving the reward, or is fluency dominating?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import random
import statistics
from pathlib import Path
from difflib import SequenceMatcher

from paraphraser.model    import Paraphraser
from paraphraser.score    import score_candidates, W_DETECTOR, W_FLUENCY, W_SEMANTIC
from paraphraser.fluency  import fluency_score
from paraphraser.semantic import semantic_score
from detector.detector    import Detector

VAL_FILE = Path("data/processed/val.jsonl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_texts(path: Path, n: int, seed: int = 42) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    texts.append(json.loads(line)["text"])
                except (json.JSONDecodeError, KeyError):
                    continue
    random.seed(seed)
    random.shuffle(texts)
    return texts[:n]


def char_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def word_count(s: str) -> int:
    return len(s.split())


def trunc(s: str, n: int = 110) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def bar(value: float, width: int = 20, max_val: float = 1.0) -> str:
    """Tiny ASCII bar for at-a-glance scoring."""
    filled = int(round(width * min(value, max_val) / max_val))
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diagnose_sample(idx: int, original: str, scored: list[dict],
                    original_score: float) -> dict:
    """Print a per-sample report and return aggregate metrics."""
    print(f"\n{'═' * 100}")
    print(f"SAMPLE {idx}")
    print(f"{'═' * 100}")
    print(f"ORIGINAL ({word_count(original)} words, P(human)={original_score:.4f}):")
    print(f"  {trunc(original, 200)}")

    if not scored:
        print("\n  ⚠ No scored candidates produced.")
        return {"ratio": [], "wc_ratio": [], "drifted": 0, "best_d": 0.0}

    print(f"\n  {'#':>2} {'P(h)':>6} {'flu':>6} {'sem':>6} {'rwd':>6} "
          f"{'wc':>4} {'sim':>5}  rewrite")
    print(f"  {'─' * 98}")

    word_ratios  = []
    char_ratios  = []
    drifted_n    = 0  # semantic < 0.5 = saying something different

    for j, r in enumerate(scored, 1):
        wc       = word_count(r["text"])
        wc_ratio = wc / max(1, word_count(original))
        ch_ratio = char_ratio(original, r["text"])
        word_ratios.append(wc_ratio)
        char_ratios.append(ch_ratio)
        if r["semantic"] < 0.5:
            drifted_n += 1

        # Mark suspicious patterns inline
        flag = ""
        if wc_ratio < 0.5:        flag += "S"  # too short
        if r["semantic"] < 0.5:   flag += "D"  # drifted
        if ch_ratio > 0.9:        flag += "C"  # near-copy

        print(f"  {j:>2} {r['detector']:>6.3f} {r['fluency']:>6.3f} {r['semantic']:>6.3f} "
              f"{r['reward']:>6.3f} {wc:>4} {ch_ratio:>5.2f}  {trunc(r['text'])} {flag}")

    # Reward decomposition for the BEST candidate — where is the score coming from?
    best = scored[0]
    d_contrib = W_DETECTOR * best["detector"]
    f_contrib = W_FLUENCY  * best["fluency"]
    s_contrib = W_SEMANTIC * best["semantic"]
    total     = d_contrib + f_contrib + s_contrib
    if total > 0:
        print(f"\n  Best candidate reward composition (pre-boost):")
        print(f"    detector contrib: {d_contrib:.4f} ({d_contrib/total:>5.1%})  {bar(d_contrib/total)}")
        print(f"    fluency  contrib: {f_contrib:.4f} ({f_contrib/total:>5.1%})  {bar(f_contrib/total)}")
        print(f"    semantic contrib: {s_contrib:.4f} ({s_contrib/total:>5.1%})  {bar(s_contrib/total)}")

    return {
        "ratio":   char_ratios,
        "wc_ratio": word_ratios,
        "drifted": drifted_n,
        "best_d":  best["detector"],
        "best_d_contrib_frac": d_contrib / total if total > 0 else 0.0,
    }


def aggregate_report(per_sample: list[dict], original_scores: list[float]):
    print(f"\n\n{'═' * 100}")
    print("AGGREGATE DIAGNOSTICS")
    print(f"{'═' * 100}")

    flat_wc      = [w for s in per_sample for w in s["wc_ratio"]]
    flat_char    = [c for s in per_sample for c in s["ratio"]]
    drifted_pct  = sum(s["drifted"] for s in per_sample) / max(
        1, sum(len(s["ratio"]) for s in per_sample)
    )
    best_ds      = [s["best_d"] for s in per_sample]
    best_d_frac  = [s["best_d_contrib_frac"] for s in per_sample if s["best_d_contrib_frac"] > 0]

    def stats(label, xs):
        if not xs:
            print(f"  {label}: no data")
            return
        print(f"  {label:35s} min={min(xs):.3f}  median={statistics.median(xs):.3f}  "
              f"mean={statistics.mean(xs):.3f}  max={max(xs):.3f}")

    print("\n[1] BREVITY COLLAPSE CHECK")
    print("    word_count(rewrite) / word_count(original)")
    print("    < 0.5 = rewrites are dramatically shorter (suspicious)")
    stats("    word ratio across all candidates", flat_wc)
    too_short_pct = sum(1 for w in flat_wc if w < 0.5) / max(1, len(flat_wc))
    print(f"    fraction of candidates with ratio < 0.5: {too_short_pct:.1%}")

    print("\n[2] NEAR-COPY CHECK")
    print("    char_similarity(original, rewrite)")
    print("    > 0.9 = barely changed (model not actually rewriting)")
    stats("    char similarity across candidates", flat_char)
    near_copy_pct = sum(1 for c in flat_char if c > 0.9) / max(1, len(flat_char))
    print(f"    fraction of candidates with similarity > 0.9: {near_copy_pct:.1%}")

    print("\n[3] SEMANTIC DRIFT CHECK")
    print(f"    fraction of candidates with semantic < 0.5: {drifted_pct:.1%}")
    print( "    high drift means high-detector wins are 'cheating' by changing the meaning")

    print("\n[4] DETECTOR PROGRESS CHECK")
    stats("    best detector score per sample", best_ds)
    crossed_pct = sum(1 for d in best_ds if d >= 0.5) / max(1, len(best_ds))
    print(f"    fraction of samples with best candidate >= 0.5 (would evade): {crossed_pct:.1%}")
    delta = [b - o for b, o in zip(best_ds, original_scores)]
    stats("    delta (best rewrite − original)", delta)

    print("\n[5] REWARD ATTRIBUTION CHECK")
    print("    For the best candidate per sample, what fraction of reward comes from detector?")
    print("    With weights detector=0.7, this should be HIGH if detector is driving learning.")
    print("    If it's < 0.3, your model is mostly being trained on fluency/semantic, not evasion.")
    stats("    detector contribution fraction", best_d_frac)

    print(f"\n{'═' * 100}")
    print("INTERPRETATION CHEAT SHEET")
    print(f"{'═' * 100}")
    print("""
S flag = candidate is < 50% the length of the original (brevity collapse)
D flag = semantic similarity to original < 0.5 (drifted, saying something else)
C flag = char similarity > 0.9 (essentially a copy of the original)
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/paraphraser/epoch_3",
        help="Path to a saved checkpoint, or a non-existent path to use the base model.",
    )
    parser.add_argument("--n_samples",    type=int, default=10)
    parser.add_argument("--n_candidates", type=int, default=6)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument(
        "--val_file", type=str, default=str(VAL_FILE),
        help="JSONL file to sample originals from (one {'text': ...} per line).",
    )
    args = parser.parse_args()

    print(f"Loading paraphraser…")
    paraphraser = Paraphraser()
    if Path(args.checkpoint).exists():
        paraphraser.load(args.checkpoint)
        print(f"checkpoint loaded: {args.checkpoint}")
    else:
        print(f"no checkpoint at {args.checkpoint}, using base model")

    print("Loading detector…")
    detector = Detector()

    print(f"Loading {args.n_samples} samples from {args.val_file}…")
    texts = load_texts(Path(args.val_file), args.n_samples, seed=args.seed)
    if not texts:
        print(f"No texts found in {args.val_file}")
        return
    print(f"loaded {len(texts)} samples")

    # Pre-score originals in one batch for efficiency
    original_scores = detector.score_batch(texts)

    per_sample = []
    for i, text in enumerate(texts, 1):
        candidates = paraphraser.generate(text, n=args.n_candidates)
        scored     = score_candidates(text, candidates, detector=detector)
        info       = diagnose_sample(i, text, scored, original_scores[i - 1])
        per_sample.append(info)

    aggregate_report(per_sample, original_scores)


if __name__ == "__main__":
    main()