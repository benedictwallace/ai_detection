import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from paraphraser.model import Paraphraser
from paraphraser.score import score_candidates
from detector.detector import Detector

CHECKPOINT = os.getenv("PARAPHRASER_CHECKPOINT", "past_versions/12_05/epoch_1/")

EXAMPLES = [
    "The utilization of artificial intelligence in modern healthcare systems has demonstrated significant improvements in diagnostic accuracy and patient outcomes across multiple clinical domains.",
    "Renewable energy sources such as solar and wind power have become increasingly cost-effective and are now being deployed at unprecedented scale across the globe.",
    "The implementation of remote work policies has fundamentally transformed organizational structures and employee productivity metrics in ways that were previously unanticipated.",
    "Recent advancements in large language models have raised significant questions regarding the nature of intelligence and the potential societal implications of increasingly capable AI systems.",
    "Urban planning initiatives that prioritize pedestrian infrastructure and public transportation have been shown to reduce carbon emissions and improve overall quality of life.",
]


def run_test(checkpoint: str = CHECKPOINT):
    print(f"\nLoading paraphraser from: {checkpoint}")
    paraphraser = Paraphraser()

    best = Path(checkpoint)
    if best.exists():
        paraphraser.load(checkpoint)
        print("Loaded fine-tuned checkpoint.")
    else:
        print(f"No checkpoint found at {checkpoint}, using base T5 model.")

    print("Loading detector...")
    detector = Detector()

    print("\n" + "=" * 80)
    print("PARAPHRASER TEST RESULTS")
    print("=" * 80)

    # Cache results in a single pass; previously test.py regenerated
    # candidates twice (once for printing, once for the saved JSON).
    per_example = []

    # Pre-score originals in a single batch call
    original_scores = detector.score_batch(EXAMPLES)

    for i, original in enumerate(EXAMPLES, 1):
        print(f"\nExample {i}")
        print(f"{'─' * 80}")
        print(f"Original:\n  {original}")
        print(f"  Detector score (original): P(human) = {original_scores[i-1]:.4f}")

        candidates = paraphraser.generate(original, n=4)
        scored     = score_candidates(original, candidates, detector=detector)

        print(f"\nRewrites (ranked by reward):")
        for j, r in enumerate(scored, 1):
            print(f"\n  [{j}] {r['text']}")
            print(
                f"        detector={r['detector']:.3f}  "
                f"fluency={r['fluency']:.3f}  "
                f"semantic={r['semantic']:.3f}  "
                f"reward={r['reward']:.3f}  "
                f"pass={'yes' if r['passes'] else 'no'}"
            )

        if scored:
            best_r = scored[0]
            delta  = best_r["detector"] - original_scores[i-1]
            print(f"\n  Best rewrite P(human) delta: {delta:+.4f}")

        per_example.append({
            "original":       original,
            "original_score": round(original_scores[i-1], 4),
            "candidates":     scored,
            "best_rewrite":   scored[0]["text"]     if scored else original,
            "rewrite_score":  scored[0]["detector"] if scored else None,
            "reward":         scored[0]["reward"]   if scored else None,
        })

    print("\n" + "=" * 80)
    print("BATCH EVASION RATE")
    print("=" * 80)

    rewrites           = [ex["best_rewrite"]   for ex in per_example]
    rewrite_scores_raw = [ex["rewrite_score"]  for ex in per_example if ex["rewrite_score"] is not None]

    original_evasion = sum(1 for s in original_scores if s >= 0.5) / len(original_scores)
    rewrite_evasion  = (sum(1 for s in rewrite_scores_raw if s >= 0.5) / len(rewrite_scores_raw)
                        if rewrite_scores_raw else 0.0)

    print(f"  Original text evasion rate: {original_evasion:.1%}")
    print(f"  Rewritten text evasion rate: {rewrite_evasion:.1%}")
    print(f"  Improvement: {rewrite_evasion - original_evasion:+.1%}")
    print("=" * 80 + "\n")

    results = {
        "checkpoint":       checkpoint,
        "original_evasion": round(original_evasion, 4),
        "rewrite_evasion":  round(rewrite_evasion, 4),
        "improvement":      round(rewrite_evasion - original_evasion, 4),
        "examples": [
            {
                "original":       ex["original"],
                "best_rewrite":   ex["best_rewrite"],
                "original_score": ex["original_score"],
                "rewrite_score":  ex["rewrite_score"],
                "reward":         ex["reward"],
            }
            for ex in per_example
        ],
    }

    Path("data").mkdir(exist_ok=True)
    out = Path("data/test_results_25.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    run_test()