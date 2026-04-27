import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from paraphraser.model import Paraphraser
from paraphraser.score import score_candidates
from detector.detector import Detector

CHECKPOINT = os.getenv("PARAPHRASER_CHECKPOINT", "checkpoints/paraphraser/best")

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
        print("No checkpoint found, using base T5 model.")

    print("Loading detector...")
    detector = Detector()

    print("\n" + "=" * 80)
    print("PARAPHRASER TEST RESULTS")
    print("=" * 80)

    for i, original in enumerate(EXAMPLES, 1):
        print(f"\nExample {i}")
        print(f"{'─' * 80}")
        print(f"Original:\n  {original}")
        print(f"  Detector score (original): P(human) = {detector.score(original):.4f}")

        candidates = paraphraser.generate(original, n=4)
        scored     = score_candidates(original, candidates)

        print(f"\nRewrites (ranked by reward):")
        for j, r in enumerate(scored, 1):
            print(f"\n  [{j}] {r['text']}")
            print(f"        detector={r['detector']:.3f}  fluency={r['fluency']:.3f}  semantic={r['semantic']:.3f}  reward={r['reward']:.3f}  pass={'yes' if r['passes'] else 'no'}")

        if scored:
            best_r = scored[0]
            delta = best_r["detector"] - detector.score(original)
            print(f"\n  Best rewrite P(human) delta: {delta:+.4f}")

    print("\n" + "=" * 80)
    print("BATCH EVASION RATE")
    print("=" * 80)

    rewrites = []
    for original in EXAMPLES:
        candidates = paraphraser.generate(original, n=4)
        scored     = score_candidates(original, candidates)
        rewrites.append(scored[0]["text"] if scored else original)

    original_evasion = detector.evasion_rate(EXAMPLES)
    rewrite_evasion  = detector.evasion_rate(rewrites)

    print(f"  Original text evasion rate: {original_evasion:.1%}")
    print(f"  Rewritten text evasion rate: {rewrite_evasion:.1%}")
    print(f"  Improvement: {rewrite_evasion - original_evasion:+.1%}")
    print("=" * 80 + "\n")

    results = {
        "checkpoint": checkpoint,
        "original_evasion": round(original_evasion, 4),
        "rewrite_evasion":  round(rewrite_evasion, 4),
        "improvement":      round(rewrite_evasion - original_evasion, 4),
        "examples": [
            {
                "original":        ex,
                "best_rewrite":    scored[0]["text"] if scored else ex,
                "original_score":  round(detector.score(ex), 4),
                "rewrite_score":   scored[0]["detector"] if scored else None,
                "reward":          scored[0]["reward"] if scored else None,
            }
            for ex, scored in [
                (ex, score_candidates(ex, paraphraser.generate(ex, n=4)))
                for ex in EXAMPLES
            ]
        ]
    }

    out = Path("data/test_results_25.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    run_test()