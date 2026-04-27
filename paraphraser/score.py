import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from dotenv import load_dotenv
from detector.detector import Detector
from paraphraser.fluency import fluency_score
from paraphraser.semantic import semantic_score

load_dotenv()

W_DETECTOR = float(os.getenv("REWARD_WEIGHT_DETECTOR", 0.5))
W_FLUENCY  = float(os.getenv("REWARD_WEIGHT_FLUENCY",  0.3))
W_SEMANTIC = float(os.getenv("REWARD_WEIGHT_SEMANTIC", 0.2))
THRESHOLD  = float(os.getenv("REWARD_THRESHOLD", 0.65))

_detector  = None


def _load_detector():
    global _detector
    if _detector is None:
        _detector = Detector()


def reward(original: str, rewrite: str) -> dict:
    """
    Score a candidate rewrite against the original.
    Returns a dict with individual scores and the combined reward R.

    R = W_DETECTOR * D + W_FLUENCY * F + W_SEMANTIC * S

    D = P(human) from the frozen detector (higher = looks more human)
    F = fluency score from GPT-2 perplexity (higher = more natural)
    S = semantic similarity (cosine) (higher = meaning preserved)
    """
    _load_detector()

    # Reject rewrites that are less than x% of the original length
    original_words = len(original.split())
    rewrite_words  = len(rewrite.split())
    if rewrite_words < original_words * 0.5:
        return {
            "detector": 0.0,
            "fluency":  0.0,
            "semantic": 0.0,
            "reward":   0.0,
            "passes":   False,
        }
    
    d = _detector.score(rewrite)
    f = fluency_score(rewrite)
    s = semantic_score(original, rewrite)

    if s < 0.5:
        r = 0.0
    else:
        r = W_DETECTOR * d + W_FLUENCY * f + W_SEMANTIC * s

    return {
        "detector":  round(d, 4),
        "fluency":   round(f, 4),
        "semantic":  round(s, 4),
        "reward":    round(r, 4),
        "passes":    r >= THRESHOLD,
    }


def score_candidates(original: str, candidates: list[str]) -> list[dict]:
    """
    Score a list of candidate rewrites and return them sorted
    by reward descending. Each item includes the rewrite text
    and all score components.
    """
    results = []
    for c in candidates:
        scores = reward(original, c)
        results.append({"text": c, **scores})

    results.sort(key=lambda x: x["reward"], reverse=True)
    return results


def top_k(original: str, candidates: list[str], k: int = 3) -> list[dict]:
    """
    Return the top-k candidates that pass the reward threshold,
    sorted by reward descending. Returns fewer than k if not enough
    candidates pass.
    """
    scored  = score_candidates(original, candidates)
    passing = [r for r in scored if r["passes"]]
    return passing[:k]


if __name__ == "__main__":
    original = (
        "The utilization of artificial intelligence in modern healthcare "
        "systems has demonstrated significant improvements in diagnostic "
        "accuracy and patient outcomes across multiple clinical domains."
    )

    candidates = [
        "AI has improved diagnosis and patient care in healthcare significantly.",
        "Modern hospitals are using machine learning to get better results for patients.",
        "The weather today is sunny with a chance of rain in the afternoon.",
        original,
    ]

    print(f"Original:\n  {original}\n")
    print(f"Weights: detector={W_DETECTOR} fluency={W_FLUENCY} semantic={W_SEMANTIC}")
    print(f"Threshold: {THRESHOLD}\n")
    print(f"{'Rewrite':<60} {'D':>6} {'F':>6} {'S':>6} {'R':>6} {'Pass':>5}")
    print("-" * 95)

    for r in score_candidates(original, candidates):
        text  = r["text"][:58]
        print(
            f"{text:<60} "
            f"{r['detector']:>6.3f} "
            f"{r['fluency']:>6.3f} "
            f"{r['semantic']:>6.3f} "
            f"{r['reward']:>6.3f} "
            f"{'yes' if r['passes'] else 'no':>5}"
        )

    print(f"\nTop-k passing rewrites (k=3):")
    for i, r in enumerate(top_k(original, candidates, k=3), 1):
        print(f"  [{i}] R={r['reward']:.3f}  {r['text'][:80]}")