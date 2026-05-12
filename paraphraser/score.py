import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from difflib import SequenceMatcher
from dotenv import load_dotenv
from detector.detector import Detector
from paraphraser.fluency import fluency_score
from paraphraser.semantic import semantic_score

load_dotenv()

# Defaults are intentionally aligned with the .env values so all modules
# read the same numbers when the env var is unset.
W_DETECTOR = float(os.getenv("REWARD_WEIGHT_DETECTOR", 0.7))
W_FLUENCY  = float(os.getenv("REWARD_WEIGHT_FLUENCY",  0.15))
W_SEMANTIC = float(os.getenv("REWARD_WEIGHT_SEMANTIC", 0.15))
THRESHOLD  = float(os.getenv("REWARD_THRESHOLD", 0.2))

_detector = None


def _load_detector():
    global _detector
    if _detector is None:
        _detector = Detector()


# def reward(original: str, rewrite: str) -> dict:

#     # NOT USEEEDDDDDD
#     _load_detector()

#     original_words = len(original.split())
#     rewrite_words  = len(rewrite.split())
#     if rewrite_words < original_words * 0.3:
#         return {
#             "detector": 0.0,
#             "fluency":  0.0,
#             "semantic": 0.0,
#             "reward":   0.0,
#             "passes":   False,
#         }

#     d = _detector.score(rewrite)
#     s = semantic_score(original, rewrite)

#     # Skip fluency if detector or semantic already make it a reject
#     if d < 0.05 or s < 0.65:
#         return {
#             "detector": round(d, 4),
#             "fluency":  0.0,
#             "semantic": round(s, 4),
#             "reward":   0.0,
#             "passes":   False,
#         }

#     f = fluency_score(rewrite)

#     d_shaped = d ** 0.6
#     r = W_DETECTOR * d_shaped + W_FLUENCY * f + W_SEMANTIC * s

#     return {
#         "detector": round(d, 4),
#         "fluency":  round(f, 4),
#         "semantic": round(s, 4),
#         "reward":   round(r, 4),
#         "passes":   r >= THRESHOLD,
#     }


def score_candidates(original: str, candidates: list[str], detector=None) -> list[dict]:
    """
    Score a list of candidate rewrites. Filters out ones that are too
    similar to the original (>=0.95 char ratio) or too short (<5 words).
    Returns the surviving candidates sorted by reward descending.
    """
    if detector is None:
        _load_detector()
        detector = _detector

    # Single pre-filter: too similar to original is the only structural reject
    # done up front. The length check is done before any model calls below
    # to save compute (was previously done after fluency/semantic forward passes).
    original_words = len(original.split())

    # pre model check filter
    stage1 = []
    for c in candidates:
        if len(c.split()) < 5:
            continue
        if len(c.split()) < original_words * 0.3:
            continue
        ratio = SequenceMatcher(None, original.lower(), c.lower()).ratio()
        if ratio >= 0.95:
            continue
        stage1.append(c)

    if not stage1:
        return []

    # dector score filter
    d_scores = detector.score_batch(stage1)

    stage2 = [(c, d) for c, d in zip(stage1, d_scores) if d >= 0.05]
    if not stage2:
        return []
    
    # semantic score filter
    stage3 = []
    for c, d in stage2:
        s = semantic_score(original, c)
        if s < 0.65:
            continue
        stage3.append((c, d, s))

    if not stage3:
        return []
    
    # passed run fluency
    results = []
    for c, d, s in stage3:
        f = fluency_score(c)

        # Square-root shaping on detector
        d_shaped = d ** 0.6

        r = W_DETECTOR * d_shaped + W_FLUENCY * f + W_SEMANTIC * s

        results.append({
            "text":     c,
            "detector": round(d, 4),
            "fluency":  round(f, 4),
            "semantic": round(s, 4),
            "reward":   round(r, 4),
            "passes":   r >= THRESHOLD,
        })

    results.sort(key=lambda x: x["reward"], reverse=True)
    return results


# def top_k(original: str, candidates: list[str], k: int = 3) -> list[dict]:
#     """
#     Return the top-k candidates that pass the reward threshold,
#     sorted by reward descending. Returns fewer than k if not enough
#     candidates pass. Currently unused by the GRPO training path but
#     kept for offline analysis and inference.
#     """
#     scored  = score_candidates(original, candidates)
#     passing = [r for r in scored if r["passes"]]
#     return passing[:k]


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
        text = r["text"][:58]
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