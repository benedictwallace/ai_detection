import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
from paraphraser.model import Paraphraser
from paraphraser.score import score_candidates
from detector.detector import Detector

CHECKPOINT = "past_versions/08_05/paraphraser/epoch_3" 
N_EVAL = 100

paraphraser = Paraphraser()
paraphraser.load(CHECKPOINT)
detector = Detector()

# Load 100 val examples
with open("data/processed/val.jsonl") as f:
    texts = [json.loads(line)["text"] for line in f][:N_EVAL]

original_scores = detector.score_batch(texts)
rewrite_scores = []
not_scored = 0

for i, text in enumerate(texts):
    candidates = paraphraser.generate(text, n=16) 
    scored = score_candidates(text, candidates, detector=detector)
    if scored:
        rewrite_scores.append(scored[0])
        print(f"{i+1}/{N_EVAL}: orig={original_scores[i]:.3f} → rewrite={rewrite_scores[-1]["detector"]:.3f}")
    else:
        not_scored += 1
    

orig_evasion    = sum(1 for s in original_scores if s >= 0.5) / len(original_scores)
rewrite_evasion = sum(1 for s in rewrite_scores if s["detector"] >= 0.5) / len(rewrite_scores)
rewrite_avg     = sum(s["detector"] for s in rewrite_scores) / len(rewrite_scores)
semantic_avg    = sum(s["semantic"] for s in rewrite_scores) / len(rewrite_scores)
fluency_avg     = sum(s["fluency"]  for s in rewrite_scores) / len(rewrite_scores)
reward_avg      = sum(s["reward"]   for s in rewrite_scores) / len(rewrite_scores)

print(f"\nOriginal evasion: {orig_evasion:.1%}")
print(f"Rewrite evasion:  {rewrite_evasion:.1%}")
print(f"Improvement:      {rewrite_evasion - orig_evasion:+.1%}")