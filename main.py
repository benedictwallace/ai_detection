"""
main.py - single entry point for the NLP assignment autograder.

Runs the full pipeline end-to-end:
    1. Data collection      (skipped if cached file present; needs Ollama)
    2. Split + clean        (always reruns; fast, deterministic)
    3. SFT pair generation  (skipped if cached file present; needs Ollama)
    4. SFT warm-up          (skipped if checkpoint present)
    5. GRPO training        (reduced demo run: few epochs, capped samples)
    6. Evaluation on held-out test set
    7. Plots saved to disk  (loss curve + evasion rate)

Ships with cached data and checkpoints so the
autograder, which has no Ollama, can still produce results.

Run from the repo root:
    python main.py
"""

import os
import sys
import json
import random
import logging
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Quiet noisy libs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("main")
for noisy in ("sentence_transformers", "transformers", "httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DATA       = Path("data/raw/ollama_samples.jsonl")
TRAIN_FILE     = Path("data/processed/train.jsonl")
VAL_FILE       = Path("data/processed/val.jsonl")
TEST_FILE      = Path("data/processed/test.jsonl")
SFT_PAIRS      = Path("data/processed/sft_pairs.jsonl")
SFT_CKPT       = Path("checkpoints/paraphraser/sft_init")
GRPO_HISTORY   = Path("data/train_history.json")
PLOTS_DIR      = Path("data/plots")
RESULTS_FILE   = Path("data/main_results.json")


# ---------------------------------------------------------------------------
# Step 1: Data collection (Ollama-dependent, skip for autograder)
# ---------------------------------------------------------------------------
def step_collect_data() -> None:
    if RAW_DATA.exists() and RAW_DATA.stat().st_size > 0:
        logger.info(f"[1/6] Raw data found at {RAW_DATA}, skipping collection.")
        return

    logger.info(f"[1/6] No raw data at {RAW_DATA}, attempting Ollama collection...")
    try:
        from collection.ollama_collection import main as collect_main
        collect_main()
    except Exception as e:
        logger.error(
            f"Data collection failed: {e}. "
            f"The repo should ship with cached data at {RAW_DATA}. "
            f"Cannot proceed without raw data."
        )
        raise

# ---------------------------------------------------------------------------
# Step 2: Split + clean
# ---------------------------------------------------------------------------
def step_split_data() -> None:
    logger.info("[2/6] Splitting raw data into train/val/test...")
    from processing.split_data import main as split_main
    split_main()
    for f in (TRAIN_FILE, VAL_FILE, TEST_FILE):
        if not f.exists():
            raise RuntimeError(f"Split did not produce expected file: {f}")


# ---------------------------------------------------------------------------
# Step 3: SFT pair generation (Ollama-dependent, skip for autograder)
# ---------------------------------------------------------------------------
def step_generate_sft_pairs(target: int = 500) -> None:
    if SFT_PAIRS.exists() and SFT_PAIRS.stat().st_size > 0:
        n = sum(1 for _ in open(SFT_PAIRS, encoding="utf-8"))
        logger.info(f"[3/6] SFT pairs found at {SFT_PAIRS} ({n} pairs), skipping generation.")
        return

    logger.info(f"[3/6] No SFT pairs at {SFT_PAIRS}, attempting Ollama generation...")
    try:
        from collection.generate_sft_data import generate
        generate(target_pairs=target)
    except Exception as e:
        logger.warning(
            f"SFT pair generation failed: {e}. "
            f"Continuing without SFT warm-up: GRPO will start from the base model."
        )


# ---------------------------------------------------------------------------
# Step 4: SFT warm-up (skipped if checkpoint already present)
# ---------------------------------------------------------------------------
def step_sft_warmup() -> None:
    if SFT_CKPT.exists() and any(SFT_CKPT.iterdir()):
        logger.info(f"[4/6] SFT checkpoint found at {SFT_CKPT}, skipping warm-up.")
        return

    if not SFT_PAIRS.exists():
        logger.warning("[4/6] No SFT pairs available; skipping warm-up.")
        return

    logger.info("[4/6] Running SFT warm-up...")
    from paraphraser.sft_warmup import train as sft_train
    sft_train()


# ---------------------------------------------------------------------------
# Step 5: GRPO training (reduced demo run for autograder)
# ---------------------------------------------------------------------------
def step_grpo_training() -> None:
    logger.info("[5/6] Running GRPO training (reduced demo run)...")

    # Max running over 50 samples for 1 epoch
    os.environ.setdefault("EPOCHS", "1")
    os.environ.setdefault("MAX_TRAIN_SAMPLES", "50")
    logger.info(
        f"Demo settings: EPOCHS={os.environ['EPOCHS']}, "
        f"MAX_TRAIN_SAMPLES={os.environ['MAX_TRAIN_SAMPLES']}"
    )

    from paraphraser.training_loop import train as grpo_train
    grpo_train()


# ---------------------------------------------------------------------------
# Step 6: Evaluation on held-out test set
# ---------------------------------------------------------------------------
def step_evaluate() -> dict:
    logger.info("[6/6] Evaluating on held-out test set...")

    from paraphraser.model import Paraphraser
    from paraphraser.score import score_candidates
    from detector.detector import Detector

    # Prefer the best GRPO checkpoint, then last epoch, then SFT, then base.
    candidates = [
        Path("checkpoints/paraphraser/best"),
        Path("checkpoints/paraphraser/epoch_1"),
        SFT_CKPT,
    ]
    ckpt = next((c for c in candidates if c.exists() and any(c.iterdir())), None)

    paraphraser = Paraphraser()
    if ckpt is not None:
        paraphraser.load(str(ckpt))
        logger.info(f"Evaluating checkpoint: {ckpt}")
    else:
        logger.warning("No checkpoint available, evaluating base model.")

    detector = Detector()

    # Cap test set size so eval runtime is bounded for the autograder.
    n_eval = int(os.getenv("N_EVAL", 50))
    texts: list[str] = []
    with open(TEST_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                texts.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                continue
            if len(texts) >= n_eval:
                break

    original_scores: list[float] = detector.score_batch(texts)
    rewrite_scores: list[float] = []
    for i, text in enumerate(texts):
        cands  = paraphraser.generate(text, n=8)
        scored = score_candidates(text, cands, detector=detector)
        if scored:
            rewrite_scores.append(scored[0]["detector"])
        else:
            rewrite_scores.append(original_scores[i])

    orig_evasion    = sum(1 for s in original_scores if s >= 0.5) / len(original_scores)
    rewrite_evasion = sum(1 for s in rewrite_scores  if s >= 0.5) / len(rewrite_scores)
    improvement     = rewrite_evasion - orig_evasion

    orig_avg    = sum(original_scores) / len(original_scores)
    rewrite_avg = sum(rewrite_scores)  / len(rewrite_scores)

    logger.info(f"Original evasion rate: {orig_evasion:.1%}")
    logger.info(f"Rewrite evasion rate:  {rewrite_evasion:.1%}")
    logger.info(f"Improvement:           {improvement:+.1%}")
    logger.info(
        f"Original detector scores: avg={orig_avg:.4f} "
        f"min={min(original_scores):.4f} max={max(original_scores):.4f}"
    )
    logger.info(
        f"Rewrite  detector scores: avg={rewrite_avg:.4f} "
        f"min={min(rewrite_scores):.4f} max={max(rewrite_scores):.4f}"
    )
    return {
        "n_eval":           len(texts),
        "checkpoint":       str(ckpt) if ckpt else "base_model",
        "original_evasion": round(orig_evasion, 4),
        "rewrite_evasion":  round(rewrite_evasion, 4),
        "improvement":      round(improvement, 4),
        "original_scores":  [round(s, 4) for s in original_scores],
        "rewrite_scores":   [round(s, 4) for s in rewrite_scores],
    }


# ---------------------------------------------------------------------------
# Step 7: Plots (loss curve + evasion rate)
# ---------------------------------------------------------------------------
def step_plots(eval_results: dict) -> None:
    logger.info("Saving plots to disk...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Plot 1: training loss curve from train_history.json
    if GRPO_HISTORY.exists():
        with open(GRPO_HISTORY, encoding="utf-8") as f:
            history = json.load(f)
        epochs       = [e["epoch"]         for e in history["epochs"]]
        raw_losses   = [e["avg_raw_loss"]  for e in history["epochs"]]
        grpo_losses  = [e["avg_grpo_loss"] for e in history["epochs"]]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, raw_losses,  marker="o", label="raw policy loss (NLL)")
        ax.plot(epochs, grpo_losses, marker="s", label="GRPO objective")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training loss per epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = PLOTS_DIR / "loss_curve.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info(f"  saved {out}")
    else:
        logger.warning(f"No training history at {GRPO_HISTORY}, skipping loss curve.")

    # --- Plot 2: evasion rate before vs after
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Original", "Rewrite"]
    values = [eval_results["original_evasion"], eval_results["rewrite_evasion"]]
    bars   = ax.bar(labels, values, color=["#888", "#3a7"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Evasion rate (P(human) >= 0.5)")
    ax.set_title(f"Detector evasion on {eval_results['n_eval']} test samples")
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.02,
            f"{v:.1%}", ha="center", fontsize=11,
        )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / "evasion_rate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"  saved {out}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("NLP assignment pipeline, main.py")
    logger.info(f"Working directory: {ROOT}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info("=" * 60)

    # Make sure all output dirs exist up front
    for d in (Path("data"), Path("data/raw"), Path("data/processed"), Path("checkpoints/paraphraser"), PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    step_collect_data()
    step_split_data()
    step_generate_sft_pairs()
    step_sft_warmup()
    step_grpo_training()
    eval_results = step_evaluate()
    step_plots(eval_results)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Final results written to {RESULTS_FILE}")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()