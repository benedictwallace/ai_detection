import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"

import json
import random
import logging
from pathlib import Path
from dotenv import load_dotenv

from paraphraser.model import Paraphraser
from paraphraser.score import score_candidates, top_k
from detector.detector import Detector

load_dotenv()

# stop verbose logs
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_FILE = Path("data/processed/train.jsonl")
VAL_FILE = Path("data/processed/val.jsonl")
EPOCHS = int(os.getenv("EPOCHS", 3))
N_CANDIDATES = int(os.getenv("N_CANDIDATES", 8))
TOP_K = int(os.getenv("TOP_K", 3))
THRESHOLD = float(os.getenv("REWARD_THRESHOLD", 0.65))
EVASION_TARGET = float(os.getenv("EVASION_TARGET", 70)) / 100
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/train_log.txt", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

def log_device_info():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"VRAM free:  {torch.cuda.memory_reserved(0) / 1e9:.1f} GB reserved")
    else:
        logger.warning("CUDA not available, running on CPU.")
    return device

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_texts(path: Path) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    texts.append(json.loads(line)["text"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return texts


def run_epoch(paraphraser, detector, texts, epoch):
    random.seed(SEED + epoch)
    random.shuffle(texts)

    total = len(texts)
    trained_on = 0
    skipped = 0
    losses = []

    for i, text in enumerate(texts, 1):
        candidates = paraphraser.generate(text, n=N_CANDIDATES)

        if not candidates:
            skipped += 1
            continue

        winners = top_k(text, candidates, k=TOP_K)

        if not winners:
            skipped += 1
            continue

        for w in winners:
            loss = paraphraser.train_step(text, w["text"], w["reward"])
            losses.append(loss)
            trained_on += 1

        if i % 20 == 0 or i == total:
            avg_loss = sum(losses[-20:]) / len(losses[-20:]) if losses else 0
            logger.info(
                f"Epoch {epoch} | {i}/{total} samples | "
                f"trained_on={trained_on} skipped={skipped} | "
                f"avg_loss={avg_loss:.4f}"
            )

    return trained_on, skipped, losses


def evaluate(paraphraser, detector, val_texts, epoch):
    """
    Generate one rewrite per val sample and measure evasion rate.
    Uses top-1 candidate only for speed.
    """
    rewrites = []
    for text in val_texts:
        candidates = paraphraser.generate(text, n=4)
        if candidates:
            scored = score_candidates(text, candidates)
            if scored:
                rewrites.append(scored[0]["text"])
        else:
            rewrites.append(text)

    rate = detector.evasion_rate(rewrites)
    logger.info(f"Epoch {epoch} | val evasion rate: {rate:.1%} (target: {EVASION_TARGET:.1%})")
    return rate


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def measure_baseline(detector, texts):
    logger.info("Measuring baseline evasion rate on raw AI text...")
    rate = detector.evasion_rate(texts)
    logger.info(f"Baseline evasion rate: {rate:.1%}")
    return rate

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    log_device_info()
    logger.info("Loading data...")
    train_texts = load_texts(TRAIN_FILE)
    val_texts = load_texts(VAL_FILE)
    logger.info(f"Train: {len(train_texts)} | Val: {len(val_texts)}")

    logger.info("Loading models...")
    detector = Detector()
    paraphraser = Paraphraser()

    # Measure baseline before any training
    baseline = measure_baseline(detector, val_texts[:100])

    best_evasion = baseline
    history = []

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{EPOCHS}")
        logger.info(f"{'='*50}")

        trained_on, skipped, losses = run_epoch(
            paraphraser, detector, train_texts, epoch
        )

        avg_loss = sum(losses) / len(losses) if losses else 0
        logger.info(
            f"Epoch {epoch} complete | "
            f"trained_on={trained_on} | skipped={skipped} | "
            f"avg_loss={avg_loss:.4f}"
        )

        evasion = evaluate(paraphraser, detector, val_texts, epoch)

        history.append({
            "epoch":      epoch,
            "trained_on": trained_on,
            "skipped":    skipped,
            "avg_loss":   round(avg_loss, 4),
            "evasion":    round(evasion, 4),
        })

        # Save checkpoint each epoch
        paraphraser.save(epoch)

        # Track best
        if evasion > best_evasion:
            best_evasion = evasion
            best_path = Path("checkpoints/paraphraser/best")
            best_path.mkdir(parents=True, exist_ok=True)
            paraphraser.model.save_pretrained(best_path)
            paraphraser.tokenizer.save_pretrained(best_path)
            logger.info(f"New best model saved ({evasion:.1%})")

        # Early stop if target reached
        if evasion >= EVASION_TARGET:
            logger.info(f"Evasion target reached ({evasion:.1%}). Stopping early.")
            break

    # Save training history
    history_path = Path("data/train_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": round(baseline, 4),
            "epochs": history,
            "best_evasion": round(best_evasion, 4),
        }, f, indent=2)

    logger.info(f"\nTraining complete.")
    logger.info(f"Baseline evasion: {baseline:.1%}")
    logger.info(f"Best evasion: {best_evasion:.1%}")
    logger.info(f"History saved to: {history_path}")


if __name__ == "__main__":
    train()