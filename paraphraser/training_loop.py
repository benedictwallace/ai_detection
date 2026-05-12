import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import json
import random
import logging
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import torch

from paraphraser.model import Paraphraser
from paraphraser.score import score_candidates
from detector.detector import Detector

load_dotenv()

# Quiet down library logs
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_FILE     = Path("data/processed/train.jsonl")
VAL_FILE       = Path("data/processed/val.jsonl")
EPOCHS         = int(os.getenv("EPOCHS", 3))
N_CANDIDATES   = int(os.getenv("N_CANDIDATES", 8))

THRESHOLD      = float(os.getenv("REWARD_THRESHOLD", 0.2))
EVASION_TARGET = float(os.getenv("EVASION_TARGET", 70)) / 100
SEED           = 42

# Make sure data/ exists before the FileHandler tries to open a log inside it
Path("data").mkdir(exist_ok=True)

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


# ---------------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------------

def run_epoch(paraphraser, detector, texts, epoch):
    random.seed(SEED + epoch)
    random.shuffle(texts)

    trained_on = 0
    skipped    = 0
    losses     = []

    skip_reasons = {
        "no_candidates": 0,
        "no_scored":     0,
        "zero_loss":     0,
    }

    total_kept    = 0   # sum of len(scored) over non-empty samples
    samples_kept = 0 # count of samples with survivors

    bar = tqdm(texts, desc=f"Epoch {epoch}", unit="sample", ncols=160)

    for text in bar:

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t = time.perf_counter()
        candidates = paraphraser.generate(text, n=N_CANDIDATES)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_gen = time.perf_counter() - t


        if not candidates:
            skipped += 1
            skip_reasons["no_candidates"] += 1
            bar.set_postfix(trained=trained_on, skipped=skipped, loss="n/a")
            continue

        # score_candidates already drops too-similar and too-short candidates,
        # so the duplicate pre-filter that used to live here has been removed.
        t = time.perf_counter()
        scored = score_candidates(text, candidates, detector=detector)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_score = time.perf_counter() - t

        print(f"[loop timing] generate={t_gen:.2f}s  score={t_score:.2f}s  n_cands={len(candidates)}  n_scored={len(scored)}")
        
        if not scored:
            skipped += 1
            skip_reasons["no_scored"] += 1
            continue

        total_kept   += len(scored)
        samples_kept += 1

        # GRPO trains on ALL scored candidates (including low-reward ones)
        # so normalised rewards have signed spread. Returns mean raw policy
        # loss (always positive) when an optimizer step actually ran, or 0.0
        # when it bailed before stepping.
        result = paraphraser.train_step_grpo(text, scored)

        if result is not None:
            losses.append(result)  # dict with grpo_loss and raw_loss
            trained_on += 1
        else:
            skipped += 1
            skip_reasons["zero_loss"] += 1

        # Show both losses in the progress bar
        avg_group = total_kept / samples_kept if samples_kept else 0
        recent = losses[-10:]
        if recent:
            avg_grpo = sum(r["grpo_loss"] for r in recent) / len(recent)
            avg_raw  = sum(r["raw_loss"]  for r in recent) / len(recent)
            bar.set_postfix(
                trained=trained_on, **skip_reasons,
                grpo=f"{avg_grpo:+.4f}", raw=f"{avg_raw:.4f}",
            )
        else:
            bar.set_postfix(trained=trained_on, **skip_reasons, grpo="n/a", raw="n/a", grp=f"{avg_group:.1f}")


    return trained_on, skipped, losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(paraphraser, detector, val_texts, epoch):
    rewrites        = []
    detector_scores = []

    for text in val_texts:
        candidates = paraphraser.generate(text, n=4)
        if candidates:
            scored = score_candidates(text, candidates, detector=detector)
            if scored:
                rewrites.append(scored[0]["text"])
                detector_scores.append(scored[0]["detector"])
            else:
                rewrites.append(text)
                detector_scores.append(detector.score(text))
        else:
            rewrites.append(text)
            detector_scores.append(detector.score(text))

    # Compute evasion rate from cached scores instead of re-running the detector
    evaded = sum(1 for s in detector_scores if s >= 0.5)
    rate   = evaded / len(detector_scores) if detector_scores else 0.0

    avg_human_score    = sum(detector_scores) / len(detector_scores) if detector_scores else 0.0
    original_scores    = detector.score_batch(val_texts)
    avg_original_score = sum(original_scores) / len(original_scores)

    logger.info(f"Epoch {epoch} | val evasion rate: {rate:.1%} (target: {EVASION_TARGET:.1%})")
    logger.info(
        f"Epoch {epoch} | avg P(human) original: {avg_original_score:.4f} | "
        f"rewrite: {avg_human_score:.4f} | delta: {avg_human_score - avg_original_score:+.4f}"
    )
    logger.info(
        f"Epoch {epoch} | detector score distribution: "
        f"min={min(detector_scores):.3f} "
        f"max={max(detector_scores):.3f} "
        f"median={sorted(detector_scores)[len(detector_scores)//2]:.3f}"
    )

    return rate, avg_human_score


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
    val_texts   = load_texts(VAL_FILE)

    max_samples = int(os.getenv("MAX_TRAIN_SAMPLES", 0))
    if max_samples > 0:
        train_texts = train_texts[:max_samples]
        logger.info(f"Limited to {max_samples} training samples")

    logger.info(f"Train: {len(train_texts)} | Val: {len(val_texts)}")

    logger.info("Loading models...")
    detector    = Detector()
    paraphraser = Paraphraser()

    # CHECKPOINT PATH ===============================================================
    sft_path = Path("checkpoints/paraphraser/sft_init")
    if sft_path.exists():
        paraphraser.load(str(sft_path))
        logger.info(f"Loaded SFT warm-start from {sft_path}")
    else:
        logger.warning(f"No SFT checkpoint at {sft_path}, starting from base model")    
    # ===============================================================


    # Estimate scheduler size based on expected real step count.
    # We empirically expect ~30-50% of samples to actually take an optimizer
    # step (others are skipped by no_candidates / no_scored / zero_loss).
    # Sizing for ~40% gives the LR schedule a closer-to-correct decay curve.
    expected_step_fraction = 0.4
    total_steps = max(1, int(EPOCHS * len(train_texts) * expected_step_fraction))
    paraphraser.setup_scheduler(total_steps)
    logger.info(f"LR scheduler sized for ~{total_steps} steps "
                f"(EPOCHS={EPOCHS} * samples={len(train_texts)} * frac={expected_step_fraction})")

    # Use the same val slice for baseline and per-epoch evaluation so the
    # numbers are directly comparable across training.
    eval_slice = val_texts[:100]
    baseline = measure_baseline(detector, eval_slice)

    best_evasion = baseline
    history      = []

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Epoch {epoch}/{EPOCHS}")
        logger.info(f"{'=' * 50}")

        trained_on, skipped, epoch_losses = run_epoch(
            paraphraser, detector, train_texts, epoch
        )

        if epoch_losses:
            avg_grpo = sum(r["grpo_loss"] for r in epoch_losses) / len(epoch_losses)
            avg_raw  = sum(r["raw_loss"]  for r in epoch_losses) / len(epoch_losses)
        else:
            avg_grpo = 0.0
            avg_raw  = 0.0

        evasion, avg_human = evaluate(paraphraser, detector, eval_slice, epoch)

        logger.info(
            f"Epoch {epoch} complete | "
            f"trained_on={trained_on} | skipped={skipped} | "
            f"avg_grpo_loss={avg_grpo:+.4f} | avg_raw_loss={avg_raw:.4f} | "
            f"avg_human_score={avg_human:.4f}"
        )

        history.append({
            "epoch":           epoch,
            "trained_on":      trained_on,
            "skipped":         skipped,
            "avg_grpo_loss":   round(avg_grpo, 4),
            "avg_raw_loss":    round(avg_raw,  4),
            "evasion":         round(evasion,  4),
            "avg_human_score": round(avg_human, 4),
    })

        paraphraser.save(epoch)

        if evasion > best_evasion:
            best_evasion = evasion
            best_path    = Path("checkpoints/paraphraser/best")
            best_path.mkdir(parents=True, exist_ok=True)
            paraphraser.model.save_pretrained(best_path)
            paraphraser.tokenizer.save_pretrained(best_path)
            logger.info(f"New best model saved ({evasion:.1%})")

        if evasion >= EVASION_TARGET:
            logger.info(f"Evasion target reached ({evasion:.1%}). Stopping early.")
            break

    history_path = Path("data/train_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline":     round(baseline, 4),
            "epochs":       history,
            "best_evasion": round(best_evasion, 4),
        }, f, indent=2)

    logger.info(f"Training complete.")
    logger.info(f"Baseline evasion: {baseline:.1%}")
    logger.info(f"Best evasion: {best_evasion:.1%}")
    logger.info(f"History saved to: {history_path}")


if __name__ == "__main__":
    train()