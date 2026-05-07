"""
sft_warmup.py

Supervised fine-tuning warm-up for the paraphraser before GRPO.

Loads (original, rewrite) pairs from data/processed/sft_pairs.jsonl and
trains the LoRA adapters with plain teacher-forcing cross-entropy. Saves
to checkpoints/paraphraser/sft_init/ so training_loop.py can load it as
the GRPO starting point.

Run from project root:
    python -m paraphraser.sft_warmup
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
import logging
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
from tqdm import tqdm

from paraphraser.model import Paraphraser, PROMPT_TEMPLATE, MAX_TOKENS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

SFT_FILE   = Path("data/processed/sft_pairs.jsonl")
SFT_CKPT   = Path("checkpoints/paraphraser/sft_init")

# Smaller batch size so this fits on consumer GPUs (8GB VRAM is enough).
# Bump SFT_BATCH_SIZE in env if you have more headroom.
BATCH_SIZE = int(os.getenv("SFT_BATCH_SIZE", 2))
EPOCHS     = int(os.getenv("SFT_EPOCHS", 3))   # 3 epochs for ~116 pairs
LR         = float(os.getenv("SFT_LR", 2e-4))
SEED       = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """
    Each sample tokenises the canonical PROMPT_TEMPLATE (input) and the
    rewrite (label). Uses dynamic padding inside the collate to avoid
    wasting compute on always-max-length tensors.
    """

    def __init__(self, pairs: list[dict], tokenizer):
        self.pairs     = pairs
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        pair    = self.pairs[idx]
        prompt  = PROMPT_TEMPLATE.format(text=pair["original"])
        target  = pair["rewrite"]

        enc_in = self.tokenizer(
            prompt, max_length=MAX_TOKENS, truncation=True,
            return_tensors="pt",
        )
        enc_lbl = self.tokenizer(
            target, max_length=MAX_TOKENS, truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      enc_in.input_ids.squeeze(0),
            "attention_mask": enc_in.attention_mask.squeeze(0),
            "labels":         enc_lbl.input_ids.squeeze(0),
        }


def collate_fn(batch, pad_token_id):
    """
    Dynamic padding collate. Pads to max-in-batch rather than max-overall,
    which roughly halves training time on this dataset since most examples
    are well below MAX_TOKENS.
    """
    max_in  = max(b["input_ids"].size(0)      for b in batch)
    max_lbl = max(b["labels"].size(0)         for b in batch)

    input_ids = torch.full((len(batch), max_in), pad_token_id, dtype=torch.long)
    attn_mask = torch.zeros((len(batch), max_in), dtype=torch.long)
    labels    = torch.full((len(batch), max_lbl), -100, dtype=torch.long)  # -100 = ignore in loss

    for i, b in enumerate(batch):
        n_in  = b["input_ids"].size(0)
        n_lbl = b["labels"].size(0)
        input_ids[i, :n_in]  = b["input_ids"]
        attn_mask[i, :n_in]  = b["attention_mask"]
        labels[i, :n_lbl]    = b["labels"]

    # Mask pad tokens that snuck into labels (shouldn't, but defensive)
    labels[labels == pad_token_id] = -100

    return {
        "input_ids":      input_ids,
        "attention_mask": attn_mask,
        "labels":         labels,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def load_pairs(path: Path) -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "original" in obj and "rewrite" in obj:
                    pairs.append(obj)
            except json.JSONDecodeError:
                continue
    return pairs


def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
          val_frac: float = 0.10):
    if not SFT_FILE.exists():
        raise FileNotFoundError(
            f"{SFT_FILE} not found. Run generate_sft_data.py first."
        )

    pairs = load_pairs(SFT_FILE)
    logger.info(f"Loaded {len(pairs)} SFT pairs from {SFT_FILE}")
    if len(pairs) < 30:
        logger.warning("Very few SFT pairs - results will be limited.")

    random.seed(SEED)
    random.shuffle(pairs)
    n_val      = max(2, int(len(pairs) * val_frac))
    val_pairs  = pairs[:n_val]
    train_pairs = pairs[n_val:]
    logger.info(f"SFT split: train={len(train_pairs)} val={len(val_pairs)}")

    paraphraser = Paraphraser()
    # Override LR specifically for the SFT phase
    for pg in paraphraser.optimizer.param_groups:
        pg["lr"] = LR
    logger.info(f"SFT LR: {LR}, batch_size: {batch_size}, epochs: {epochs}")

    pad_id   = paraphraser.tokenizer.pad_token_id
    train_ds = SFTDataset(train_pairs, paraphraser.tokenizer)
    val_ds   = SFTDataset(val_pairs,   paraphraser.tokenizer)

    collate = lambda b: collate_fn(b, pad_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate)

    device = paraphraser.device
    logger.info(f"Training on: {device}")

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        paraphraser.model.train()
        train_losses = []

        bar = tqdm(train_loader, desc=f"SFT epoch {epoch}", ncols=120)
        for batch in bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            paraphraser.optimizer.zero_grad()
            out  = paraphraser.model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(paraphraser.model.parameters(), max_norm=1.0)
            paraphraser.optimizer.step()

            train_losses.append(loss.item())
            avg_recent = sum(train_losses[-20:]) / len(train_losses[-20:])
            bar.set_postfix(loss=f"{avg_recent:.4f}")

        # Validation
        paraphraser.model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_losses.append(paraphraser.model(**batch).loss.item())

        train_avg = sum(train_losses) / len(train_losses) if train_losses else 0.0
        val_avg   = sum(val_losses) / len(val_losses) if val_losses else 0.0
        logger.info(
            f"Epoch {epoch}/{epochs} | train_loss={train_avg:.4f} | val_loss={val_avg:.4f}"
        )

        # Save best on val loss so we don't ship an over-fit checkpoint
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            SFT_CKPT.mkdir(parents=True, exist_ok=True)
            paraphraser.model.save_pretrained(SFT_CKPT)
            paraphraser.tokenizer.save_pretrained(SFT_CKPT)
            logger.info(f"New best val loss; checkpoint saved to {SFT_CKPT}")

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"SFT checkpoint at: {SFT_CKPT}")

    # Sanity check generations from the trained model on held-out val
    logger.info("=" * 60)
    logger.info("Sanity check generations from SFT model:")
    logger.info("=" * 60)
    for i, sample in enumerate(val_pairs[:3], 1):
        logger.info(f"\nSample {i}")
        logger.info(f"  ORIG:    {sample['original'][:160]}")
        logger.info(f"  TARGET:  {sample['rewrite'][:160]}")
        cands = paraphraser.generate(sample["original"], n=2)
        for j, c in enumerate(cands, 1):
            logger.info(f"  GEN [{j}]: {c[:160]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size)