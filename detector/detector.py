"""
detector/detector.py

Detector wrapper for the paraphraser scoring.
Uses desklib/ai-text-detector-v1.01

Usage:
    from detector.detector import Detector

    detector = Detector()
    score    = detector.score("Some text here")      # float 0-1, higher = more human
    scores   = detector.score_batch(["text1", ...])  # list of floats
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DETECTOR_MODEL      = os.getenv("DETECTOR_MODEL", "desklib/ai-text-detector-v1.01")
DETECTOR_MAX_LEN    = int(os.getenv("DETECTOR_MAX_LEN", 512))
DETECTOR_BATCH_SIZE = int(os.getenv("DETECTOR_BATCH_SIZE", 16))


# ---------------------------------------------------------------------------
# Plain nn.Module
# ---------------------------------------------------------------------------

class DesklibClassifier(nn.Module):
    """
    Recreates the desklib architecture as a plain nn.Module.
    Loads the pretrained transformer backbone via AutoModel,
    then adds the same mean-pool + linear classifier head.
    Avoids all PreTrainedModel subclass compatibility issues.
    """

    def __init__(self, model_name: str, hidden_size: int):
        super().__init__()
        # Build structure only, weights loaded separately via load_state_dict
        config          = AutoConfig.from_pretrained(model_name)
        self.backbone   = AutoModel.from_config(config)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs  = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden   = outputs.last_hidden_state                          # (B, T, H)
        mask_exp = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        pooled   = torch.sum(hidden * mask_exp, dim=1) / \
                   torch.clamp(mask_exp.sum(dim=1), min=1e-9)         # (B, H)
        logits   = self.classifier(pooled)                            # (B, 1)
        return logits


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Detector:
    """
    Frozen oracle detector wrapping desklib/ai-text-detector-v1.01.
    Always returns P(human) as a float in [0, 1].
    Higher = text looks more human-written.
    Weights are frozen
    """

    def __init__(self, model_name: str = DETECTOR_MODEL):
        self.model_name  = model_name
        self.max_len     = DETECTOR_MAX_LEN
        self.batch_size  = DETECTOR_BATCH_SIZE
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading detector: {model_name} on {self.device}")
        self._load_model()
        logger.info("Detector loaded and frozen.")

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        config         = AutoConfig.from_pretrained(self.model_name)
        hidden_size    = config.hidden_size

        # Build model structure without loading weights yet
        self.model = DesklibClassifier(self.model_name, hidden_size).to(self.device)

        # Download checkpoint and remap weight keys to match our structure.
        ckpt_path  = hf_hub_download(self.model_name, filename="model.safetensors")
        state_dict = load_file(ckpt_path)

        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                remapped["backbone." + k[len("model."):]] = v
            else:
                remapped[k] = v

        self.model.load_state_dict(remapped, strict=True)
        logger.info("Weights loaded and remapped successfully.")

        # freeze
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    # ------------------------------------------------------------------
    # Public scoring interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(self, text: str) -> float:
        """
        Score a single piece of text.
        Returns P(human) in [0, 1]. Higher = more human-like.
        """
        return self.score_batch([text])[0]

    @torch.no_grad()
    def score_batch(self, texts: list[str]) -> list[float]:
        """
        Score a list of texts efficiently.
        Splits into sub-batches of DETECTOR_BATCH_SIZE to avoid OOM.
        Returns a list of P(human) floats in [0, 1].
        """
        all_scores = []
        for i in range(0, len(texts), self.batch_size):
            batch  = texts[i : i + self.batch_size]
            scores = self._score_batch_internal(batch)
            all_scores.extend(scores)
        return all_scores

    def _score_batch_internal(self, texts: list[str]) -> list[float]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(self.device)

        logits  = self.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        ).squeeze(-1)                          # (B,)

        # Higher percent = more AI. Invert sigmoid to get P(human).
        p_ai    = torch.sigmoid(logits)
        p_human = 1.0 - p_ai
        return p_human.cpu().tolist()

    def label(self, text: str, threshold: float = 0.5) -> str:
        """Returns 'human' or 'ai' string label."""
        return "human" if self.score(text) >= threshold else "ai"

    def evasion_rate(self, texts: list[str], threshold: float = 0.5) -> float:
        """
        Fraction of texts the detector classifies as human.
        Use after each paraphraser training epoch to track progress.
        """
        scores = self.score_batch(texts)
        evaded = sum(1 for s in scores if s >= threshold)
        rate   = evaded / len(scores) if scores else 0.0
        logger.info(
            f"Evasion rate: {evaded}/{len(scores)} = {rate:.1%} "
            f"(threshold={threshold})"
        )
        return rate


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    detector = Detector()

    test_cases = [
        (
            "AI text",
            "The utilization of artificial intelligence in modern healthcare "
            "systems has demonstrated significant improvements in diagnostic "
            "accuracy and patient outcomes across multiple clinical domains.",
        ),
        (
            "Human text",
            "went to the shop yesterday and they were out of the good bread "
            "again, so i just grabbed some crackers. not ideal but whatever",
        ),
    ]

    print("\n" + "=" * 55)
    print(f"Model: {DETECTOR_MODEL}")
    print("=" * 55)
    for lbl, text in test_cases:
        score = detector.score(text)
        pred  = detector.label(text)
        print(f"\n[{lbl}]")
        print(f"  Text:       {text[:80]}...")
        print(f"  P(human):   {score:.4f}")
        print(f"  Prediction: {pred}")
    print("=" * 55)