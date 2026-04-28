import os
import torch
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dotenv import load_dotenv

load_dotenv()

BASE_MODEL   = os.getenv("PARAPHRASER_BASE_MODEL", "google/flan-t5-base")
CKPT_DIR     = Path(os.getenv("PARAPHRASER_CHECKPOINT_DIR", "checkpoints/paraphraser"))
MAX_TOKENS   = int(os.getenv("MAX_TOKENS", 512))
N_CANDIDATES = int(os.getenv("N_CANDIDATES", 8))
LR           = float(os.getenv("LEARNING_RATE", 2e-5))


class Paraphraser:

    def __init__(self, model_name: str = BASE_MODEL):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model     = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR)
        # GradScaler for mixed precision disabled automatically on CPU
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        print(f"Paraphraser loaded: {model_name} on {self.device}")

    def generate(self, text: str, n: int = N_CANDIDATES) -> list[str]:
        prompt  = f"paraphrase: {text}"
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_TOKENS,
            truncation=True,
        ).to(self.device)

        input_length = encoded["input_ids"].shape[1]
        min_length   = max(10, int(input_length * 0.7))
        max_length   = int(input_length * 1.3)

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                num_return_sequences=n,
                num_beam_groups=n,      # diverse beam search
                num_beams=n,
                diversity_penalty=0.8,  # force varied outputs
                min_new_tokens=min_length,
                max_new_tokens=max_length,
                early_stopping=True,
            )

        candidates = [
            self.tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]

        seen, unique = set(), []
        for c in candidates:
            c = c.strip()
            if c and c not in seen and c != text.strip():
                seen.add(c)
                unique.append(c)

        return unique

    def train_step(self, original: str, rewrite: str, reward: float) -> float:
        self.model.train()

        prompt = f"paraphrase: {original}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_TOKENS,
            truncation=True,
        ).to(self.device)

        labels = self.tokenizer(
            rewrite,
            return_tensors="pt",
            max_length=MAX_TOKENS,
            truncation=True,
        ).input_ids.to(self.device)

        labels[labels == self.tokenizer.pad_token_id] = -100

        self.optimizer.zero_grad()

        # Mixed precision forward pass — uses float16 on GPU, float32 on CPU
        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            loss          = self.model(**inputs, labels=labels).loss
            weighted_loss = loss * reward

        # Scaler handles gradient scaling to prevent underflow in float16
        self.scaler.scale(weighted_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def save(self, epoch: int) -> None:
        path = CKPT_DIR / f"epoch_{epoch}"
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Checkpoint saved: {path}")

    def load(self, checkpoint_path: str) -> None:
        self.model     = T5ForConditionalGeneration.from_pretrained(checkpoint_path).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR)
        self.scaler    = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")
        print(f"Checkpoint loaded: {checkpoint_path}")


if __name__ == "__main__":
    p = Paraphraser()

    test = (
        "The utilization of artificial intelligence in modern healthcare "
        "systems has demonstrated significant improvements in diagnostic "
        "accuracy and patient outcomes across multiple clinical domains."
    )

    print(f"\nOriginal:\n  {test}\n")
    print("Candidates:")
    for i, c in enumerate(p.generate(test, n=4), 1):
        print(f"  [{i}] {c}")