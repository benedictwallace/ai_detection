import os
import copy
import torch
import torch.nn.functional as F

from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from dotenv import load_dotenv
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

load_dotenv()

BASE_MODEL   = os.getenv("PARAPHRASER_BASE_MODEL", "google/flan-t5-large")
CKPT_DIR     = Path(os.getenv("PARAPHRASER_CHECKPOINT_DIR", "checkpoints/paraphraser"))
MAX_TOKENS   = int(os.getenv("MAX_TOKENS", 512))
N_CANDIDATES = int(os.getenv("N_CANDIDATES", 8))
LR           = float(os.getenv("LEARNING_RATE", 2e-5))

PROMPT_TEMPLATE = "Rewrite this in a casual, conversational tone: {text}"

class Paraphraser:

    def __init__(self, model_name: str = BASE_MODEL):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        base = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

        # Freeze encoder
        for param in base.encoder.parameters():
            param.requires_grad = False

        # Wrap with LoRA: only trains small adapter weights
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"],
        )
        self.model = get_peft_model(base, lora_config)
        self.model.print_trainable_parameters()

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=LR
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.scheduler = None
        print(f"Paraphraser loaded: {model_name} on {self.device}")

    def setup_scheduler(self, total_steps: int) -> None:
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )

    def generate(self, text: str, n: int = N_CANDIDATES) -> list[str]:
        prompt = PROMPT_TEMPLATE.format(text=text)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_TOKENS,
            truncation=True,
        ).to(self.device)

        text_only  = self.tokenizer(text, return_tensors="pt").input_ids.shape[1]
        min_length = max(8, int(text_only * 0.25))
        # Floor of 64 so short inputs still get room to breathe
        max_length = max(64, int(text_only * 2.0))

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                num_return_sequences=n,
                do_sample=True,
                temperature=1,
                top_p=0.98,
                top_k=0,
                repetition_penalty=1.2,
                min_new_tokens=min_length,
                max_new_tokens=max_length,
                early_stopping=False,
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


    def train_step_grpo(self, original: str, scored: list[dict],
                        kl_coeff: float = 0.02) -> float:
        """
        GRPO: train on all scored candidates simultaneously.
        Each candidate is weighted by its reward normalised across the group.
        A KL penalty vs the frozen ref distribution prevents huge drift.

        Returns the mean *raw* policy loss (cross-entropy) across candidates,
        which is always positive and meaningful for monitoring. Returns 0.0
        only when no optimizer step was taken.
        """
        self.model.train()
        
        scored = [r for r in scored if r["reward"] > 0]
        if len(scored) < 2:
            return None

        prompt = PROMPT_TEMPLATE.format(text=original)
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            max_length=MAX_TOKENS, truncation=True,
        ).to(self.device)

        # Normalise rewards across the group (GRPO core idea)
        rewards = [r["reward"] for r in scored]
        mean_r  = sum(rewards) / len(rewards)
        std_r   = (sum((x - mean_r) ** 2 for x in rewards) / len(rewards)) ** 0.5
        if std_r < 1e-7:
            return None

        self.optimizer.zero_grad()
        losses = []
        raw_policy_losses = []  # for monitoring; always positive

        for r in scored:
            norm_reward = (r["reward"] - mean_r) / (std_r + 1e-8)
            norm_reward = max(-1.5, min(1.5, norm_reward))

            labels = self.tokenizer(
                r["text"], return_tensors="pt",
                max_length=MAX_TOKENS, truncation=True,
            ).input_ids.to(self.device)
            labels[labels == self.tokenizer.pad_token_id] = -100

            # Policy forward pass with adapters ON
            policy_out = self.model(**inputs, labels=labels)
            policy_loss = policy_out.loss
            policy_logits = policy_out.logits

            # Reference forward pass with adapters OFF (same base weights).
            # No deepcopy needed; PEFT toggles the LoRA contribution.
            with torch.no_grad():
                with self.model.disable_adapter():
                    ref_logits = self.model(**inputs, labels=labels).logits

            # Proper token-level KL divergence between policy and reference output distributions. Penalises drift in either direction.
            label_mask = (labels != -100).unsqueeze(-1).float()
            kl_per_token = F.kl_div(
                F.log_softmax(policy_logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction="none",
            ).sum(dim=-1, keepdim=True)
            kl_penalty = (kl_per_token * label_mask).sum() / label_mask.sum().clamp(min=1.0)

            # minimise NLL for high-reward candidates,
            candidate_loss = -policy_loss * norm_reward + kl_coeff * kl_penalty

            if not torch.isnan(candidate_loss) and not torch.isinf(candidate_loss):
                losses.append(candidate_loss)
                raw_policy_losses.append(policy_loss.item())

        if not losses:
            return None

        total_loss = torch.stack(losses).mean()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        # Return both:
        #   grpo_loss: the actual objective being minimised. Can be negative when high-reward candidates dominate the gradient.
        #   raw_loss: mean cross-entropy NLL across the group. Always positive. Useful as a sanity check that the model still finds its candidates plausible.
        return {
            "grpo_loss": total_loss.item(),
            "raw_loss":  sum(raw_policy_losses) / len(raw_policy_losses),
        }

    def save(self, epoch: int) -> None:
        path = CKPT_DIR / f"epoch_{epoch}"
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Checkpoint saved: {path}")

    def load(self, checkpoint_path: str) -> None:
        base = T5ForConditionalGeneration.from_pretrained(BASE_MODEL).to(self.device)
        for param in base.encoder.parameters():
            param.requires_grad = False

        self.model = PeftModel.from_pretrained(
            base, checkpoint_path, is_trainable=True
        ).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)

        trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        if not trainable:
            raise RuntimeError(
                "No trainable parameters after loading LoRA checkpoint. "
                "Check that the checkpoint contains adapter weights and that "
                "is_trainable=True was passed to PeftModel.from_pretrained."
            )

        self.optimizer = torch.optim.AdamW(trainable, lr=LR)
        self.scaler    = torch.cuda.amp.GradScaler(enabled=False)
        print(f"Checkpoint loaded: {checkpoint_path}")


if __name__ == "__main__":
    p = Paraphraser()

    test = "The cat sat on the mat."
    print(f"\n=== GENERATION SANITY CHECK ===")
    print(f"Original: {test}")
    candidates = p.generate(test, n=4)
    if not candidates:
        print("WARNING: model produced no candidates, generation is broken")
    else:
        for i, c in enumerate(candidates, 1):
            print(f"  [{i}] {c}")

    test2 = (
        "The utilization of artificial intelligence in modern healthcare "
        "systems has demonstrated significant improvements in diagnostic "
        "accuracy and patient outcomes across multiple clinical domains."
    )
    print(f"\nOriginal: {test2}")
    for i, c in enumerate(p.generate(test2, n=4), 1):
        print(f"  [{i}] {c}")