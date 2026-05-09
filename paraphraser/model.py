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
LR           = float(os.getenv("LEARNING_RATE", 2e-4))

# Single canonical prompt used everywhere: generation, scoring conditioning,
# and every training step. Mismatched prompts attach reward signal to the
# wrong context and waste training.
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

        # No separate ref_model copy. Instead we use PEFT's disable_adapter()
        # context to get reference logits from the same base weights with
        # adapters turned off. Saves ~3 GB VRAM and avoids the deepcopy bug
        # where ref_model would inherit the LoRA-injected base.

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=LR
        )
        self.scaler    = torch.cuda.amp.GradScaler(enabled=False)
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
                temperature=1.1,
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
                        kl_coeff: float = 0.1) -> float:
        """
        GRPO: train on all scored candidates simultaneously.
        Each candidate is weighted by its reward normalised across the group.
        A KL penalty vs the frozen ref distribution prevents catastrophic drift.

        Returns the mean *raw* policy loss (cross-entropy) across candidates,
        which is always positive and meaningful for monitoring. Returns 0.0
        only when no optimizer step was taken.
        """
        self.model.train()

        # Remove bottom half of candiates

        TOP_K_FOR_GRPO = int(N_CANDIDATES/2)
        if len(scored) > TOP_K_FOR_GRPO:
            scored = scored[:TOP_K_FOR_GRPO]

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
            norm_reward = max(-3.0, min(3.0, norm_reward))

            labels = self.tokenizer(
                r["text"], return_tensors="pt",
                max_length=MAX_TOKENS, truncation=True,
            ).input_ids.to(self.device)
            labels[labels == self.tokenizer.pad_token_id] = -100

            # Policy forward pass with adapters ON
            policy_out    = self.model(**inputs, labels=labels)
            policy_loss   = policy_out.loss
            policy_logits = policy_out.logits

            # Reference forward pass with adapters OFF (same base weights).
            # No deepcopy needed; PEFT toggles the LoRA contribution.
            with torch.no_grad():
                with self.model.disable_adapter():
                    ref_logits = self.model(**inputs, labels=labels).logits

            # Proper token-level KL divergence between policy and reference
            # output distributions. Penalises drift in either direction,
            # unlike the previous clamp(policy_loss - ref_loss, min=0) hack.
            label_mask = (labels != -100).unsqueeze(-1).float()
            kl_per_token = F.kl_div(
                F.log_softmax(policy_logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction="none",
            ).sum(dim=-1, keepdim=True)
            kl_penalty = (kl_per_token * label_mask).sum() / label_mask.sum().clamp(min=1.0)

            # Sign convention: minimise NLL for high-reward candidates,
            # maximise it for low-reward ones. The previous +policy_loss * norm_reward
            # was training the model to AVOID high-reward outputs.
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
        #   grpo_loss: the actual objective being minimised. Can be negative
        #              when high-reward candidates dominate the gradient.
        #              Trending more-negative = model being reinforced toward winners.
        #   raw_loss:  mean cross-entropy NLL across the group. Always positive.
        #              Useful as a sanity check that the model still finds its
        #              candidates plausible.
        return {
            "grpo_loss": total_loss.item(),
            "raw_loss":  sum(raw_policy_losses) / len(raw_policy_losses),
        }

    def train_step_contrastive(self, original: str, winner: str, loser: str,
                                winner_reward: float, loser_reward: float) -> float:
        """
        Pairwise contrastive update. Currently unused by the training loop
        but kept available. Uses the same canonical prompt as the rest of
        the pipeline.
        """
        prompt = PROMPT_TEMPLATE.format(text=original)
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                max_length=MAX_TOKENS, truncation=True).to(self.device)

        def get_loss(text):
            labels = self.tokenizer(text, return_tensors="pt",
                                    max_length=MAX_TOKENS, truncation=True).input_ids.to(self.device)
            labels[labels == self.tokenizer.pad_token_id] = -100
            return self.model(**inputs, labels=labels).loss

        self.optimizer.zero_grad()
        loss_good = get_loss(winner)
        loss_bad  = get_loss(loser)

        # Bound the negative term so the loss can't run away to -inf
        margin = torch.clamp(loss_bad - loss_good, max=2.0)
        combined = loss_good * winner_reward - margin * (1 - loser_reward) * 0.5

        combined.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return loss_good.item()

    def train_step(self, original: str, rewrite: str, reward: float) -> float:
        """
        Single-sample reward-weighted update. Currently unused by the
        training loop but kept available. Uses the canonical prompt.
        """
        self.model.train()

        prompt = PROMPT_TEMPLATE.format(text=original)
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

        with torch.cuda.amp.autocast(enabled=False):
            loss          = self.model(**inputs, labels=labels).loss
            weighted_loss = loss * max(reward, 1e-8)

        if torch.isnan(loss) or torch.isinf(loss):
            self.optimizer.zero_grad()
            return 0.0

        self.scaler.scale(weighted_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler:
            self.scheduler.step()

        return loss.item()

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

        # Pass is_trainable=True so PEFT keeps the LoRA adapters in train mode.
        # Without this they're loaded in inference mode (requires_grad=False on
        # all adapter weights), which causes AdamW to receive an empty
        # parameter list and crash.
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