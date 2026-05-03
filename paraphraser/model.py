import os
import torch
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from dotenv import load_dotenv
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import copy

load_dotenv()

BASE_MODEL   = os.getenv("PARAPHRASER_BASE_MODEL", "google/flan-t5-large")
CKPT_DIR     = Path(os.getenv("PARAPHRASER_CHECKPOINT_DIR", "checkpoints/paraphraser"))
MAX_TOKENS   = int(os.getenv("MAX_TOKENS", 512))
N_CANDIDATES = int(os.getenv("N_CANDIDATES", 8))
LR           = float(os.getenv("LEARNING_RATE", 2e-5))


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
            target_modules=["q", "v"],  # attention query/value projections
        )
        self.model = get_peft_model(base, lora_config)
        self.model.print_trainable_parameters()  # sanity check: should be around 1-2%

        # Frozen reference model for KL penalty, keeps output from drifting
        import copy
        self.ref_model = copy.deepcopy(base)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

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
        prompt = f"paraphrase: {text}"
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_TOKENS,
            truncation=True,
        ).to(self.device)

        text_only  = self.tokenizer(text, return_tensors="pt").input_ids.shape[1]
        min_length = max(8, int(text_only * 0.5))
        max_length = int(text_only * 2.0)

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                num_return_sequences=n,
                do_sample=True,
                temperature=1.3,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.3,
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
        A KL penalty vs the frozen ref model prevents catastrophic drift.
        """
        self.model.train()

        prompt = f"paraphrase: {original}"
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            max_length=MAX_TOKENS, truncation=True,
        ).to(self.device)

        # Normalise rewards across the group (GRPO core idea)
        rewards = [r["reward"] for r in scored]
        mean_r  = sum(rewards) / len(rewards)
        std_r   = (sum((x - mean_r) ** 2 for x in rewards) / len(rewards)) ** 0.5
        if std_r < 1e-6:
            return 0.0  # all identical rewards, skip

        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for r in scored:
            norm_reward = (r["reward"] - mean_r) / (std_r + 1e-8)

            labels = self.tokenizer(
                r["text"], return_tensors="pt",
                max_length=MAX_TOKENS, truncation=True,
            ).input_ids.to(self.device)
            labels[labels == self.tokenizer.pad_token_id] = -100

            # Policy loss weighted by normalised reward
            policy_loss = self.model(**inputs, labels=labels).loss

            # KL penalty: how far policy has drifted from frozen ref model
            with torch.no_grad():
                ref_loss = self.ref_model(**inputs, labels=labels).loss
            kl_penalty = torch.clamp(policy_loss - ref_loss, min=0.0)

            candidate_loss = policy_loss * norm_reward + kl_coeff * kl_penalty
            total_loss = total_loss + candidate_loss

        total_loss = total_loss / len(scored)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            self.optimizer.zero_grad()
            return 0.0

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return total_loss.item()

    def train_step_contrastive(self, original: str, winner: str, loser: str, 
                                winner_reward: float, loser_reward: float) -> float:
        prompt = f"paraphrase: {original}"
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

        # Maximise good output, minimise bad output
        combined = loss_good * winner_reward - loss_bad * (1 - loser_reward) * 0.5
        combined.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return loss_good.item()

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

        # Mixed precision forward pass uses float16 on GPU, float32 on CPU
        with torch.cuda.amp.autocast(enabled=False):
            loss          = self.model(**inputs, labels=labels).loss
            weighted_loss = loss * max(reward, 1e-8)

        if torch.isnan(loss) or torch.isinf(loss):
            self.optimizer.zero_grad()
            return 0.0
        
        # Scaler handles gradient scaling to prevent underflow in float16
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
        # save_pretrained on a peft model saves only the adapter weights
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Checkpoint saved: {path}")

    def load(self, checkpoint_path: str) -> None:
        base = T5ForConditionalGeneration.from_pretrained(BASE_MODEL).to(self.device)
        for param in base.encoder.parameters():
            param.requires_grad = False
        self.model = PeftModel.from_pretrained(base, checkpoint_path).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=LR
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        self.ref_model = copy.deepcopy(base)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

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

    # Longer test
    test2 = (
        "The utilization of artificial intelligence in modern healthcare "
        "systems has demonstrated significant improvements in diagnostic "
        "accuracy and patient outcomes across multiple clinical domains."
    )
    print(f"\nOriginal: {test2}")
    for i, c in enumerate(p.generate(test2, n=4), 1):
        print(f"  [{i}] {c}")