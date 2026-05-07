"""
generate_sft_data.py

Generate (original, rewrite) pairs for SFT warm-up using a local Ollama model.
Filters generated rewrites by detector score and length sanity, then writes
pairs to data/processed/sft_pairs.jsonl.

Long originals (>90 words) are sentence-chunked before being sent to Ollama,
because instruction-tuned models tend to over-compress long input. Each chunk
becomes its own training pair.

Run:
    python -m paraphraser.generate_sft_data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
import time
import random
import logging
import argparse
from pathlib import Path
from difflib import SequenceMatcher

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from detector.detector import Detector

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
# Default switched to qwen2.5:14b-instruct - significantly better at sounding
# human than the 3B models, fits comfortably on a single 24GB GPU.
# Override via OLLAMA_COLLECTION_MODEL in .env if needed.
OLLAMA_MODEL     = os.getenv("OLLAMA_COLLECTION_MODEL", "qwen2.5:14b-instruct")
RETRY_LIMIT      = int(os.getenv("RETRY_LIMIT", 3))
RETRY_DELAY      = int(os.getenv("RETRY_DELAY", 5))

TRAIN_FILE       = Path("data/processed/train.jsonl")
OUTPUT_FILE      = Path("data/processed/sft_pairs.jsonl")

# Quality gates: a pair is kept only if all of these are satisfied.
DETECTOR_THRESHOLD = 0.5    # P(human) >= this; the whole point of SFT warm-up
MIN_LEN_RATIO      = 0.25   # rewrite words >= 25% of (chunked) original
MAX_SIM_RATIO      = 0.95   # rewrite must not be a near-copy
MIN_WORDS          = 5      # absolute floor

# Chunking: anything above CHUNK_TRIGGER_WORDS gets split. Chunk size is
# chosen so the model can comfortably casual-rewrite each chunk without
# resorting to summarisation.
CHUNK_TRIGGER_WORDS = 90    # passages longer than this get split
CHUNK_MAX_WORDS     = 90    # never let a chunk exceed this
CHUNK_TARGET_WORDS  = 60    # aim around this when packing sentences

# Per-call timeout. 14B models take longer than 3B models, especially on
# first warm-up. 300s gives headroom without being wasteful.
OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", 300))

PROMPT_VARIANTS = [
    "Rewrite the following passage in a casual, conversational tone, as if you were chatting with a friend. Keep all the details and roughly the same length. Do not summarise or shorten. Only output the rewrite, no preamble.\n\nPassage: {text}\n\nRewrite (similar length):",
    "Rephrase this text so it sounds like a real person wrote it informally. Cover every point from the original - do not condense or skip anything. Use everyday language. Output only the rewrite.\n\nText: {text}\n\nRewrite (full coverage):",
    "Take the following passage and reword it like a Reddit comment - natural, slightly informal. Keep every detail and a similar word count. Output only the rewrite.\n\nPassage: {text}\n\nRewrite:",
    "Rewrite this in plain everyday English at roughly the same length. Do NOT summarise. Cover every point the original makes, just in casual language. Output only the rewrite.\n\nText: {text}\n\nRewrite:",
    "Paraphrase the passage below in a relaxed, human-sounding way. Keep the same level of detail and a similar length. Avoid formal or robotic phrasing. Output only the new version.\n\nPassage: {text}\n\nParaphrase:",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

def check_ollama_ready() -> None:
    """
    Verify Ollama is reachable and the configured model is installed BEFORE
    starting the long loop. Fails fast with a clear message if not.
    """
    logger.info(f"Checking Ollama at {OLLAMA_BASE_URL}...")
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Is the server running? "
            f"Underlying error: {e}"
        )

    installed = [m["name"] for m in r.json().get("models", [])]
    if OLLAMA_MODEL not in installed:
        raise RuntimeError(
            f"Model '{OLLAMA_MODEL}' is not installed on this Ollama instance. "
            f"Installed models: {installed}. "
            f"Run `ollama pull {OLLAMA_MODEL}` first."
        )

    logger.info(f"Ollama OK. Using model: {OLLAMA_MODEL}")
    logger.info("Pre-warming model with a sample call (may take 30-90s on first run)...")

    # Pre-warm so the first real call doesn't time out while the 14B model loads
    t0 = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":   OLLAMA_MODEL,
                "prompt":  "Say hello.",
                "stream":  False,
                "options": {"num_predict": 10},
            },
            timeout=OLLAMA_TIMEOUT_S,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Pre-warm call failed. Model may not fit in VRAM, or Ollama is "
            f"misbehaving. Error: {e}"
        )
    logger.info(f"Pre-warm complete in {time.time() - t0:.1f}s. Ready to generate.")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, target_words: int = CHUNK_TARGET_WORDS,
               max_words: int = CHUNK_MAX_WORDS) -> list[str]:
    """
    Split a long passage into sentence-aware chunks of roughly target_words.
    Sentences stay intact; a chunk grows until adding the next sentence would
    push it past max_words. A single sentence longer than max_words goes
    through on its own.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text]

    chunks      = []
    current     = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent.split())

        if sent_len > max_words:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.append(sent)
            continue

        if current_len + sent_len > max_words and current:
            chunks.append(" ".join(current))
            current, current_len = [sent], sent_len
        else:
            current.append(sent)
            current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, temperature: float = 0.8,
                num_predict: int = 768) -> str | None:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.95,
            "num_predict": num_predict,
        },
    }

    for attempt in range(RETRY_LIMIT):
        try:
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT_S,
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except (requests.RequestException, ValueError) as e:
            if attempt < RETRY_LIMIT - 1:
                logger.warning(f"Ollama call failed ({e}), retrying...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Ollama call failed after {RETRY_LIMIT} attempts: {e}")
                return None
    return None


# ---------------------------------------------------------------------------
# Cleaning + filtering
# ---------------------------------------------------------------------------

def clean_rewrite(raw: str) -> str:
    if not raw:
        return ""

    text = raw.strip()

    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    lowers = text.lower()
    for prefix in ("rewrite:", "paraphrase:", "here is the rewrite:",
                   "here's the rewrite:", "here is a rewrite:",
                   "here's a casual version:", "casual version:",
                   "rewrite (similar length):", "rewrite (full coverage):"):
        if lowers.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if parts:
        text = parts[0]

    return text


def passes_filters(original: str, rewrite: str, detector_score: float) -> tuple[bool, str]:
    if not rewrite:
        return False, "empty"

    if len(rewrite.split()) < MIN_WORDS:
        return False, "too_short"

    orig_words    = len(original.split())
    rewrite_words = len(rewrite.split())
    if rewrite_words < orig_words * MIN_LEN_RATIO:
        return False, "len_ratio"

    sim = SequenceMatcher(None, original.lower(), rewrite.lower()).ratio()
    if sim >= MAX_SIM_RATIO:
        return False, "too_similar"

    if detector_score < DETECTOR_THRESHOLD:
        return False, "detector"

    return True, ""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_train_texts(path: Path) -> list[str]:
    texts            = []
    n_originals      = 0
    n_chunks_created = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                full_text = json.loads(line)["text"]
                n_originals += 1

                if len(full_text.split()) > CHUNK_TRIGGER_WORDS:
                    chunks = chunk_text(full_text)
                    texts.extend(chunks)
                    n_chunks_created += len(chunks)
                else:
                    texts.append(full_text)
            except (json.JSONDecodeError, KeyError):
                continue

    logger.info(
        f"Loaded {n_originals} originals; expanded long ones into {n_chunks_created} chunks. "
        f"Total candidate strings: {len(texts)}"
    )
    return texts


def _format_postfix(reject_counts: dict, attempts: int, accepted: int) -> dict:
    rate = (accepted / attempts) if attempts else 0.0
    postfix = {
        "tries": attempts,
        "rate":  f"{rate:.1%}",
    }
    for k, v in reject_counts.items():
        if v > 0:
            postfix[k] = v
    return postfix


def generate(target_pairs: int, max_attempts_per_text: int = 2):
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Fail fast if Ollama or the model isn't ready, before doing any other work
    check_ollama_ready()

    existing_pairs = []
    seen_originals = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing_pairs.append(obj)
                    seen_originals.add(obj["original"])
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info(f"Resuming: {len(existing_pairs)} pairs already in {OUTPUT_FILE}")

    if len(existing_pairs) >= target_pairs:
        logger.info(f"Already have {len(existing_pairs)} >= {target_pairs} target pairs. Done.")
        return

    logger.info(f"Loading detector...")
    detector = Detector()

    logger.info(f"Loading training texts from {TRAIN_FILE}")
    texts = load_train_texts(TRAIN_FILE)
    random.seed(42)
    random.shuffle(texts)

    texts = [t for t in texts if t not in seen_originals]
    logger.info(f"{len(texts)} candidate strings available after dedup")

    accepted = list(existing_pairs)
    reject_counts = {"empty": 0, "too_short": 0, "len_ratio": 0,
                     "too_similar": 0, "detector": 0, "ollama_fail": 0}
    attempts = 0

    bar = tqdm(total=target_pairs, initial=len(accepted), desc="SFT pairs", ncols=160)
    bar.set_postfix(**_format_postfix(reject_counts, attempts, 0))

    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")
    try:
        for original in texts:
            if len(accepted) >= target_pairs:
                break

            for attempt in range(max_attempts_per_text):
                prompt_template = random.choice(PROMPT_VARIANTS)
                prompt = prompt_template.format(text=original)

                raw = call_ollama(prompt, temperature=0.8 + 0.1 * attempt)
                attempts += 1

                def refresh():
                    accepted_this_run = len(accepted) - len(existing_pairs)
                    bar.set_postfix(
                        **_format_postfix(reject_counts, attempts, accepted_this_run)
                    )

                if raw is None:
                    reject_counts["ollama_fail"] += 1
                    refresh()
                    continue

                rewrite = clean_rewrite(raw)

                if not rewrite:
                    reject_counts["empty"] += 1
                    refresh()
                    continue

                if len(rewrite.split()) < MIN_WORDS:
                    reject_counts["too_short"] += 1
                    refresh()
                    continue

                orig_words    = len(original.split())
                rewrite_words = len(rewrite.split())
                if rewrite_words < orig_words * MIN_LEN_RATIO:
                    reject_counts["len_ratio"] += 1
                    refresh()
                    continue

                sim = SequenceMatcher(None, original.lower(), rewrite.lower()).ratio()
                if sim >= MAX_SIM_RATIO:
                    reject_counts["too_similar"] += 1
                    refresh()
                    continue

                d_score = detector.score(rewrite)
                ok, reason = passes_filters(original, rewrite, d_score)
                if not ok:
                    reject_counts[reason] += 1
                    refresh()
                    continue

                pair = {
                    "original":       original,
                    "rewrite":        rewrite,
                    "detector_score": round(d_score, 4),
                    "prompt_variant": PROMPT_VARIANTS.index(prompt_template),
                }
                out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                out_f.flush()
                accepted.append(pair)
                bar.update(1)
                refresh()
                break
    finally:
        out_f.close()
        bar.close()

    logger.info(f"Done. Accepted {len(accepted)} pairs, written to {OUTPUT_FILE}")
    logger.info(f"Total attempts this run: {attempts}")
    logger.info(f"Rejection counts: {reject_counts}")

    if len(accepted) < target_pairs:
        logger.warning(
            f"Only got {len(accepted)} / {target_pairs} pairs. "
            f"Most common rejection: {max(reject_counts, key=reject_counts.get)}."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--attempts", type=int, default=2)
    args = parser.parse_args()

    generate(target_pairs=args.target, max_attempts_per_text=args.attempts)