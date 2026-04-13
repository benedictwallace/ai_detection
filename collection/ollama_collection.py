"""
Ollama AI text collection script for AI detection training data.

Usage:
    pip install requests tqdm python-dotenv
    ollama serve
    ollama pull llama3.2
    python collect_ollama_data.py

Outputs:
    data/ollama_samples.jsonl  - one JSON record per line
    data/collection_log.txt    - progress and error log
"""

import requests
import json
import time
import hashlib
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from itertools import product
from dotenv import load_dotenv
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR  = Path(os.getenv("OUTPUT_DIR", "data"))
OUTPUT_FILE = OUTPUT_DIR / "ollama_samples.jsonl"
LOG_FILE    = OUTPUT_DIR / "collection_log.txt"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL           = os.getenv("OLLAMA_COLLECTION_MODEL", "llama3.2")
RATE_LIMIT_RPM  = int(os.getenv("RATE_LIMIT_RPM", 200))
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", 512))
RETRY_LIMIT     = int(os.getenv("RETRY_LIMIT", 3))
RETRY_DELAY     = int(os.getenv("RETRY_DELAY", 5))

# ---------------------------------------------------------------------------
# Prompt library
# ---------------------------------------------------------------------------

DOMAINS = [
    "academic essay",
    "news article",
    "creative fiction",
    "product review",
    "social media post",
    "technical explanation",
    "email",
    "legal text",
    "medical explanation",
    "persuasive argument",
]

TOPICS = [
    "climate change",
    "artificial intelligence",
    "renewable energy",
    "mental health",
    "space exploration",
    "economic inequality",
    "the history of the internet",
    "vaccine development",
    "urban planning",
    "remote work",
]

STYLES = [
    "",
    "Write formally.",
    "Write in a casual, conversational tone.",
    "Be concise - use short sentences.",
    "Be thorough and detailed.",
]

TEMPERATURES = [0.7, 1.0, 1.3]

PROMPT_TEMPLATES = [
    "Write a short {domain} about {topic}.",
    "Explain {topic} to someone with no background knowledge.",
    "Write a {domain} arguing that {topic} is the most important issue today.",
    "Summarise the key points about {topic} in the style of a {domain}.",
    "Write a {domain} that takes an unusual or contrarian view on {topic}.",
]


def build_prompt(domain: str, topic: str, template: str, style: str) -> str:
    base = template.format(domain=domain, topic=topic)
    return f"{base} {style}".strip()


def prompt_id(prompt: str) -> str:
    """Stable hash used to deduplicate across runs."""
    return hashlib.md5(prompt.encode()).hexdigest()[:12]


def build_prompt_list() -> list[dict]:
    """
    Enumerate all (domain, topic, template, style) combinations.
    Each becomes one collection job.
    """
    jobs = []
    for domain, topic, template, style in product(
        DOMAINS, TOPICS, PROMPT_TEMPLATES, STYLES
    ):
        prompt = build_prompt(domain, topic, template, style)
        jobs.append({
            "prompt":    prompt,
            "prompt_id": prompt_id(prompt),
            "domain":    domain,
            "topic":     topic,
            "style":     style or "default",
            "template":  template,
        })
    return jobs


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def load_seen_ids(output_file: Path) -> set[str]:
    """Read already-collected prompt IDs so we never regenerate."""
    seen = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    seen.add(record["prompt_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return seen


def check_ollama_running() -> bool:
    """Verify Ollama is reachable before starting a long collection run."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def call_ollama(prompt: str, temperature: float) -> str | None:
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model":  MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": MAX_TOKENS,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["response"]

        except requests.exceptions.ConnectionError:
            logging.error("Ollama not running — start it with: ollama serve")
            return None

        except Exception as e:
            logging.error(f"Ollama error: {e} (attempt {attempt})")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)

    return None


def collect(jobs: list[dict], output_file: Path) -> None:
    seen    = load_seen_ids(output_file)
    pending = [j for j in jobs if j["prompt_id"] not in seen]
    total   = len(pending)

    logging.info(
        f"Total jobs: {len(jobs)} | Already collected: {len(seen)} | Pending: {total}"
    )

    if total == 0:
        logging.info("Nothing to do — all prompts already collected.")
        return

    random.shuffle(pending)

    collected = 0
    failed    = 0
    interval  = 60.0 / RATE_LIMIT_RPM

    with open(output_file, "a", buffering=1, encoding="utf-8") as out:
        for i, job in enumerate(pending, 1):
            t_start = time.monotonic()

            for temp in TEMPERATURES:
                text = call_ollama(job["prompt"], temp)

                if text is None:
                    failed += 1
                    logging.warning(
                        f"[{i}/{total}] FAILED  prompt_id={job['prompt_id']} temp={temp}"
                    )
                    continue

                record = {
                    **job,
                    "text":         text,
                    "label":        "ai",
                    "source_model": MODEL,
                    "temperature":  temp,
                    "max_tokens":   MAX_TOKENS,
                    "word_count":   len(text.split()),
                    "char_count":   len(text),
                    "timestamp":    datetime.now(timezone.utc).isoformat(),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                collected += 1

                elapsed    = time.monotonic() - t_start
                sleep_for  = max(0.0, interval - elapsed)
                if sleep_for > 0:
                    time.sleep(sleep_for)
                t_start = time.monotonic()

            if i % 50 == 0 or i == total:
                logging.info(
                    f"Progress: {i}/{total} prompts | "
                    f"Collected: {collected} samples | "
                    f"Failed: {failed}"
                )

    logging.info(f"Done. Collected {collected} samples, {failed} failures.")
    logging.info(f"Output: {output_file.resolve()}")


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def print_summary(output_file: Path) -> None:
    """Print a quick breakdown of what's been collected."""
    if not output_file.exists():
        print("No data file found.")
        return

    records = []
    with open(output_file) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total = len(records)
    print(f"\n{'='*50}")
    print(f"Total samples collected: {total}")

    domain_counts: dict[str, int] = {}
    temp_counts:   dict[float, int] = {}
    for r in records:
        domain_counts[r["domain"]] = domain_counts.get(r["domain"], 0) + 1
        temp_counts[r["temperature"]] = temp_counts.get(r["temperature"], 0) + 1

    print("\nBy domain:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {domain:<30} {count:>5}")

    print("\nBy temperature:")
    for temp, count in sorted(temp_counts.items()):
        print(f"  temp={temp}   {count:>5}")

    word_counts = [r.get("word_count", 0) for r in records]
    avg_words   = sum(word_counts) / len(word_counts) if word_counts else 0
    print(f"\nAverage word count: {avg_words:.0f}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE),
        ],
    )

    # Verify Ollama is running before starting
    if not check_ollama_running():
        raise RuntimeError(
            "Cannot reach Ollama at "
            f"{OLLAMA_BASE_URL}\n"
            "Run:  ollama serve\n"
            "Then: ollama pull llama3.2"
        )

    logging.info(f"Ollama running at {OLLAMA_BASE_URL} | Model: {MODEL}")

    jobs = build_prompt_list()
    logging.info(
        f"Built {len(jobs)} unique prompts across "
        f"{len(DOMAINS)} domains × {len(TOPICS)} topics × "
        f"{len(PROMPT_TEMPLATES)} templates × {len(STYLES)} styles"
    )
    logging.info(
        f"With {len(TEMPERATURES)} temperatures each -> "
        f"{len(jobs) * len(TEMPERATURES)} total samples when complete"
    )

    collect(jobs, OUTPUT_FILE)
    print_summary(OUTPUT_FILE)


if __name__ == "__main__":
    main()