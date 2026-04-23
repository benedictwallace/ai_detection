import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

_model     = None
_tokenizer = None
_device    = None


def _load():
    global _model, _tokenizer, _device
    if _model is None:
        _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        _model     = GPT2LMHeadModel.from_pretrained("gpt2").to(_device)
        _model.eval()


def perplexity(text: str) -> float:
    """
    Compute GPT-2 perplexity for a piece of text.
    Lower perplexity = more fluent / natural language.
    """
    _load()

    encodings = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(_device)

    input_ids = encodings.input_ids

    with torch.no_grad():
        loss = _model(input_ids, labels=input_ids).loss

    return torch.exp(loss).item()


def fluency_score(text: str, max_perplexity: float = 1000.0) -> float:
    """
    Convert perplexity into a 0-1 score where higher = more fluent.
    Clips at max_perplexity so extreme values don't dominate.

    A well-formed English sentence typically has perplexity 50-200.
    Gibberish or incoherent text scores 500+.
    """
    ppl   = perplexity(text)
    ppl   = min(ppl, max_perplexity)
    score = 1.0 - (ppl / max_perplexity)
    return round(score, 4)


if __name__ == "__main__":
    tests = [
        "I went to the shop and bought some milk.",
        "The utilization of AI systems has demonstrated improvements.",
        "xkqz banana purple telephone grammar wrong the.",
    ]

    print(f"{'Text':<55} {'PPL':>8}  {'Score':>6}")
    print("-" * 75)
    for t in tests:
        ppl   = perplexity(t)
        score = fluency_score(t)
        print(f"{t[:54]:<55} {ppl:>8.1f}  {score:>6.4f}")