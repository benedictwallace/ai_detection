import torch
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import os


_model  = None
_device = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _load():
    global _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model  = SentenceTransformer("all-MiniLM-L6-v2", device=str(_device))


def semantic_score(original: str, rewrite: str) -> float:
    """
    Cosine similarity between sentence embeddings of the original
    and the rewrite. Returns a float in [0, 1].

    Higher = rewrite preserves the meaning of the original.
    Below 0.5 usually means the rewrite has drifted too far.
    """
    _load()

    embeddings = _model.encode(
        [original, rewrite],
        convert_to_tensor=True,
        device=str(_device),
        show_progress_bar=False,
    )

    score = cosine_similarity(
        embeddings[0].unsqueeze(0),
        embeddings[1].unsqueeze(0),
    ).item()

    # Clamp to [0, 1]
    return round(max(0.0, score), 4)


if __name__ == "__main__":
    pairs = [
        (
            "The cat sat on the mat.",
            "A cat was sitting on the mat.", # close paraphrase
        ),
        (
            "The cat sat on the mat.",
            "Feline animals often rest on flat surfaces.",  # loose paraphrase
        ),
        (
            "The cat sat on the mat.",
            "Artificial intelligence is transforming healthcare.",  # unrelated
        ),
    ]

    print(f"{'Original':<30} {'Rewrite':<48} {'Score':>6}")
    print("-" * 90)
    for original, rewrite in pairs:
        score = semantic_score(original, rewrite)
        print(f"{original[:29]:<30} {rewrite[:47]:<48} {score:>6.4f}")