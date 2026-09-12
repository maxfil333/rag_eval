from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

HF_DATASET = "wikimedia/wikipedia"
HF_CONFIG = "20231101.en"

# Stable, early-in-stream, non-political topics (verified against the dump).
DOC_TITLES = [
    "International Atomic Time",
    "Agricultural science",
    "Arithmetic mean",
]


def load_hf_documents(titles: list[str] = DOC_TITLES) -> list[str]:
    """Load full Wikipedia articles by title (streaming, stops when all found)."""

    found: dict[str, str] = {}

    stream = load_dataset(HF_DATASET, HF_CONFIG, split="train", streaming=True)
    for row in stream:
        title = row["title"]
        if title in DOC_TITLES:
            found[title] = row["text"]
            if len(found) == len(DOC_TITLES):
                break

    missing = set(DOC_TITLES) - found.keys()
    if missing:
        raise RuntimeError(f"Titles not found in {HF_DATASET}/{HF_CONFIG}: {sorted(missing)}")

    return [found[title] for title in titles]


docs = load_hf_documents()
doc_titles = list(DOC_TITLES)


if __name__ == "__main__":
    print(f"Loaded {len(docs)} documents from {HF_DATASET}/{HF_CONFIG}")
    for i, (title, doc) in enumerate(zip(doc_titles, docs)):
        preview = " ".join(doc.split())[:100]
        print(f"[{i}] {title!r}  chars={len(doc)}  preview={preview!r}")

    print(docs)
