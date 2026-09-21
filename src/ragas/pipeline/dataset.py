import os
import json
import pathlib
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

HF_DATASET = "wikimedia/wikipedia"
HF_CONFIG = "20231101.en"
DATASET_SAVE_PATH = "datas/docs.json"
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


def save_documents(pth: pathlib.Path, docs: list[str]) -> None:
    os.makedirs(pth.parent, exist_ok=True)
    with open(DATASET_SAVE_PATH, "w", encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=4)


def get_docs() -> list[str]:
    if os.path.exists(DATASET_SAVE_PATH):
        with open(DATASET_SAVE_PATH, "r", encoding='utf-8') as f:
            docs = json.load(f)
    else:
        docs = load_hf_documents(DOC_TITLES)
        save_documents(pth=pathlib.Path(DATASET_SAVE_PATH), docs=docs)
    return docs



if __name__ == "__main__":
    docs = get_docs()
    for i, (title, doc) in enumerate(zip(DOC_TITLES, docs)):
        preview = " ".join(doc.split())[:100]
        print(f"[{i}] {title!r}  chars={len(doc)}  preview={preview!r}")
    print(docs)
