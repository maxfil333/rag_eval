import pandas as pd
from pathlib import Path

from src.ragas.pipeline.rag import rag


eval_dataset = Path(__file__).resolve().parent / "eval_dataset.csv"
after_rag_dataset = Path(__file__).resolve().parent / "after_rag_dataset.csv"


def run_rag(eval_dataset_pth, after_rag_dataset_pth) -> None:
    df = pd.read_csv(eval_dataset_pth)
    responses = []
    retrieved_contexts = []
    for user_input in df["user_input"]:
        result = rag(user_input)
        responses.append(result["answer"])
        retrieved_contexts.append(result["retrieved_contexts"])
    df["response"] = responses
    df["retrieved_contexts"] = retrieved_contexts
    df.to_csv(after_rag_dataset_pth, index=False)

