import chromadb
from pathlib import Path
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

from src.models.config import cfg


load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_PATH = BASE_DIR / ".chroma"

COLLECTION_NAME = "rag_mini_wikipedia"
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-5-mini"
TOP_K = 3
EMBED_BATCH_SIZE = 100

openai_client = OpenAI(
    api_key=cfg.AI_TUNNEL_API_KEY,
    base_url="https://api.aitunnel.ru/v1/",
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def index_corpus(collection) -> None:
    if collection.count() > 0:
        print(f"коллекция уже проиндексирована: {collection.count()} пассажей")
        return

    corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
    passages = [p for p in corpus["passages"]["passage"] if p and p.strip()]

    print(f"индексирую {len(passages)} пассажей в Chroma...")
    for start in range(0, len(passages), EMBED_BATCH_SIZE):
        batch = passages[start : start + EMBED_BATCH_SIZE]
        embeddings = embed_texts(batch)
        ids = [str(start + i) for i in range(len(batch))]
        collection.add(ids=ids, documents=batch, embeddings=embeddings)
        print(f"  {min(start + EMBED_BATCH_SIZE, len(passages))}/{len(passages)}")


def retrieve(collection, question: str, k: int = TOP_K) -> list[str]:
    query_embedding = embed_texts([question])[0]
    result = collection.query(query_embeddings=[query_embedding], n_results=k)
    return result["documents"][0]


def generate_answer(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, 1))
    llm_kwargs = {}
    if cfg.TEMPERATURE is not None:
        llm_kwargs["temperature"] = cfg.TEMPERATURE

    completion = openai_client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {
                "role": "system",
                "content": """Отвечай на вопрос только по приведённому контексту. 
                Если ответа в контексте нет, так и скажи. По возможности отвечай кратко. 
                Используй контекст для ответа, но не упоминай в ответе про контекст.""",
            },
            {
                "role": "user",
                "content": f"Контекст:\n{context_block}\n\nВопрос: {question}",
            },
        ],
        **llm_kwargs,
    )
    return completion.choices[0].message.content


def rag(question: str) -> dict:
    collection = get_collection()
    index_corpus(collection)
    contexts = retrieve(collection, question)
    answer = generate_answer(question, contexts)
    return {
        "question": question,
        "retrieved_contexts": contexts,
        "answer": answer,
    }


if __name__ == "__main__":
    qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
    question = qa["test"]["question"][0]
    result = rag(question)

    print("\nquestion:", result["question"])
    print("\nretrieved_contexts:")
    for i, ctx in enumerate(result["retrieved_contexts"], 1):
        print(f"  [{i}] {ctx[:300]}{'...' if len(ctx) > 300 else ''}")
    print("\nanswer:", result["answer"])
