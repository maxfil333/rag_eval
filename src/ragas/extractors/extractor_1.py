import asyncio
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms.extractors import NERExtractor

from src.ragas.dataset import docs
from src.models.config import cfg


openai_client = OpenAI(
    api_key=cfg.AI_TUNNEL_API_KEY,
    base_url="https://api.aitunnel.ru/v1/",
)

llm_kwargs = {
    "max_tokens": 8192,
}
if cfg.TEMPERATURE is not None:
    llm_kwargs["temperature"] = cfg.TEMPERATURE

generator_llm = llm_factory("gpt-5-mini", client=openai_client, **llm_kwargs)


kg = KnowledgeGraph()
for doc in docs[0:5]:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc},
        )
    )


async def extract_entities(kg: KnowledgeGraph):
    extractor = NERExtractor(llm=generator_llm)
    output = [await extractor.extract(node) for node in kg.nodes]
    print(output)
    return output


if __name__ == "__main__":
    asyncio.run(extract_entities(kg))
