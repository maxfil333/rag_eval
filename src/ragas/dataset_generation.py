from openai import OpenAI
from time import perf_counter

from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.transforms import KeyphrasesExtractor, apply_transforms

from src.models.config import cfg
from src.ragas.dataset import docs


# init parameters

start_time = perf_counter()

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
generator_embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")


# KnowledgeGraph

kg = KnowledgeGraph()
for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc},
        )
    )
print(kg)


# transforms

apply_transforms(
    kg,
    transforms=[KeyphrasesExtractor(llm=generator_llm)],
)


# personas

personas = [
    Persona(name="boy", role_description="6 years old boy"),
    Persona(name="professor", role_description="university professor"),
]


# synthesizers

query_distribution = [
    (
        SingleHopSpecificQuerySynthesizer(
            llm=generator_llm,
            property_name="keyphrases",
        ),
        1.0,
    ),
]


# generator

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    knowledge_graph=kg,
    persona_list=personas,
)

if __name__ == "__main__":
    testset = generator.generate(
        testset_size=4,
        query_distribution=query_distribution,
    )
    testset.to_pandas().to_csv("testset.csv", index=False)

    print(f"End time: {perf_counter() - start_time}")
