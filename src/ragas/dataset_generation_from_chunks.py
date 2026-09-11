from openai import OpenAI
from time import perf_counter

from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

from src.models.config import cfg
from src.ragas.dataset import docs


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
generator_embeddings = OpenAIEmbeddings(
    client=openai_client,
    model="text-embedding-3-small",
)

personas = [
    Persona(name="boy", role_description="6 years old boy"),
    Persona(name="professor", role_description="university professor"),
]

query_distribution = [
    (
        SingleHopSpecificQuerySynthesizer(
            llm=generator_llm,
            property_name="entities",  # или "themes"
        ),
        1.0,
    ),
]

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    persona_list=personas,
)


def create_testset():
    # docs уже нарезаны → NodeType.CHUNK + default_transforms_for_prechunked
    return generator.generate_with_chunks(
        chunks=docs,
        testset_size=4,
        query_distribution=query_distribution,
    )


if __name__ == "__main__":
    testset = create_testset()
    testset.to_pandas().to_csv("testset_chunks.csv", index=False)
    print(f"End time: {perf_counter() - start_time}")