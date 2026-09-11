import asyncio
from time import perf_counter

import ragas.async_utils as ragas_async_utils
import ragas.executor as ragas_executor
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory

from src.models.config import cfg

start_time = perf_counter()


def _share_event_loop_across_ragas_runs(runner: asyncio.Runner) -> None:
    """Ragas вызывает asyncio.run() отдельно для transforms, сценариев и сэмплов.

    AsyncOpenAI/httpx привязываются к loop; после закрытия loop следующий
    вызов даёт Connection error / Event loop is closed (особенно на Windows).
    asyncio.Runner держит один и тот же loop между этими этапами.
    """

    def run_on_shared_loop(async_func, allow_nest_asyncio: bool = True):
        coro = async_func() if callable(async_func) else async_func
        return runner.run(coro)

    ragas_async_utils.run = run_on_shared_loop
    ragas_executor.run = run_on_shared_loop


openai_client = AsyncOpenAI(
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

from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType
from src.ragas.dataset import docs

kg = KnowledgeGraph()

for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc}
        )
    )

print(kg)

# transforms

from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import KeyphrasesExtractor

keyphrase_extractor = KeyphrasesExtractor(llm=generator_llm)

transforms = [
    keyphrase_extractor
]

# personas

from ragas.testset.persona import Persona

boy = Persona(
    name="boy",
    role_description="6 years old boy",
)

professor = Persona(
    name="professor",
    role_description="university professor",
)

personas = [boy, professor]


# synthesizers

from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

# Только keyphrases: HeadlinesExtractor не применялся, узлов с headlines нет.
query_distibution = [
    (
        SingleHopSpecificQuerySynthesizer(
            llm=generator_llm, property_name="keyphrases"
        ),
        1.0,
    ),
]

from ragas.testset import TestsetGenerator

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    knowledge_graph=kg,
    persona_list=personas,
)

if __name__ == "__main__":
    with asyncio.Runner() as runner:
        _share_event_loop_across_ragas_runs(runner)
        apply_transforms(kg, transforms=transforms)
        testset = generator.generate(
            testset_size=4,
            query_distribution=query_distibution,
        )
    testset.to_pandas().to_csv(
        "testset.csv",
        index=False,
    )
    print(f"End time: {perf_counter() - start_time}")