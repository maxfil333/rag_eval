import os
import asyncio
from pathlib import Path
from time import perf_counter
from dotenv import load_dotenv

import ragas.async_utils as ragas_async_utils
import ragas.executor as ragas_executor
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.transforms import (
    CosineSimilarityBuilder,
    EmbeddingExtractor,
    HeadlineSplitter,
    HeadlinesExtractor,
    KeyphrasesExtractor,
    OverlapScoreBuilder,
    Parallel,
    apply_transforms,
)

# Here, `docs` is a regular `list[str]`, where each element of the list is a document.
from src.ragas.pipeline.dataset import docs

load_dotenv()


def _share_event_loop_across_ragas_runs(runner: asyncio.Runner) -> None:
    """Keep one event loop for transforms + generation (Windows / AsyncOpenAI)."""

    def run_on_shared_loop(async_func, allow_nest_asyncio: bool = True):
        coro = async_func() if callable(async_func) else async_func
        return runner.run(coro)

    ragas_async_utils.run = run_on_shared_loop
    ragas_executor.run = run_on_shared_loop


def _count_rel_types(kg: KnowledgeGraph) -> dict[str, int]:
    """Helper function for displaying logs."""
    counts: dict[str, int] = {}
    for rel in kg.relationships:
        counts[rel.type] = counts.get(rel.type, 0) + 1
    return counts


def build_knowledge_graph(documents: list[str]) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for doc in documents:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={"page_content": doc},
            )
        )
    return kg


def build_transforms(llm, embedding_model):

    def _is_document(node: Node) -> bool:
        return node.type == NodeType.DOCUMENT

    def _is_chunk(node: Node) -> bool:
        return node.type == NodeType.CHUNK

    return [
        # First, we extract the `headlines` feature.
        HeadlinesExtractor(llm=llm, filter_nodes=_is_document),

        # Based on the `headlines` feature, we split the original documents into chunks.
        HeadlineSplitter(min_tokens=300, max_tokens=1000),

        # Extractors
        Parallel(
            KeyphrasesExtractor(llm=llm, property_name="keyphrases", filter_nodes=_is_chunk),
            EmbeddingExtractor(
                embedding_model=embedding_model,
                property_name="embedding",
                embed_property_name="page_content",
                filter_nodes=_is_chunk,
            ),
        ),

        # Relations
        Parallel(
            CosineSimilarityBuilder(
                property_name="embedding",
                new_property_name="cosine_similarity",
                threshold=0.75,
                filter_nodes=_is_chunk,
            ),
            OverlapScoreBuilder(
                property_name="keyphrases",
                new_property_name="overlap_score",
                threshold=0.01,
                distance_threshold=0.9,
                filter_nodes=_is_chunk,
            ),
        ),
    ]


def build_generator(llm, embedding_model, kg: KnowledgeGraph) -> TestsetGenerator:
    personas = [
        Persona(name="student", role_description="curious university student"),
        Persona(name="professor", role_description="university professor"),
    ]
    return TestsetGenerator(
        llm=llm,
        embedding_model=embedding_model,
        knowledge_graph=kg,
        persona_list=personas,
    )


def query_distribution(llm):
    return [
        (
            SingleHopSpecificQuerySynthesizer(
                llm=llm,
                property_name="keyphrases"
            ),
            0.5,
        ),
        (
            MultiHopSpecificQuerySynthesizer(
                llm=llm,
                property_name="keyphrases",
                relation_type="keyphrases_overlap",
            ),
            0.5,
        ),
    ]


def create_testset() -> None:
    start = perf_counter()

    openai_client = AsyncOpenAI(
        api_key= os.getenv("AI_TUNNEL_API_KEY"),
        base_url="https://api.aitunnel.ru/v1/",
    )
    llm_kwargs = {"max_tokens": 8192}

    llm = llm_factory("gpt-5-mini", client=openai_client, **llm_kwargs)
    embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")

    kg = build_knowledge_graph(docs)
    print(f"KG before transforms: {kg}")

    transforms = build_transforms(llm, embeddings)
    generator = build_generator(llm, embeddings, kg)

    with asyncio.Runner() as runner:
        _share_event_loop_across_ragas_runs(runner)
        apply_transforms(kg, transforms=transforms)
        print(
            f"KG after transforms: nodes={len(kg.nodes)} "
            f"relationships={len(kg.relationships)}"
        )
        for rel_type, count in _count_rel_types(kg).items():
            print(f"  rel {rel_type}: {count}")

        testset = generator.generate(
            testset_size=6,
            query_distribution=query_distribution(llm),
        )

    out_csv = Path(__file__).resolve().parents[3] / "pipeline_eval_dataset.csv"
    testset.to_pandas().to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(testset)} samples)")
    print(f"Elapsed: {perf_counter() - start:.1f}s")


if __name__ == "__main__":
    create_testset()
