import asyncio

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextRelevance,
    Faithfulness,
)

from src.models.config import cfg

client = AsyncOpenAI(
    api_key=cfg.AI_TUNNEL_API_KEY,
    base_url="https://api.aitunnel.ru/v1/",
)

llm_kwargs = {}
if cfg.TEMPERATURE is not None:
    llm_kwargs["temperature"] = cfg.TEMPERATURE

judge_llm = llm_factory("gpt-5-mini", client=client, **llm_kwargs)

embeddings = OpenAIEmbeddings(
    client=client,
    model="text-embedding-3-small",
)

question = "Когда была основана OpenAI?"
contexts = [
    "OpenAI was founded in 2014."
]
answer = "OpenAI была основана в 2015 году."
reference = "OpenAI была основана в 2014 году."


async def main1():
    """ 
    Faithfulness (Достоверность)
    Правда ли ответ следует из найденного контекста?
    """
    print(f"counting Faithfulness score...")
    metric = Faithfulness(llm=judge_llm)
    result = await metric.ascore(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
    print(f"Faithfulness score: {result.value}")


async def main2():
    """
    AnswerRelevancy
    Получен ли ответ на вопрос пользователя?
    """
    print(f"counting AnswerRelevancy score...")
    metric = AnswerRelevancy(llm=judge_llm, embeddings=embeddings)
    result = await metric.ascore(
        user_input=question,
        response=answer,
    )
    print(f"AnswerRelevancy score: {result.value}")


async def main3():
    """
    ContextRelevance
    Насколько найденный контекст относится к вопросу пользователя?
    """
    print(f"counting ContextRelevance score...")
    metric = ContextRelevance(llm=judge_llm)
    result = await metric.ascore(
        user_input=question,
        retrieved_contexts=contexts,
    )
    print(f"ContextRelevance score: {result.value}")


async def main4():
    """
    AnswerCorrectness
    Насколько ответ совпадает с эталонным?
    """
    print(f"counting AnswerCorrectness score...")
    metric = AnswerCorrectness(llm=judge_llm, embeddings=embeddings)
    result = await metric.ascore(
        user_input=question,
        response=answer,
        reference=reference,
    )
    print(f"AnswerCorrectness score: {result.value}")


async def main():
    tasks = [main1(), main2(), main3(), main4()]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())