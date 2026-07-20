import asyncio

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

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


async def main1():
    """Достоверность. Правда ли ответ следует из найденного контекста?"""
    metric = Faithfulness(llm=judge_llm)
    result = await metric.ascore(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
    print(f"Faithfulness score: {result.value}")


async def main2():
    """Получен ли ответ на вопрос пользователя?"""
    metric = AnswerRelevancy(llm=judge_llm, embeddings=embeddings)
    result = await metric.ascore(
        user_input=question,
        response=answer,
    )
    print(f"AnswerRelevancy score: {result.value}")


if __name__ == "__main__":
    asyncio.run(main1())
    asyncio.run(main2())
