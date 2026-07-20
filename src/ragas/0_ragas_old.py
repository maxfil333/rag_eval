import asyncio
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy

from src.models.model_ai_tunnel import create_model
from src.models.config import cfg


model = create_model(model_name='gpt-5-mini')

embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        api_key=cfg.AI_TUNNEL_API_KEY,
        base_url="https://api.aitunnel.ru/v1/",
        model="text-embedding-3-small",
    )
)

question = "Когда была основана OpenAI?"
contexts = [
    "OpenAI was founded in 2014."
]
answer = "OpenAI была основана в 2015 году."

sample = SingleTurnSample(
    user_input=question,
    response=answer,
    retrieved_contexts=contexts,
)

judge_llm = LangchainLLMWrapper(model)


async def main1():
    """ Достоверность. Правда ли ответ следует из найденного контекста? """
    metric = Faithfulness(llm=judge_llm)
    score = await metric.single_turn_ascore(sample)
    print(f"Faithfulness score: {score}")


async def main2():
    """ Получен ли ответ на вопрос пользователя? """
    metric = AnswerRelevancy(llm=judge_llm, embeddings=embeddings)
    score = await metric.single_turn_ascore(sample)
    print(f"AnswerRelevancy score: {score}")


if __name__ == "__main__":
    asyncio.run(main1())
    asyncio.run(main2())
