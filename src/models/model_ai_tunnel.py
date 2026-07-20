import os
from langchain_openai import ChatOpenAI

from .config import cfg


def create_model(model_name):
    model = ChatOpenAI(
        model=model_name,
        api_key=cfg.AI_TUNNEL_API_KEY,
        base_url="https://api.aitunnel.ru/v1/",
        timeout=30,
        temperature=cfg.TEMPERATURE,
    )
    return model


if __name__ == "__main__":
    model = create_model(cfg.MODEL_NAME)
    print(model.invoke("Hello World"))
