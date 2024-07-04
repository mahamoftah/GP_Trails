import requests
from langchain_community.llms import Ollama


def text_to_text_ollam_api(text, model):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": text,
            "stream": False
        }
    )
    resp = response.json()
    return resp["message"]['content']


def text_to_text_ollam_langchain(text, model):

    llm = Ollama(model=model)
    return llm.invoke(text)

