import requests
from langchain_community.llms import Ollama
from groq import Groq
import json


def text_to_text_ollam_api(question, model):
    quest = [{
        "role": "user",
        "content": question,
    }]

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": quest,
            "stream": False
        }
    )

    resp = response.json()
    return resp["message"]['content']


def text_to_text_ollam_langchain(question, model):
    quest = [{
        "role": "user",
        "content": question,
    }]

    llm = Ollama(model=model)
    return llm.invoke(quest)


def text_to_text_groq(question, model):

    groq = Groq(api_key="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")
    quest = [{
        "role": "user",
        "content": question,
    }]

    response = groq.chat.completions.create(
        messages=quest,
        model=model,
        temperature=0,
        stream=False,
    )

    return response.choices[0].message.content
