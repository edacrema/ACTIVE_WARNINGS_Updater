from langchain_google_vertexai import ChatVertexAI
from config import PROJECT_ID, LOCATION, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


def get_llm():
    return ChatVertexAI(
        model=LLM_MODEL,
        project=PROJECT_ID,
        location=LOCATION,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


# Singleton instance
llm = get_llm()
