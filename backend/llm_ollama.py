from langchain.llms import Ollama

def get_local_llm(model_name: str = "mistral"):
    return Ollama(model=model_name)
