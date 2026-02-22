"""LLM factory for different providers (OpenAI, Google, Groq, Ollama)."""

from typing import Dict


def get_llm(provider_config: Dict, temperature: float = 0.0, streaming: bool = False):
    """
    Return an LLM instance based on provider configuration.
    Compatible with modern LangChain >= 0.2
    """

    provider = provider_config["provider"]
    model_name = provider_config["model"]

    # ---- OPENAI ----
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except Exception:
            raise ImportError(
                "Please install: pip install langchain-openai"
            )

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=streaming,
        )

    # ---- GOOGLE GEMINI ----
    elif provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception:
            raise ImportError(
                "Please install: pip install langchain-google-genai"
            )

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
        )

    # ---- GROQ ----
    elif provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except Exception:
            raise ImportError(
                "Please install: pip install langchain-groq"
            )

        return ChatGroq(
            model_name=model_name,
            temperature=temperature,
        )

    # ---- OLLAMA ----
    elif provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
        except Exception:
            raise ImportError(
                "Please install: pip install langchain-community"
            )

        return ChatOllama(
            model=model_name,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")