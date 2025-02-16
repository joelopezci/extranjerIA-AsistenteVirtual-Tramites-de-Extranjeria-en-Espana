"""
get_embedding_function.py

Este script define una función para cargar y configurar el modelo de embeddings.
Permite seleccionar entre modelos de Hugging Face y OpenAI.

Uso:
    - Para Hugging Face: modelo="distiluse-base-multilingual-cased-v1"
    - Para OpenAI: modelo="text-embedding-ada-002" (requiere API key)
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

def get_embedding_function(model_name="distiluse-base-multilingual-cased-v1", provider="huggingface"):
    """
    Devuelve la función de embeddings para LangChain.

    Args:
        model_name (str): Nombre del modelo a cargar. 
                         Por defecto, 'distiluse-base-multilingual-cased-v1' para Hugging Face.
                         Para OpenAI, usar 'text-embedding-ada-002'.
        provider (str): Proveedor del modelo ('huggingface' o 'openai').

    Returns:
        HuggingFaceEmbeddings | OpenAIEmbeddings: Modelo de embeddings cargado.

    Raises:
        ValueError: Si el modelo no se puede cargar o el proveedor es incorrecto.
    """
    try:
        if provider == "huggingface":
            print(f"⚙️ Cargando modelo de embeddings de Hugging Face: {model_name}")
            return HuggingFaceEmbeddings(model_name=model_name)
        
        elif provider == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("La API Key de OpenAI no está configurada. Defínela en la variable de entorno OPENAI_API_KEY.")

            print(f"⚙️ Cargando modelo de embeddings de OpenAI: {model_name}")
            return OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)
        
        else:
            raise ValueError("Proveedor no válido. Usa 'huggingface' o 'openai'.")

    except Exception as e:
        raise ValueError(f"Error al cargar el modelo '{model_name}' del proveedor '{provider}': {e}")