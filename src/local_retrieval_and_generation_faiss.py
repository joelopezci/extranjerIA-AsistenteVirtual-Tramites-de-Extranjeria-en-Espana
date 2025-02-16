"""
local_retrieval_and_generation_faiss.py

Este script recupera información relevante desde FAISS y genera respuestas usando un LLM.
"""

import os
import argparse
import numpy as np
from colorama import Fore, Style
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.llms import OpenAI
from langchain_community.vectorstores.utils import DistanceStrategy
# from get_embedding_function import get_embedding_function
from src.get_embedding_function import get_embedding_function
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Configurar la clave de API de Groq
os.environ["GROQ_API_KEY"] = "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"


def get_faiss_path(embedding_provider, model_name):
    """
    Devuelve la ruta de la base de datos FAISS basada en el proveedor de embeddings y modelo.

    Args:
        embedding_provider (str): Proveedor del modelo de embeddings ('huggingface', 'openai').
        model_name (str): Nombre del modelo de embeddings.

    Returns:
        str: Ruta de la base de datos FAISS.
    """
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", f"faiss_index_{embedding_provider}_{safe_model_name}")


def normalize_query_embedding(embedding):
    """
    Normaliza un embedding de consulta para garantizar compatibilidad con FAISS.

    Args:
        embedding (np.ndarray): Embedding de consulta generado por el modelo.

    Returns:
        np.ndarray: Embedding normalizado.
    """
    return embedding / np.linalg.norm(embedding)


def display_prompt(prompt):
    """
    Muestra el prompt en color rojo en la terminal.

    Args:
        prompt (str): El prompt formateado.
    """
    print(Fore.RED + "\n" + prompt + "\n" + Style.RESET_ALL)


PROMPT_TEMPLATE = """
Eres un asistente experto en extranjería en España. Debes responder en **español** y basarte únicamente en el contexto siguiente:

***Contexto:

{context}
--------------------------

Responde en español a la siguiente pregunta basándote únicamente en el anterior contexto:
{question}
"""


def format_context(results, k=3):
    """
    Formatea los resultados para cumplir con la estructura requerida en la generación.

    Args:
        results (list): Lista de documentos relevantes con contenido y metadatos.
        k (int): Número máximo de textos relevantes a incluir.

    Returns:
        str: Contexto formateado según el requerimiento.
    """
    formatted_context = ""
    for i, (doc, _) in enumerate(results[:k]):
        formatted_context += f"--Texto {i + 1}: {doc.page_content}\n\n----\n\n"
    return formatted_context.strip()


def query_rag(query_text, embedding_provider, embedding_model, llm_provider, llm_model, hf_pipeline=None, k=3):
    """
    Recupera los documentos más relevantes y genera una respuesta usando Groq, OpenAI o Hugging Face.

    Args:
        query_text (str): Pregunta a realizar al asistente.
        embedding_provider (str): Proveedor del modelo de embeddings.
        embedding_model (str): Modelo de embeddings a utilizar.
        llm_provider (str): Proveedor del modelo de generación.
        llm_model (str): Modelo de lenguaje para la generación.
        k (int): Número de documentos relevantes a recuperar.

    Returns:
        dict: Respuesta generada y documentos fuente con sus puntuaciones.
    """
    try:
        # Cargar FAISS
        faiss_path = get_faiss_path(embedding_provider, embedding_model)
        embeddings = get_embedding_function(embedding_model, embedding_provider)

        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"No se encontró un índice FAISS en {faiss_path}")

        print(f"📂 Cargando índice FAISS desde: {faiss_path}")
        db = FAISS.load_local(
            faiss_path, embeddings, allow_dangerous_deserialization=True, distance_strategy=DistanceStrategy.COSINE
        )


        # Normalizar la consulta
        query_embedding = normalize_query_embedding(embeddings.embed_query(query_text))
        
        # Obtener los documentos más similares con su score
        results = db.similarity_search_with_score_by_vector(query_embedding, k)

        if not results:
            raise ValueError("No se encontraron documentos relevantes para la consulta.")

        # Formatear el contexto
        formatted_context = format_context(results, k)
        prompt = PROMPT_TEMPLATE.format(context=formatted_context, question=query_text)

        # Mostrar el prompt
        display_prompt(prompt)

        # Generación de respuesta con LLM
        if llm_provider == "groq":
            print(f"⚙️ Usando modelo de Groq: {llm_model}")
            llm = ChatGroq(model_name=llm_model, temperature=0.7, max_tokens=500)
            response = llm.invoke(prompt).content

        elif llm_provider == "huggingface":
            print(f"⚙️ Usando modelo de Hugging Face: {llm_model}")
            
            if hf_pipeline is None:
                from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
                import torch

                def load_hf_model(model_name):
                    """
                    Carga un modelo de Hugging Face en memoria si aún no se ha cargado.
                    """
                    print(f"⚡ Precargando modelo Hugging Face: {model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
                    return pipeline("text-generation", model=model, tokenizer=tokenizer)

                hf_pipeline = load_hf_model(llm_model)

            response = hf_pipeline(
                prompt,
                max_new_tokens=300,  # Reduce la longitud máxima de respuesta
                do_sample=True,
                temperature=0.7,
                pad_token_id=hf_pipeline.tokenizer.eos_token_id,
            )[0]["generated_text"]

        elif llm_provider == "openai":
            print(f"⚙️ Usando modelo de OpenAI: {llm_model}")
            llm = OpenAI(model=llm_model)
            response = llm.invoke(prompt).content

        else:
            raise ValueError("Proveedor de LLM no válido. Usa 'huggingface', 'openai' o 'groq'.")

        # Extraer metadatos y scores
        sources_with_scores = [
            {"metadata": doc.metadata.get("source"), "score": score}
            for doc, score in results
        ]

        return {"response": response, "sources": sources_with_scores}

    except Exception as e:
        raise RuntimeError(f"Error durante la consulta: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consulta en la base de datos FAISS y genera una respuesta.")
    parser.add_argument("query", type=str, help="Texto de la consulta.")
    parser.add_argument("--embedding_model", type=str, required=True, help="Modelo de embeddings a utilizar.")
    parser.add_argument("--embedding_provider", type=str, required=True, choices=["huggingface", "openai"], help="Proveedor del modelo de embeddings.")
    parser.add_argument("--llm_provider", type=str, required=True, choices=["huggingface", "openai", "groq"], help="Proveedor del modelo de generación.")
    parser.add_argument("--llm_model", type=str, default="mixtral-8x7b-32768", help="Modelo de lenguaje para la generación.")
    args = parser.parse_args()

    try:
        result = query_rag(args.query, args.embedding_provider, args.embedding_model, args.llm_provider, args.llm_model)

        print("\n////////////////////////////////")
        print("\nRespuesta:")
        print(result["response"])
        print("\n------------------------------\n")
        print("\nDocumentos fuente con puntuación:")

        for item in result["sources"]:
            print(f"Documento: {item['metadata']} - Score: {item['score']:.4f}")

        print("////////////////////////////////\n")

    except Exception as e:
        print(f"Error: {e}")


# Ejecutar el script con:
# python src/local_retrieval_and_generation_faiss.py "¿Cuáles son los requisitos para la residencia en España?" --embedding_model distiluse-base-multilingual-cased-v1 --embedding_provider huggingface --llm_provider groq --llm_model mixtral-8x7b-32768
