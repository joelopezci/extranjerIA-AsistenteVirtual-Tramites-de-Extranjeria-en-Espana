import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util

from src.get_embedding_function import get_embedding_function
from src.local_retrieval_and_generation_faiss import query_rag  # Habilitar para ejecutar desde notebook
import numpy as np


def normalize_vector(vector):
    """
    Normaliza un vector para garantizar que la similitud sea equivalente a coseno.
    
    Args:
        vector (np.ndarray): Vector a normalizar.

    Returns:
        np.ndarray: Vector normalizado.
    """
    return vector / np.linalg.norm(vector)


def evaluate_rag_performance(input_excel, output_excel, embedding_provider, embedding_model, llm_provider, llm_model):
    """
    Evalúa el rendimiento del sistema RAG comparando las respuestas generadas con las respuestas humanas.

    Args:
        input_excel (str): Ruta del archivo Excel de entrada.
        output_excel (str): Ruta del archivo Excel de salida con los resultados.
        embedding_provider (str): Proveedor de embeddings ('huggingface', 'openai').
        embedding_model (str): Modelo de embeddings a utilizar.
        llm_provider (str): Proveedor del modelo de lenguaje ('huggingface', 'openai', 'groq').
        llm_model (str): Modelo de lenguaje para la generación.
    """
    try:
        # Cargar el archivo Excel
        df = pd.read_excel(input_excel)

        # Verificar las columnas necesarias
        required_columns = ["Pregunta", "Respuesta_Humana"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"La columna requerida '{col}' no está presente en el archivo Excel.")

        # Cargar modelo de embeddings para evaluación
        eval_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # 🔥 Precargar el modelo Hugging Face solo si se usa
        hf_pipeline = None
        if llm_provider == "huggingface":
            from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
            import torch

            def load_hf_model(model_name):
                """
                Carga un modelo de Hugging Face en memoria una sola vez.
                """
                print(f"⚡ Precargando modelo Hugging Face: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
                return pipeline("text-generation", model=model, tokenizer=tokenizer)

            hf_pipeline = load_hf_model(llm_model)

        for index, row in df.iterrows():
            query_text = row["Pregunta"]

            # ⚡ Pasar el modelo precargado para evitar recargas
            result = query_rag(query_text, embedding_provider, embedding_model, llm_provider, llm_model, hf_pipeline=hf_pipeline)

            # Almacenar los resultados en el DataFrame
            df.at[index, "Respuesta_RAG"] = result["response"]
            df.at[index, "Modelo_embedding_usado"] = embedding_model
            df.at[index, "LLM_usado"] = llm_model
            df.at[index, "Documentos_relevantes_con_score"] = str(result["sources"])

            # Calcular similitud de respuesta humana vs RAG
            human_answer = row["Respuesta_Humana"]
            rag_answer = result["response"]

            # Obtener embeddings normalizados
            human_embedding = normalize_vector(eval_model.encode(human_answer, convert_to_tensor=True))
            rag_embedding = normalize_vector(eval_model.encode(rag_answer, convert_to_tensor=True))

            similarity_score = util.pytorch_cos_sim(human_embedding, rag_embedding).item()
            df.at[index, "Score_RHvsRRAG"] = similarity_score

        # Guardar los resultados en un nuevo archivo
        df.to_excel(output_excel, index=False)
        print(f"✅ Evaluación completada. Resultados guardados en {output_excel}")

    except Exception as e:
        print(f"❌ Error durante la evaluación: {e}")


if __name__ == "__main__":
    input_excel = "../data/corpus_extranjerIA_muestra.xlsx"
    output_excel = "../data/evaluations/evaluacion_rag_faiss.xlsx"
    embedding_provider = "huggingface"
    embedding_model = "distiluse-base-multilingual-cased-v1"
    llm_provider = "groq"
    llm_model = "mixtral-8x7b-32768"

    evaluate_rag_performance(input_excel, output_excel, embedding_provider, embedding_model, llm_provider, llm_model)