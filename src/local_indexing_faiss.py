"""
local_indexing_faiss.py

Este script indexa documentos PDF en una base de datos FAISS.
Divide los documentos en fragmentos, genera embeddings y los almacena en FAISS
utilizando la similitud del coseno.
"""

import os
import argparse
import numpy as np
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy
# from get_embedding_function import get_embedding_function
from src.get_embedding_function import get_embedding_function
import faiss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_FOLDER = os.path.join(BASE_DIR, "data", "pdfs")

def get_faiss_path(provider, model_name):
    """
    Devuelve la ruta del índice FAISS basada en el proveedor y modelo.
    """
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    return os.path.join(BASE_DIR, "data", f"faiss_index_{provider}_{safe_model_name}")

def normalize_embeddings(embeddings):
    """
    Normaliza los embeddings para garantizar la similitud del coseno en FAISS.
    """
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def load_and_process_pdfs(pdf_folder):
    """
    Carga documentos PDF, los divide en fragmentos y los prepara para su indexación.
    """
    chunks = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file_name)
            print(f"📄 Procesando: {file_name}")
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
            doc_chunks = splitter.split_documents(documents)

            for chunk in doc_chunks:
                chunk.metadata["source"] = pdf_path

            chunks.extend(doc_chunks)

    print(f"✅ Total de fragmentos generados: {len(chunks)}")
    return chunks

def save_to_faiss(chunks, model_name, provider):
    """
    Guarda los fragmentos en una base de datos FAISS utilizando similitud del coseno.
    """
    embeddings_model = get_embedding_function(model_name, provider)
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    embeddings = embeddings_model.embed_documents(texts)
    
    # Normalizar embeddings para similitud del coseno
    normalized_embeddings = normalize_embeddings(np.array(embeddings, dtype=np.float32))
    
    # Crear el índice FAISS con IndexFlatIP (para similitud del coseno)
    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)
    
    # Crear FAISS VectorStore
    docstore = InMemoryDocstore()
    index_to_docstore_id = {}
    for i, (doc, metadata) in enumerate(zip(chunks, metadatas)):
        doc_id = str(i)
        docstore.add({doc_id: doc})
        index_to_docstore_id[i] = doc_id
    
    faiss_vectorstore = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        distance_strategy=DistanceStrategy.COSINE,
    )
    
    # Guardar índice FAISS
    faiss_path = get_faiss_path(provider, model_name)
    faiss_vectorstore.save_local(faiss_path)
    print(f"✅ Índice FAISS guardado en {faiss_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexar documentos en FAISS con similitud del coseno")
    parser.add_argument("--model", type=str, required=True, help="Modelo de embeddings a utilizar.")
    parser.add_argument("--provider", type=str, choices=["huggingface", "openai"], required=True,
                        help="Proveedor del modelo de embeddings.")
    args = parser.parse_args()

    try:
        fragments = load_and_process_pdfs(PDF_FOLDER)
        save_to_faiss(fragments, args.model, args.provider)
    except Exception as e:
        print(f"❌ Error durante la indexación: {e}")