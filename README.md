# 📌 extranjerIA: Asistente Virtual para Trámites de Extranjería en España 🇪🇸

**extranjerIA** es un asistente virtual basado en **Retrieval-Augmented Generation (RAG)** diseñado para facilitar el acceso a información sobre trámites de extranjería en España. Utiliza **FAISS** para la indexación y recuperación de documentos y **modelos de lenguaje (LLMs)** a través de **Groq y Hugging Face** para la generación de respuestas precisas.

---

## 🚀 Características Principales
✔ **Búsqueda semántica en documentos oficiales** de extranjería.  
✔ **Generación de respuestas contextuales** con modelos de lenguaje avanzados.  
✔ **Indexación eficiente** de documentos PDF mediante **FAISS**.  
✔ **Evaluación del rendimiento** del sistema con métricas de similitud entre respuestas generadas y humanas.  

---

## 📁 Estructura del Proyecto

```plaintext
extranjerIA/
├── data/                     # Almacén de datos y documentos
│   ├── pdfs/                 # Documentos PDF utilizados en el sistema
│   ├── faiss_index/           # Índices FAISS generados
│   ├── evaluations/           # Resultados de las evaluaciones del RAG
│   └── corpus/                # Corpus de preguntas y respuestas
├── notebooks/                 # Notebooks de desarrollo y evaluación
├── src/                       # Código fuente del proyecto
│   ├── local_indexing_faiss.py      # Script para indexación de documentos
│   ├── local_retrieval_and_generation_faiss.py  # Recuperación y generación de respuestas
│   ├── evaluate_rag_faiss.py  # Evaluación de rendimiento del sistema
│   ├── get_embedding_function.py  # Función para obtener embeddings
│   └── utils/                  # Módulos auxiliares
├── .gitignore                  # Archivos y carpetas ignoradas por Git
├── README.md                   # Descripción del proyecto
├── requirements.txt             # Dependencias necesarias
└── venv/                        # Entorno virtual (opcional)


## 📌 Instalación y Configuración

### 🔹 1️⃣ Clonar el Repositorio
Para obtener una copia del proyecto en tu máquina local:
```bash
git clone <url-del-repositorio>
cd extranjerIA
