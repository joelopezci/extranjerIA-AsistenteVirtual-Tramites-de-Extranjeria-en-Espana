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
```

---

## 📌 Instalación y Configuración

### 🔹 1️⃣ Clonar el Repositorio
Para obtener una copia del proyecto en tu máquina local, clona el repositorio y accede a la carpeta del proyecto.
```bash
git clone <url-del-repositorio>
cd extranjerIA
```

### 🔹 2️⃣ Crear un Entorno Virtual y Activarlo
Se recomienda usar un entorno virtual para gestionar dependencias.
```bash
python -m venv venv
source venv/bin/activate  # En macOS/Linux
venv\Scripts\activate     # En Windows
```

### 🔹 3️⃣ Instalar Dependencias
Ejecuta la instalación de todas las dependencias necesarias.
```bash
pip install -r requirements.txt
```

### 🔹 4️⃣ Configurar Variables de Entorno (Opcional)
Si utilizas **Groq** como proveedor de modelos de lenguaje, necesitas configurar la API Key.
```powershell
En Windows:
$env:GROQ_API_KEY="tu_api_key" # En Windows (PowerShell)
```

## ⚙️ Uso del Proyecto

### 🔍 1️⃣ Indexación de Documentos
Para procesar los PDFs y almacenarlos en FAISS, ejecuta el script correspondiente.
```bash
python src/local_indexing_faiss.py --model distiluse-base-multilingual-cased-v1 --provider huggingface
```

### 🔎 2️⃣ Recuperación y Generación de Respuestas
Para realizar consultas y obtener respuestas del asistente.
```bash
python src/local_retrieval_and_generation_faiss.py "¿Cuáles son los requisitos para la residencia en España?" --embedding_model distiluse-base-multilingual-cased-v1 --embedding_provider huggingface --llm_provider groq --llm_model mixtral-8x7b-32768
```

### 📊 3️⃣ Evaluación del RAG
Para evaluar la calidad de las respuestas generadas en comparación con respuestas humanas.
```bash
python src/evaluate_rag_faiss.py
```

## 📊 Evaluación y Resultados

El sistema fue evaluado con un **corpus de 40 preguntas** extraídas de documentos oficiales.  
Se compararon respuestas generadas por diferentes **modelos de lenguaje y embeddings**, utilizando **similitud del coseno** como métrica principal.  

### 🔹 Modelos evaluados:

| Modelo de Embeddings | LLM Utilizado | Score Promedio |
|----------------------|--------------|---------------|
| distiluse-base-multilingual-cased-v1 | mixtral-8x7b-32768 | **0.82** |
| distiluse-base-multilingual-cased-v1 | llama-3.2-3b-preview | 0.78 |
| distiluse-base-multilingual-cased-v1 | llama-3.2-1b-preview | 0.76 |
| multilingual-e5-large-ft | mixtral-8x7b-32768 | **0.85** |
| multilingual-e5-large-ft | llama-3.2-3b-preview | 0.79 |
| multilingual-e5-large-ft | llama-3.2-1b-preview | 0.77 |

🔹 **Conclusión**: La combinación **multilingual-e5-large-ft + Mixtral-8x7b** obtuvo el mejor rendimiento en términos de similitud con respuestas humanas.

## 📢 Contribuciones y Mejora del Proyecto
Este proyecto está en desarrollo activo. Puedes contribuir de la siguiente forma:

✅ **Reportando errores**: Abre un issue en el repositorio.  
✅ **Mejorando la documentación**: Propuestas y PRs son bienvenidos.  
✅ **Probando nuevos modelos**: Comparte resultados con diferentes embeddings o LLMs.

## 📜 Licencia
Este proyecto está bajo la licencia **MIT**, lo que permite su uso y modificación libre con atribución adecuada.

## 📌 Referencias
1. Lewis, P., et al. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. *arXiv preprint arXiv:2005.11401*.
2. Johnson, J., et al. (2024). **FAISS: Facebook AI Similarity Search**. Disponible en: [https://faiss.ai](https://faiss.ai)
3. Hugging Face. (2024). **Meta LLaMA-3.2**. Disponible en: [https://huggingface.co/meta-llama](https://huggingface.co/meta-llama)
4. Groq. (2024). **Groq LLM API**. Disponible en: [https://groq.com](https://groq.com)
