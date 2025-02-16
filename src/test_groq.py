import os
from langchain_groq import ChatGroq

# Configuración de la API Key (si no está en variables de entorno)
os.environ["GROQ_API_KEY"] = "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"

def query_groq(model_name, query_text, max_tokens=500, temperature=0.7):
    """
    Envía una consulta al modelo de Groq y devuelve la respuesta.

    Args:
        model_name (str): Nombre del modelo de Groq a utilizar.
        query_text (str): Pregunta o consulta a realizar.
        max_tokens (int): Número máximo de tokens a generar.
        temperature (float): Control de aleatoriedad de generación.

    Returns:
        str: Respuesta generada por el modelo.
    """
    try:
        # Inicializar el modelo de Groq
        llm = ChatGroq(
            model_name=model_name, 
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Generar la respuesta
        response = llm.invoke(query_text)
        return response.content

    except Exception as e:
        print(f"❌ Error al consultar el modelo de Groq: {e}")
        return None

if __name__ == "__main__":
    # Definir la consulta de prueba
    query = "¿Cuáles son los requisitos para la residencia en España?"
    
    # Elegir un modelo de Groq (por ejemplo, 'mistral-7b' o 'mixtral-8x7b-32768')
    model_name = "mixtral-8x7b-32768"  # Puedes probar con "mistral-7b"

    print(f"🔎 Consultando Groq con el modelo '{model_name}'...\n")
    response = query_groq(model_name, query)

    if response:
        print("📝 Respuesta del modelo:")
        print(response)
    else:
        print("⚠️ No se recibió respuesta del modelo.")





----
--------------------------

Responde en español a la siguiente pregunta basándote únicamente en el anterior contexto:
¿Cuáles son los requisitos para la residencia en España?

[0m
⚙️ Usando modelo de Groq: llama-3.2-1b-preview

📝 Respuesta generada:
Según el texto proporcionado, los requisitos para la residencia en España son los siguientes:

1. **Padrón histórico o inmigrante**: La mayoría de las personas que buscan residir en España tienen una parrilla histórica o están inmigrantes. 
No suelen tener un requisito de permanencia continuada en España durante los 2 años anteriores a la solicitud.

2. **Compromiso de formación EX10**: La autorización de residencia temporal por circunstancias excepcionales requiere que los solicitantes tengan
un compromiso de formación EX10, que comprende 4 años de formación, 1 año de práctica profesional y 1 año de experiencia laboral.

3. **Arraigo laboral (HI 35)**: La autorización de residencia temporal por circunstancias excepcionales también requiere que los solicitantes hayan
sido contratados por un despacho laboral (laboral) con una tarifa económica correspondiente (HI 35).

4. **Solicitar certificado de registro o tarjeta de residencia de familiar de ciudadano de la Unión**: Los familiares de ciudadanos de la Unión 
Europea que desean residir en España por un período superior a tres meses deben solicitar un certificado de registro o una tarjeta de residencia de familiar de ciudadano de la Unión.

📄 Documentos fuente con puntuación:
Documento: c:\Users\zepol\Documents\UAM\TFM\extranjerIA\data\pdfs\Indice_Documentacion_Esencial.pdf0.pdf - Score: 0.5161
Documento: c:\Users\zepol\Documents\UAM\TFM\extranjerIA\data\pdfs\AUT_TRABJ_Familiares_UE_sin_TIE.pdf.pdf - Score: 0.4934
Documento: c:\Users\zepol\Documents\UAM\TFM\extranjerIA\data\pdfs\Indice_Documentacion_Esencial.pdf0.pdf - Score: 0.4611

✅ Proceso de generación completado exitosamente.
