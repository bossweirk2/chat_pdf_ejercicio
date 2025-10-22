import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Asistente Repostero", page_icon="🍰", layout="wide")

# --- TÍTULO Y DESCRIPCIÓN ---
st.markdown("""
# 🍰 Asistente Repostero: Análisis de Recetas y Técnicas
Bienvenido al **Asistente Repostero**, tu ayudante experto para entender, analizar y dominar cualquier texto sobre pastelería, técnicas de postres o recetas.  
Sube un PDF con recetas o teoría y hazle preguntas al estilo de un chef pastelero profesional.  
""")

st.caption(f"Versión de Python: {platform.python_version()}")

# --- IMAGEN DE PRESENTACIÓN ---
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🥄 Panel del Chef Repostero")
    st.info("Aquí podrás subir tu documento y conversar sobre sus contenidos como si hablaras con un maestro pastelero.")
    st.markdown("**Consejo:** Puedes subir guías, recetarios o material técnico sobre postres y pedir análisis específicos.")

# --- API KEY ---
ke = st.text_input('🔑 Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# --- CARGA DEL PDF ---
pdf = st.file_uploader("📄 Carga tu archivo PDF de postres o recetas", type="pdf")

# --- PROCESAMIENTO DEL PDF ---
if pdf is not None and ke:
    try:
        # Extraer texto
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.success(f"Texto extraído correctamente. Total de {len(text)} caracteres.")
        
        # Dividir en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.info(f"Documento dividido en {len(chunks)} fragmentos de conocimiento repostero.")

        # Crear base de conocimiento
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # --- SECCIÓN DE PREGUNTAS RECOMENDADAS ---
        st.subheader("🍮 Preguntas recomendadas")
        st.markdown("Selecciona alguna o escribe la tuya propia:")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("¿Qué técnicas se usan para lograr una textura esponjosa?"):
                user_question = "¿Qué técnicas se usan para lograr una textura esponjosa en los postres?"
            elif st.button("¿Cómo equilibrar dulzura y acidez?"):
                user_question = "¿Cómo se puede equilibrar la dulzura y acidez en un postre?"
            else:
                user_question = None
        with col2:
            if st.button("Errores comunes en masas o cremas"):
                user_question = "¿Cuáles son los errores más comunes al preparar masas o cremas?"
        with col3:
            if st.button("Principios de la decoración moderna"):
                user_question = "¿Cuáles son los principios de la decoración moderna en pastelería?"

        # Campo para pregunta personalizada
        user_custom_question = st.text_area("O escribe tu pregunta:", placeholder="Ejemplo: ¿Qué temperatura ideal se usa para hornear un soufflé?")
        if user_custom_question.strip():
            user_question = user_custom_question

        # --- RESPUESTA ---
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.markdown("### 🧁 Respuesta del Chef Repostero:")
            st.markdown(response)

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar.")
else:
    st.info("📚 Carga un PDF de recetas o teoría repostera para comenzar tu análisis dulce.")

