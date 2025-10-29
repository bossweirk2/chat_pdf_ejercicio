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
st.set_page_config(page_title="💼 BrainDesk", page_icon="🧠", layout="wide")

# --- ESTILO VISUAL ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #e5e5e5;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        text-align: center;
        color: #ffd700;
    }
    .stButton>button {
        background-color: #1e2a38;
        color: #ffd700;
        border-radius: 10px;
        font-weight: bold;
        border: 1px solid #ffd700;
    }
    .stButton>button:hover {
        background-color: #ffd700;
        color: #0b0f19;
    }
    </style>
""", unsafe_allow_html=True)

# --- TÍTULO Y DESCRIPCIÓN ---
st.markdown("""
# 💼 BrainDesk: Consultor Estratégico con IA
**Analiza documentos empresariales, informes o estrategias y obtén conclusiones ejecutivas, ideas de mejora y análisis profesionales.**

Sube tu documento en PDF y conversa con una IA especializada en **negocios, planeación y estrategia corporativa**.
""")

st.caption(f"Versión de Python: {platform.python_version()}")

# --- IMAGEN DE PRESENTACIÓN ---
try:
    image = Image.open('consultor.jpg')  # Reemplaza por tu propia imagen
    st.image(image, width=400)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📊 Panel de Estrategia")
    st.info("Sube un documento empresarial (plan de negocio, informe de mercado o análisis financiero) para consultarlo con IA.")
    st.markdown("**Consejo:** También puedes subir material académico o reportes para obtener un resumen ejecutivo o análisis de riesgos.")

# --- API KEY ---
ke = st.text_input('🔑 Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# --- CARGA DEL PDF ---
pdf = st.file_uploader("📄 Carga tu documento empresarial (PDF)", type="pdf")

# --- PROCESAMIENTO DEL PDF ---
if pdf is not None and ke:
    try:
        # Extraer texto del PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.success(f"Texto extraído correctamente. Total de {len(text)} caracteres.")

        # Dividir texto en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.info(f"Documento dividido en {len(chunks)} fragmentos de conocimiento empresarial.")

        # Crear base de conocimiento
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # --- SECCIÓN DE PREGUNTAS RECOMENDADAS ---
        st.subheader("🧩 Preguntas sugeridas de análisis estratégico")
        st.markdown("Selecciona alguna o escribe la tuya propia:")

        col1, col2, col3 = st.columns(3)
        user_question = None
        with col1:
            if st.button("🔍 Identifica debilidades del plan"):
                user_question = "¿Cuáles son las principales debilidades del plan de negocio?"
            if st.button("📈 Oportunidades de mejora"):
                user_question = "¿Qué oportunidades de mejora se pueden aplicar en esta estrategia?"
        with col2:
            if st.button("💰 Análisis financiero general"):
                user_question = "Haz un análisis financiero general del documento."
            if st.button("🎯 Evaluación de objetivos"):
                user_question = "¿Qué tan claros y alcanzables son los objetivos planteados?"
        with col3:
            if st.button("🧠 Resumen ejecutivo"):
                user_question = "Haz un resumen ejecutivo breve del documento."

        # Pregunta personalizada
        user_custom_question = st.text_area("O escribe tu pregunta personalizada:", 
                                            placeholder="Ejemplo: ¿Qué riesgos estratégicos se detectan en este plan?")
        if user_custom_question.strip():
            user_question = user_custom_question

        # --- RESPUESTA ---
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.markdown("### 🧠 Respuesta del Consultor Estratégico:")
            st.markdown(response)

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar.")
else:
    st.info("📚 Carga un documento en PDF para comenzar el análisis con tu consultor de negocios.")
