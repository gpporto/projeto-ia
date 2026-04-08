import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import faiss
import numpy as np
import os

# 🔑 OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🎨 Layout
st.set_page_config(
    page_title="Assistente de Editais",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Assistente de Análise de Editais com IA")

st.markdown("""
Envie um edital em PDF e faça perguntas sobre o conteúdo.
""")

# 📌 Inicializa histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

# 📎 Upload
uploaded_file = st.file_uploader("📎 Envie seu PDF", type="pdf")

if uploaded_file is not None:

    # 📌 TOPO FIXO
    header = st.container()
    with header:
        st.success("✅ PDF carregado com sucesso!")
        st.markdown("---")

    # 📄 Ler PDF
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    if not text.strip():
        st.warning("Não foi possível extrair texto do PDF.")
        st.stop()

    # ✂️ Dividir texto
    chunks = [text[i:i+800] for i in range(0, len(text), 800)]

    # 🧠 Index
    @st.cache_resource
    def create_index(chunks):
        embeddings = []
        for chunk in chunks:
            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            embeddings.append(emb.data[0].embedding)

        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings))
        return index

    index = create_index(chunks)

    # 💬 INPUT PRIMEIRO (ESSA É A CHAVE DO BUG)
    pergunta = st.chat_input("Faça sua pergunta")

    if pergunta:

        # salva pergunta
        st.session_state.messages.append({
            "role": "user",
            "content": pergunta
        })

        # gera resposta
        query_emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=pergunta
        ).data[0].embedding

        D, I = index.search(np.array([query_emb]), k=3)

        contexto = "\n".join([chunks[i] for i in I[0]])

        resposta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Contexto:\n{contexto}\n\nPergunta: {pergunta}"
            }]
        )

        resposta_texto = resposta.choices[0].message.content

        # salva resposta
        st.session_state.messages.append({
            "role": "assistant",
            "content": resposta_texto
        })

    # 📜 MOSTRA HISTÓRICO (SEMPRE POR ÚLTIMO)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**🧑 Pergunta:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Resposta:** {msg['content']}")