import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import faiss
import numpy as np
import os

from supabase import create_client

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# 🔑 OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🎨 Layout
st.set_page_config(
    page_title="Assistente de Editais",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Assistente de Análise de Editais")

st.markdown("""
Envie um edital em PDF e faça perguntas sobre o conteúdo.
""")

# 📌 Inicializa histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

# ******************************
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:

    st.title("🔐 Login")

    email = st.text_input("Email")
    senha = st.text_input("Senha", type="password")

    if st.button("Entrar"):
        try:
            user = supabase.auth.sign_in_with_password({
                "email": email,
                "password": senha
            })
            st.session_state.user = user
            st.rerun()
        except:
            st.error("Login inválido")

    st.stop()
#*******************************

# 📎 Upload
uploaded_file = st.file_uploader("📎 Envie seus PDFs", type="pdf", accept_multiple_files=True)

if uploaded_file:

    # 📌 TOPO FIXO
    header = st.container()
    with header:
        st.success("✅ PDF carregado com sucesso!")
        st.markdown("---")

    # 📄 Ler PDF
    
    chunks = []
    sources = []

    for uploaded_file in uploaded_file:
        reader = PdfReader(uploaded_file)

        for page in reader.pages:
            content = page.extract_text()
            if content:
                parts = [content[i:i+800] for i in range(0, len(content), 800)]
                for p in parts:
                    chunks.append(p)
                    sources.append(uploaded_file.name)

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

        # salva pergunta no histórico local
        st.session_state.messages.append({
            "role": "user",
            "content": pergunta
        })

        # 🔥 SALVA NO BANCO (USER)
        supabase.table("mensagens").insert({
            "user_id": st.session_state.user.user.id,
            "role": "user",
            "content": pergunta
        }).execute()

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

        # salva resposta local
        st.session_state.messages.append({
            "role": "assistant",
            "content": resposta_texto
        })

        # 🔥 SALVA NO BANCO (ASSISTANT)
        supabase.table("mensagens").insert({
            "user_id": st.session_state.user.user.id,
            "role": "assistant",
            "content": resposta_texto
        }).execute()

    # 📜 MOSTRA HISTÓRICO (SEMPRE POR ÚLTIMO)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**🧑 Pergunta:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Resposta:** {msg['content']}")