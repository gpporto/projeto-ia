import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import faiss
import numpy as np
import os

# 🔑 OpenAI (Render usa variável de ambiente)
client = OpenAI(api_key=os.getenv("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"))

# 🎨 Layout bonito
st.set_page_config(
    page_title="Assistente de Editais com IA",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Assistente de Análise de Editais com IA")

st.markdown("""
Envie um edital em PDF e faça perguntas como:

- Qual o objeto da licitação?
- Quais são os principais requisitos?
- Existem riscos ou exigências críticas?
- Qual o prazo e condições importantes?

---
""")

# Upload
uploaded_file = st.file_uploader("📎 Envie seu PDF", type="pdf")

if uploaded_file is not None:

    # 📄 Ler PDF
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    if not text.strip():
        st.warning("Não foi possível extrair texto do PDF.")
        st.stop()

    st.success("✅ PDF carregado com sucesso!")

    # ✂️ Dividir texto
    chunks = [text[i:i+800] for i in range(0, len(text), 800)]

    # 🧠 Criar embeddings
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

    # 💬 Pergunta
    pergunta = st.text_input("💬 Faça uma pergunta sobre o documento:")

    if pergunta:

        query_emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=pergunta
        ).data[0].embedding

        D, I = index.search(np.array([query_emb]), k=3)

        contexto = "\n".join([chunks[i] for i in I[0]])

        prompt = f"""
Você é um especialista em análise de editais e documentos.

Responda de forma clara, objetiva e profissional.

Contexto:
{contexto}

Pergunta:
{pergunta}
"""

        resposta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        st.markdown("### 🤖 Resposta")
        st.write(resposta.choices[0].message.content)