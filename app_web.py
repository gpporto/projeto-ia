import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import faiss
import numpy as np
import os

# 🔑 API KEY
#client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
st.set_page_config(page_title="Chat com PDF", layout="wide")

client = OpenAI(api_key="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")

st.title("📄 Chat com PDF (IA)")
st.write("Faça upload de um PDF e converse com ele")

# Upload do arquivo
uploaded_file = st.file_uploader("📎 Envie seu PDF", type="pdf")

if uploaded_file is not None:

    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    # dividir texto
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    st.success("✅ PDF carregado com sucesso!")

    # embeddings
    @st.cache_resource
    def create_index(chunks):
        embeddings = []
        for chunk in chunks:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            embeddings.append(response.data[0].embedding)

        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        return index, embeddings

    index, embeddings = create_index(chunks)

    pergunta = st.text_input("💬 Faça uma pergunta sobre o PDF:")

    if pergunta:

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=pergunta
        )
        query_embedding = response.data[0].embedding

        D, I = index.search(np.array([query_embedding]), k=3)

        context = "\n".join([chunks[i] for i in I[0]])

        prompt = f"""
        Você é um especialista em análise de documentos.

        Responda de forma clara e objetiva.

        Contexto:
        {context}

        Pergunta:
        {pergunta}
        """

        resposta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        st.write("🤖 Resposta:")
        st.write(resposta.choices[0].message.content)
