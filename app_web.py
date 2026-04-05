import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import faiss
import numpy as np

# 🔑 API KEY
client = OpenAI(api_key="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")

st.set_page_config(page_title="Chat com PDF", layout="wide")

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