import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import faiss
import numpy as np

# 👉 CHAVE LOCAL (DEPOIS VAMOS TIRAR)
client = OpenAI(api_key="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")

st.title("📄 Chat com PDF")

uploaded_file = st.file_uploader("Envie um PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    embeddings = []
    for chunk in chunks:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(emb.data[0].embedding)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    pergunta = st.text_input("Pergunta")

    if pergunta:
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

        st.write(resposta.choices[0].message.content)