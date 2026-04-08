import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import faiss
import numpy as np
import os
import uuid
from supabase import create_client, Client

client = OpenAI(api_key=os.getenv("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"))

supabase: Client = create_client(
    st.write("URL:", os.getenv("SUPABASE_URL")),
    st.write("KEY começa com:", str(os.getenv("SUPABASE_KEY"))[:20])
)

st.set_page_config(page_title="Chat com PDF", layout="wide")
st.title("📄 Chat com PDF")

uploaded_file = st.file_uploader("Envie um PDF", type="pdf")

if uploaded_file:
    file_id = f"{uuid.uuid4()}.pdf"
    file_bytes = uploaded_file.getvalue()

    try:
        supabase.storage.from_("pdfs").upload(
            file_id,
            file_bytes,
            {"content-type": "application/pdf", "upsert": "true"}
        )
        st.success(f"PDF salvo: {file_id}")
    except Exception as e:
        st.error(f"Erro upload: {str(e)}")
        st.stop()

    st.success(f"PDF salvo com sucesso: {file_id}")

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