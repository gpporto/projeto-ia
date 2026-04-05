import os
from openai import OpenAI
from pypdf import PdfReader
import faiss
import numpy as np

# 🔑 sua chave

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# 📄 ler PDF
reader = PdfReader("arquivo.pdf")
text = ""

for page in reader.pages:
    text += page.extract_text() + "\n"

# ✂️ dividir texto
chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# 🧠 gerar embeddings
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

embeddings = [get_embedding(chunk) for chunk in chunks]

# 💾 criar índice FAISS
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("\n📄 PDF carregado! Faça sua pergunta:\n")

# 🔎 loop de perguntas
while True:
    pergunta = input("👉 Pergunta: ")

    query_embedding = get_embedding(pergunta)

    D, I = index.search(np.array([query_embedding]), k=3)

    context = "\n".join([chunks[i] for i in I[0]])

    prompt = f"""
    Responda com base no contexto abaixo:

    {context}

    Pergunta: {pergunta}
    """

    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\n🤖 Resposta:", resposta.choices[0].message.content, "\n")