# ================== IMPORTS ==================
import os
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# ================== LOAD API KEY ==================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ================== LOAD PDFS ==================
def load_pdfs(folder_path):
    documents = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents.append(text)
    
    return documents


docs = load_pdfs("data")
print(f"Loaded {len(docs)} documents")


# ================== TEXT SPLITTING ==================
def split_text(documents, chunk_size=500, overlap=50):
    chunks = []
    
    for doc in documents:
        for i in range(0, len(doc), chunk_size - overlap):
            chunk = doc[i:i+chunk_size]
            chunks.append(chunk)
    
    return chunks


chunks = split_text(docs)
print(f"Created {len(chunks)} chunks")


# ================== EMBEDDINGS ==================
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

print("Embedding shape:", embeddings.shape)


# ================== FAISS INDEX ==================
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("FAISS index built successfully!")


# ================== RETRIEVAL ==================
def search(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [chunks[i] for i in indices[0]]
    return results


# ================== RAG GENERATION ==================
def generate_answer(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    retrieved_chunks = [chunks[i] for i in indices[0]]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an agriculture expert assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    return answer, retrieved_chunks



# ================== RUN ==================
if __name__ == "__main__":
    question = input("Ask a question: ")
    answer, sources = generate_answer(question)

    print("\n📌 Answer:\n")
    print(answer)

    print("\n📚 Sources Used:\n")
    for i, src in enumerate(sources):
        print(f"\nSource {i+1}:\n{src[:300]}...")

if __name__ == "__main__":
    question = input("Ask a question: ")

    answer, sources = generate_answer(question)

    print("\n📌 Answer:\n")
    print(answer)

    print("\n📚 Sources Used:\n")
    for i, src in enumerate(sources):
        print(f"\nSource {i+1}:\n{src[:300]}...")