import os
import chromadb
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

chat_client = AzureOpenAI(
    api_key=os.getenv("AZURE_CHAT_KEY"),
    api_version=os.getenv("AZURE_CHAT_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_CHAT_ENDPOINT")
)

test_response = client.embeddings.create(
    input=["Hello world!"],
    model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
)
print("Test embedding successful.")

chroma_client = chromadb.PersistentClient(path="./chroma_storage")
collection = chroma_client.get_or_create_collection(name="codebase")

def chunk_text(text, max_len=500):
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]

def embed(texts):
    if not texts:
        print("No texts found to embed.")
        return []
    
    print(f"Embedding {len(texts)} chunks. Example chunk:\n{texts[0][:200]}...\n")
    response = client.embeddings.create(
        model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        input=texts
    )
    return [d.embedding for d in response.data]


existing_ids = set()
BATCH_SIZE = 100
offset = 0

while True:
    results = collection.get(limit=BATCH_SIZE, offset=offset, include=["documents"])
    if not results['ids']:
        break
    existing_ids.update(results['ids'])
    offset += BATCH_SIZE

print(f"Loaded {len(existing_ids)} existing embedded chunks.")

base_dir = r"C:\Users\souravgupta\OneDrive - Deloitte (O365D)\Documents\Projects\DCAT\Code\p3_tenant_api"

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith((".cs", ".md", ".txt")):
            full_path = os.path.join(root, file)
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                chunks = chunk_text(content)

                chunk_ids = [f"{file}_{i}" for i in range(len(chunks))]

                new_chunks = []
                new_ids = []
                for i, chunk_id in enumerate(chunk_ids):
                    if chunk_id not in existing_ids:
                        new_chunks.append(chunks[i])
                        new_ids.append(chunk_id)

                if not new_chunks:
                    continue  

                vectors = embed(new_chunks)

                collection.add(
                    documents=new_chunks,
                    metadatas=[{"source": file}] * len(new_chunks),
                    ids=new_ids,
                    embeddings=vectors
                )

print("All new files embedded and stored in ChromaDB.")

def ask_question(query, top_k=3):
    print(f"\n Asking: {query}")
    
    query_embedding = client.embeddings.create(
        model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        input=[query]
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    context = "\n\n---\n\n".join(results['documents'][0])

    response = chat_client.chat.completions.create(
        model=os.getenv("AZURE_CHAT_DEPLOYMENT"), 
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions about the user's codebase written in C#."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )

    return response.choices[0].message.content

print("\nAsk any question about your codebase (type 'exit' to quit):")

while True:
    user_question = input("\n Your question: ").strip()
    if user_question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not user_question:
        print("Please enter a valid question.")
        continue

    answer = ask_question(user_question)
    print(f"\nAnswer:\n{answer}")
