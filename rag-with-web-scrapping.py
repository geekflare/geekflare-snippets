import os
import socket
from dotenv import load_dotenv

import chromadb
from geekflare_api.client import GeekflareClient
from geekflare_api.models import WebScrapeDto
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load environment variables
load_dotenv()

# Force IPv4 for outbound connections. Some networks silently drop the
# larger packets used in a TLS handshake over IPv6 (PMTU black hole) even
# though the initial TCP handshake succeeds, causing hangs/timeouts.
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = _ipv4_only_getaddrinfo

# Initialize the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="website_knowledge"
)

# Initialize the LLM client
llm = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

LLM_MODEL = os.getenv("LLM_MODEL")


def scrape_website(url: str):
    """Scrape a website and return Markdown content."""

    with GeekflareClient(api_key=os.getenv("GEEKFLARE_API_KEY")) as client:
        result = client.web_scrape(
            WebScrapeDto(
                url=url,
                format=["markdown"],
                render_js=False,
                block_ads=True
            )
        )

    if "data" not in result:
        raise RuntimeError(f"Scrape failed: {result}")

    return result["data"]["markdown"]


def split_text(text, chunk_size=1000):
    """Split text into chunks."""

    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


def store_documents(chunks):
    """Generate embeddings and store them in ChromaDB."""

    embeddings = embedding_model.encode(chunks).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(chunks))]
    )


def retrieve_context(query):
    """Retrieve the most relevant chunks."""

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    return "\n\n".join(results["documents"][0])


def ask_llm(question, context):
    """Generate a response using the retrieved context."""

    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}
"""

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def main():

    url = input("Website URL: ")

    print("Scraping website...")
    markdown = scrape_website(url)

    print("Creating embeddings...")
    chunks = split_text(markdown)
    store_documents(chunks)

    print("Knowledge base ready.\n")

    while True:

        question = input("Ask a question (or 'exit'): ")

        if question.lower() == "exit":
            break

        context = retrieve_context(question)
        answer = ask_llm(question, context)

        print("\nAnswer:\n")
        print(answer)
        print()


if __name__ == "__main__":
    main()