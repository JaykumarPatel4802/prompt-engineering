# We will follow the tutorial by
# Aashi Dutt at
# [https://www.datacamp.com/tutorial/deepseek-r1-rag](https://www.datacamp.com/tutorial/deepseek-r1-rag)
# reproduced below for convenience.

# pip install langchain chromadb gradio ollama pymupdf langchain_ollama langchain_chroma
# pip install -U langchain-community

import ollama
import re
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# . from langchain_community.embeddings import OllamaEmbeddings
from chromadb.config import Settings
from chromadb import Client

# . from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# . Load the document using PyMuPDFLoader
loader = PyMuPDFLoader("Xiao2025.pdf")

documents = loader.load()

# . Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = text_splitter.split_documents(documents)

# . Initialize Ollama embeddings using llama3.2:1b
embedding_function = OllamaEmbeddings(model="llama3.2:1b")


# . Parallelize embedding generation
def generate_embedding(chunk):
    return embedding_function.embed_query(chunk.page_content)


# . Following step takes about 6 minutes on my M1 Macbook Pro with 32GB RAM
with ThreadPoolExecutor() as executor:
    embeddings = list(executor.map(generate_embedding, chunks))

# . Initialize Chroma client and create/reset the collection
client = Client(Settings())
# client.delete_collection(name="foundations_of_llms")  # Delete existing collection (if any)
collection = client.create_collection(name="foundations_of_llms")
# . Add documents and embeddings to Chroma
for idx, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk.page_content],
        metadatas=[{"id": idx}],
        embeddings=[embeddings[idx]],
        ids=[str(idx)],  # Ensure IDs are strings
    )

# . Initialize retriever using Ollama embeddings for queries
retriever = Chroma(
    collection_name="foundations_of_llms",
    client=client,
    embedding_function=embedding_function,
).as_retriever()


def retrieve_context(question):
    # Retrieve relevant documents
    results = retriever.invoke(question)
    # Combine the retrieved content
    context = "\n\n".join([doc.page_content for doc in results])
    return context


def query_llama(question, context):
    # Format the input prompt
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    # Query llama3.2:1b using Ollama
    response = embedding_function.chat(
        model="llama3.2:1b", messages=[{"role": "user", "content": formatted_prompt}]
    )
    # Clean and return the response
    response_content = response["message"]["content"]
    final_answer = re.sub(
        r"<think>.*?</think>", "", response_content, flags=re.DOTALL
    ).strip()
    return final_answer


def ask_question(question):
    # Retrieve context and generate an answer using RAG
    context = retrieve_context(question)
    answer = query_llama(question, context)
    return answer


# . Set up the Gradio interface
interface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs="text",
    title="RAG Chatbot: Foundations of LLMs",
    description="Ask any question about the Foundations of LLMs book. Powered by llama3.2:1b.",
)
interface.launch(debug=True)
