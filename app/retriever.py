from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

PERSIST_DIR = "./vectorstore"
COLLECTION_NAME = "historia_espana"

def retrieve_documents(query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k":5},
    )

    results = retriever.invoke(query)
    
    return results

    """
    resultados_prueba = retriever.invoke(query)
    for i, doc in enumerate(resultados_prueba, 1):
        print(f"--- Fragmento {i} (fuente: {doc.metadata['source']}) ---")
        print(doc.page_content[:150], "...\n")
    """