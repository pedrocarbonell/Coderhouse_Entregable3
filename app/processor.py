from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

PERSIST_DIR = "./vectorstore"
COLLECTION_NAME = "historia_espana"


def process_documents(docs):
    print(f"Documentos cargados: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-3-small",
        chunk_size=500,
        chunk_overlap=70,
    )

    chunks = splitter.split_documents(docs)

    print(f"Fragmentos generados: {len(chunks)}")

    return chunks


def persist(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    ya_existe_indice = os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0

    if ya_existe_indice:
        print("Indice existente detectado - cargando sin reindexar")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
    else:
        print("No hay indice previo - indexando por primera vez")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
        )

    print(f"Documentos en la colección: {vectorstore._collection.count()}")
