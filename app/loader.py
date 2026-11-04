from langchain_community.document_loaders import DirectoryLoader, TextLoader
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

def load_documents(folder: Path = DATA) -> list:
    loader = DirectoryLoader(
        str(folder),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
        )
    return loader.load() 