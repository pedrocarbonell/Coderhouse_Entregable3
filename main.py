import asyncio
from app.processor import persist, process_documents
from app.loader import load_documents
from dotenv import load_dotenv
from app.search import get_rag_response

load_dotenv()


async def main() -> None:
    docs = load_documents()
    print(f"{len(docs)} documents loaded")
    
    chunks = process_documents(docs)

    persist(chunks)

    print(f"Pregunta en el contexto: Quienes fueron los reyes catolicos")

    respuesta_ok = await get_rag_response("Quienes fueron los reyes católicos")

    print("RESPUESTA:", respuesta_ok.respuesta)
    print("FUENTES:", respuesta_ok.fuentes)
    print("Fragmentos usados:", respuesta_ok.fragmentos_recuperados)

    print("---\nPregunta fuera del contexto: Cuantos puntos hizo Michael Jordan en su carrera de NBA")

    respuesta_trampa = await get_rag_response("Cuantos puntos hizo Michael Jordan en su carrera de NBA")

    print("RESPUESTA:", respuesta_trampa.respuesta)
    print("FUENTES:", respuesta_trampa.fuentes)
    print("Fragmentos usados:", respuesta_trampa.fragmentos_recuperados)


if __name__ == "__main__":
    asyncio.run(main())
