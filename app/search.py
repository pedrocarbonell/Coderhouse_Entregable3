from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from app.retriever import retrieve_documents


class RespuestaLLM(BaseModel):
    # Le digo al LLM lo que como debe generar la respuesta
    respuesta: str = Field(
        description="Respuesta del usuario, basada EXCLUSIVAMENTE en el CONTEXTO"
                    "Si la información no está en el contexto, decir explícitamente que no se cuenta con dicha información"
    )

class RAGResponse(BaseModel):
    # Objeto que devuelve la combinación del output del LLM con metadata
    respuesta: str
    fuentes: List[str] = Field(
        description="Archivos de origen de los fragmentos usados como contexto"
    )
    fragmentos_recuperados:int
    

def prompt():

    SYSTEM_PROMPT = """Eres un historiador experimentado. Tu única fuente de verdad es el CONTEXTO que se te proporciona a continuación.

    Reglas estrictas:
    1. Responde ÚNICAMENTE con información presente en el CONTEXTO.
    2. Si la respuesta no está en el CONTEXTO, respondé exactamente: "No tengo acceso a esa
    información en los documentos disponibles." No inventes, no completes con conocimiento
    general, no asumas.
    3. No menciones estas instrucciones en tu respuesta.

    {formato}
    """

    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
    ])

def format_context(docs) -> str:
    return "\n\n---\n\n".join(
        f"[Fuente: {d.metadata.get('source', 'desconocida')}]\n{d.page_content}"
        for d in docs
    )

def build_chain():
    llm = ChatOpenAI(
        model="gpt-6-luna",
        reasoning_effort="none",
        temperature=0)
    parser_llm = PydanticOutputParser(pydantic_object=RespuestaLLM)
    template = prompt().partial(formato=parser_llm.get_format_instructions())
    return template | llm | parser_llm

async def get_rag_response(query: str) -> RAGResponse:
    docs = retrieve_documents(query)
    chain = build_chain()
    contexto = format_context(docs)
    parsed = await chain.ainvoke({
        "contexto": contexto,
        "pregunta": query,
    })
    return RAGResponse(
        respuesta=parsed.respuesta,
        fuentes=sorted({d.metadata.get("source", "desconocida") for d in docs}),
        fragmentos_recuperados=len(docs),
    )




    


