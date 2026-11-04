# RAG sobre historia de España

Lee los markdown de `data/`, los fragmenta, los indexa en Chroma y responde dos preguntas: una del corpus y una fuera de contexto.

## Requisitos

- Python 3.12
- Una API key de OpenAI

## Instalación

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Credenciales

Creá un archivo `.env` en la raíz del proyecto:

```
OPENAI_API_KEY=tu_clave
```

No subas ese archivo. Ya está listado en `.gitignore`.

## Ejecución

```bash
python main.py
```

La primera corrida indexa los documentos en `./vectorstore`. Las siguientes reutilizan ese índice. La salida imprime la respuesta, las fuentes y la cantidad de fragmentos usados.
