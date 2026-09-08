import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

import preprocessar_noticia

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key or api_key.startswith("cole_sua"):
    raise SystemExit(
        "GROQ_API_KEY nao configurada. Crie ou edite o arquivo .env na "
        "raiz do projeto e informe sua chave."
    )

client = Groq(api_key=api_key)
model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if "prompt-guard" in model.lower():
    raise SystemExit(
        "O modelo Prompt Guard detecta ataques, mas nao classifica noticias. "
        "Configure no .env um modelo generativo, como openai/gpt-oss-20b."
    )

pre_prompt = Path("dicionario_categorias.txt").read_text(encoding="utf-8")
noticia = Path("noticia.txt").read_text(encoding="utf-8")

if not pre_prompt.strip():
    raise SystemExit("O arquivo dicionario_categorias.txt esta vazio.")
if not noticia.strip():
    raise SystemExit("O arquivo noticia.txt esta vazio.")

noticia = preprocessar_noticia.resumir(noticia)
Path("noticia_processada.txt").write_text(noticia, encoding="utf-8")

resposta = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": pre_prompt,
        },
        {
            "role": "user",
            "content": noticia,
        }
    ],
    temperature=0,
    response_format={"type": "json_object"},
    reasoning_effort="low",
    max_tokens=1200,
)

escolha = resposta.choices[0]
conteudo = escolha.message.content

if escolha.finish_reason == "length":
    print(
        "Aviso: a resposta foi cortada por atingir o limite de tokens "
        "(max_tokens). Considere reduzir o texto de entrada ou aumentar "
        "max_tokens ainda mais, respeitando o limite do modelo."
    )

try:
    resultado = json.loads(conteudo)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
except json.JSONDecodeError:
    print(conteudo)