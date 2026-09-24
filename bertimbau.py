"""Filtro rápido com BERTimbau (etapa 4): pré-seleciona os indicadores de texto
antes da confirmação pela LLM.

Só funciona com um checkpoint AJUSTADO para os indicadores — o BERTimbau puro
(`neuralmind/bert-base-portuguese-cased`) não tem cabeça de classificação
treinada e devolveria rótulos aleatórios, o que é pior que não filtrar. Enquanto
não houver checkpoint, `classificar` devolve None e o pipeline usa só a LLM.

Para ativar: treine um modelo multirrótulo com os nomes dos indicadores de tipo
"texto" do indicadores_fonte.json e aponte BERTIMBAU_MODELO no .env para a pasta
do checkpoint (ou para um repositório do Hugging Face).
"""
import os

_pipeline = None
_carregado = False

LIMIAR = float(os.getenv("BERTIMBAU_LIMIAR", "0.5"))


def disponivel() -> bool:
    return bool(os.getenv("BERTIMBAU_MODELO"))


def _carregar():
    global _pipeline, _carregado
    if _carregado:
        return _pipeline
    _carregado = True
    if not disponivel():
        return None
    from transformers import pipeline

    _pipeline = pipeline(
        "text-classification",
        model=os.getenv("BERTIMBAU_MODELO"),
        top_k=None,
        truncation=True,
        max_length=512,
    )
    return _pipeline


def classificar(texto: str) -> list[str] | None:
    """Indicadores de texto sugeridos, ou None quando o filtro está desativado.

    A LLM continua sendo quem confirma: esta lista entra no prompt como
    sugestão, nunca como decisão final.
    """
    modelo = _carregar()
    if modelo is None:
        return None
    resultados = modelo(texto)[0]
    return [r["label"] for r in resultados if r["score"] >= LIMIAR]
