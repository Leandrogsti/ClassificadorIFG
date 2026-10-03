"""Predição com os modelos treinados, para os testes interativos na aplicação.

Fica antes da camada de LLM: aqui se vê o que o BERT sozinho responde, o que
permite julgar se a LLM está corrigindo o classificador ou apenas repetindo-o.

Os modelos ficam em memória depois do primeiro uso, porque carregar um BERT a
cada requisição levaria segundos.
"""
import json
from pathlib import Path

import corpus
import treinamento

_carregados: dict[str, dict] = {}


def caminho_do_modelo(tarefa: str) -> Path:
    return treinamento.PASTA_MODELOS / tarefa


def treinado(tarefa: str) -> bool:
    return (caminho_do_modelo(tarefa) / "config.json").exists()


def esquecer(tarefa: str | None = None) -> None:
    """Descarta o modelo em memória após um retreino, senão a aplicação
    continuaria respondendo com os pesos antigos."""
    if tarefa is None:
        _carregados.clear()
    else:
        _carregados.pop(tarefa, None)


def _carregar(tarefa: str) -> dict:
    if tarefa in _carregados:
        return _carregados[tarefa]

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    caminho = caminho_do_modelo(tarefa)
    if not treinado(tarefa):
        raise RuntimeError(
            f"A tarefa '{tarefa}' ainda não tem modelo treinado. Rotule notícias"
            " no repositório e rode o treino."
        )

    aparelho = treinamento.dispositivo_disponivel()
    rede = AutoModelForSequenceClassification.from_pretrained(caminho).to(aparelho)
    rede.eval()
    arquivo_rotulos = caminho / "rotulos.json"
    rotulos = (
        json.loads(arquivo_rotulos.read_text(encoding="utf-8"))
        if arquivo_rotulos.exists()
        else corpus.rotulos_da_tarefa(tarefa)
    )

    _carregados[tarefa] = {
        "rede": rede,
        "tokenizador": AutoTokenizer.from_pretrained(caminho),
        "rotulos": rotulos,
        "dispositivo": aparelho,
        "torch": torch,
    }
    return _carregados[tarefa]


def prever(tarefa: str, texto: str, limiar: float = 0.5) -> dict:
    """Devolve a resposta do modelo com a confiança de cada classe.

    A confiança é parte do resultado, não um detalhe: uma classificação correta
    com 0,51 de probabilidade e outra com 0,99 dizem coisas muito diferentes
    sobre quando confiar no filtro e quando deixar a decisão para a LLM.
    """
    modelo = _carregar(tarefa)
    torch = modelo["torch"]
    rotulos = modelo["rotulos"]

    entrada = modelo["tokenizador"](
        texto, truncation=True, padding="max_length", max_length=256,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = modelo["rede"](
            **{k: v.to(modelo["dispositivo"]) for k, v in entrada.items()}
        ).logits[0]

    if tarefa == "indicadores":
        pontuacoes = torch.sigmoid(logits).cpu().numpy()
        previstos = [rotulos[i] for i, p in enumerate(pontuacoes) if p >= limiar]
        return {
            "tarefa": tarefa,
            "previsto": previstos,
            "confianca": {rotulos[i]: float(p) for i, p in enumerate(pontuacoes)},
            "limiar": limiar,
            "dispositivo": modelo["dispositivo"],
        }

    pontuacoes = torch.softmax(logits, dim=-1).cpu().numpy()
    indice = int(pontuacoes.argmax())
    return {
        "tarefa": tarefa,
        "previsto": rotulos[indice],
        "confianca_previsto": float(pontuacoes[indice]),
        "confianca": {rotulos[i]: float(p) for i, p in enumerate(pontuacoes)},
        "dispositivo": modelo["dispositivo"],
    }


def pipeline(texto: str, limiar: float = 0.5) -> dict:
    """Encadeia as duas etapas como em produção: a classificação de motivação e
    indicadores só roda se a notícia passar pelo filtro de violência armada.
    """
    resultado = {"violencia": None, "motivacao": None, "indicadores": None}
    resultado["violencia"] = prever("violencia", texto)

    if resultado["violencia"]["previsto"] != "violencia_armada":
        resultado["interrompido_em"] = "violencia"
        return resultado

    for tarefa in ("motivacao", "indicadores"):
        if treinado(tarefa):
            resultado[tarefa] = prever(tarefa, texto, limiar)
    return resultado
