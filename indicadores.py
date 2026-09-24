"""Indicadores: regras sobre a lista de vítimas e catálogo editável.

Os de tipo "regra" são recalculados a cada alteração na lista de vítimas — um
ferido que morre pode transformar a ocorrência em chacina. Os de tipo "texto"
vêm da classificação das notícias (BERTimbau + skill da LLM).
"""
import json
from pathlib import Path

FONTE = Path(__file__).parent / "indicadores_fonte.json"

# Só entram nas regras as pessoas efetivamente atingidas por disparo.
SITUACOES_BALEADAS = {"morta", "ferida"}


def carregar() -> list[dict]:
    return json.loads(FONTE.read_text(encoding="utf-8"))["indicadores"]


def salvar(indicadores: list[dict]) -> None:
    dados = json.loads(FONTE.read_text(encoding="utf-8"))
    dados["indicadores"] = indicadores
    FONTE.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def rotulos() -> dict[str, str]:
    return {i["nome"]: i.get("rotulo", i["nome"]) for i in carregar()}


def _avaliar_regra(regra: dict, vitimas: list[dict]) -> bool:
    baleadas = [v for v in vitimas if (v.get("situacao") or "") in SITUACOES_BALEADAS]
    campo = regra.get("campo")
    valor = regra.get("valor")
    operacao = regra.get("operacao")

    if operacao == "campo_menor_que":
        return any(
            v.get(campo) is not None and float(v[campo]) < float(valor)
            for v in baleadas
        )
    if operacao == "campo_igual":
        return any((v.get(campo) or "") == valor for v in baleadas)
    if operacao == "campo_preenchido":
        return any(str(v.get(campo) or "").strip() for v in baleadas)
    if operacao == "contagem_minima":
        total = sum(1 for v in baleadas if (v.get(campo) or "") == valor)
        return total >= int(regra.get("minimo", 1))
    return False


def calcular(vitimas: list[dict], indicadores_texto: list[str]) -> list[str]:
    """Consolida os indicadores de regra (sobre as vítimas) com os de texto
    (vindos da classificação). Uma ocorrência pode receber vários."""
    do_texto = set(indicadores_texto or [])
    resultado = []
    for indicador in carregar():
        if indicador["tipo"] == "regra":
            if _avaliar_regra(indicador.get("regra", {}), vitimas):
                resultado.append(indicador["nome"])
        elif indicador["nome"] in do_texto:
            resultado.append(indicador["nome"])
    return resultado
