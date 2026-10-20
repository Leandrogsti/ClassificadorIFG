"""Estados, municípios e bairros: leitura e edição da lista fechada.

POR QUE LISTA FECHADA. Cidade e bairro digitados à mão produzem grafias
diferentes para o mesmo lugar — "Vigário Geral" e "Vigario Geral" viram dois
lugares — e isso quebra o alerta de duplicidade e as estatísticas, que é o que
o registro existe para sustentar.

A CHAVE É O ESTADO, NÃO A CIDADE. Havia aqui um mapa de cidade para UF, e ele
não sobrevive à escala nacional: 232 nomes de município se repetem entre
estados, e "Bom Jesus" existe em cinco. Com a cidade como chave, quatro desses
cinco receberiam a UF errada. Por isso a estrutura é estado → cidades →
bairros, e o formulário pede o estado primeiro.

DE ONDE VEM CADA NÍVEL:

- estados (27) e municípios (5.571): API de Localidades do IBGE. São oficiais e
  completos.
- bairros: NÃO existe lista pronta. O IBGE não publica esse nível — o endpoint
  `/localidades/bairros` responde 404 — e os distritos não substituem: o Rio de
  Janeiro tem UM distrito no IBGE e 164 bairros oficiais. Os que estão no
  arquivo foram conferidos à mão, e os demais entram pela tela /locais conforme
  a cobertura cresce.
"""
import json
import unicodedata
from pathlib import Path

FONTE = Path(__file__).parent / "locais.json"


def carregar() -> dict:
    return json.loads(FONTE.read_text(encoding="utf-8"))


def salvar(dados: dict) -> None:
    FONTE.write_text(
        json.dumps(dados, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def _normalizar(texto: str) -> str:
    """Para comparar nomes sem depender de acento nem de caixa.

    Serve só para DETECTAR repetição ao acrescentar; o nome gravado é o que a
    pessoa digitou, porque "Vigário Geral" é a grafia certa e guardar
    "vigario geral" perderia a informação.
    """
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFKD", texto or "")
        if not unicodedata.combining(c)
    )
    return " ".join(sem_acento.lower().split())


def estados() -> list[tuple[str, str]]:
    """(sigla, nome) das 27 unidades da federação, em ordem de sigla."""
    dados = carregar()["estados"]
    return [(sigla, dados[sigla]["nome"]) for sigla in sorted(dados)]


def cidades(sigla: str) -> list[str]:
    estado = carregar()["estados"].get(sigla)
    return sorted(estado["cidades"]) if estado else []


def bairros(sigla: str, cidade: str) -> list[str]:
    estado = carregar()["estados"].get(sigla)
    if not estado:
        return []
    return sorted(estado["cidades"].get(cidade) or [])


def nome_do_estado(sigla: str) -> str:
    return (carregar()["estados"].get(sigla) or {}).get("nome", sigla)


def existe_cidade(sigla: str, cidade: str) -> bool:
    estado = carregar()["estados"].get(sigla)
    return bool(estado) and cidade in estado["cidades"]


# --------------------------------------------------------------------------
# Edição pela tela
# --------------------------------------------------------------------------

def acrescentar_cidade(sigla: str, cidade: str) -> str:
    """Devolve uma mensagem do que aconteceu, para a tela repetir ao analista.

    Município novo é raro — o IBGE já traz os 5.571 — mas existe: emancipação
    acontece, e a lista do IBGE é de um momento no tempo.
    """
    cidade = (cidade or "").strip()
    if not cidade:
        return "Informe o nome da cidade."
    dados = carregar()
    estado = dados["estados"].get(sigla)
    if estado is None:
        return f"Estado desconhecido: {sigla}."
    alvo = _normalizar(cidade)
    for existente in estado["cidades"]:
        if _normalizar(existente) == alvo:
            return f"{existente} já está em {sigla}."
    estado["cidades"][cidade] = []
    salvar(dados)
    return f"{cidade} acrescentada a {sigla}."


def remover_cidade(sigla: str, cidade: str) -> str:
    dados = carregar()
    estado = dados["estados"].get(sigla)
    if estado is None or cidade not in estado["cidades"]:
        return "Cidade não encontrada."
    quantos = len(estado["cidades"][cidade])
    del estado["cidades"][cidade]
    salvar(dados)
    return (
        f"{cidade} removida de {sigla}"
        + (f", junto com {quantos} bairro(s)." if quantos else ".")
    )


def acrescentar_bairro(sigla: str, cidade: str, bairro: str) -> str:
    bairro = (bairro or "").strip()
    if not bairro:
        return "Informe o nome do bairro."
    dados = carregar()
    estado = dados["estados"].get(sigla)
    if estado is None or cidade not in estado["cidades"]:
        return "Escolha um estado e uma cidade válidos."
    lista = estado["cidades"][cidade]
    alvo = _normalizar(bairro)
    for existente in lista:
        if _normalizar(existente) == alvo:
            return f"{existente} já está em {cidade}."
    lista.append(bairro)
    estado["cidades"][cidade] = sorted(lista)
    salvar(dados)
    return f"{bairro} acrescentado a {cidade} ({sigla})."


def remover_bairro(sigla: str, cidade: str, bairro: str) -> str:
    dados = carregar()
    estado = dados["estados"].get(sigla)
    if estado is None or cidade not in estado["cidades"]:
        return "Cidade não encontrada."
    lista = estado["cidades"][cidade]
    if bairro not in lista:
        return f"{bairro} não está em {cidade}."
    lista.remove(bairro)
    salvar(dados)
    return f"{bairro} removido de {cidade} ({sigla})."


def estatisticas() -> dict:
    dados = carregar()["estados"]
    cidades_total = sum(len(e["cidades"]) for e in dados.values())
    bairros_total = sum(
        len(b) for e in dados.values() for b in e["cidades"].values()
    )
    com_bairro = sum(
        1 for e in dados.values() for b in e["cidades"].values() if b
    )
    return {
        "estados": len(dados),
        "cidades": cidades_total,
        "bairros": bairros_total,
        "cidades_com_bairro": com_bairro,
    }
