"""Listas fechadas que descrevem uma vítima.

Ficam num módulo próprio porque valem para o armazenamento (db), para a
validação da resposta da LLM (classificador) e para os formulários (app).
"""

import json
from pathlib import Path

GENEROS = ["feminino", "masculino", "nao_informado"]
TIPOS_VITIMA = ["civil", "agente_seguranca", "politico", "suspeito", "nao_informado"]

# Só entra na lista quem foi atingido por disparo: não existe vítima ilesa.
SITUACOES = ["ferida", "morta"]

# "Bala perdida" e "Chacina" são circunstâncias, não campos à parte, para que
# não existam duas respostas possíveis para a mesma pergunta.
CIRCUNSTANCIAS = [
    ("feminicidio_tentativa", "Feminicídio/tentativa"),
    ("suicidio", "Suicídio"),
    ("acidente", "Acidente"),
    ("trajeto_escolar", "Trajeto escolar"),
    ("bala_perdida", "Bala perdida"),
    ("chacina", "Chacina"),
    ("lgbtqiapn", "LGBTQIAPN+"),
    ("tribunal_do_crime", "Tribunal do crime"),
    ("vitima_de_agente_de_seguranca", "Vítima de agente de segurança"),
    ("nao_se_aplica", "Não se aplica"),
]

ROTULOS_CIRCUNSTANCIA = dict(CIRCUNSTANCIAS)
CIRCUNSTANCIA_PADRAO = "nao_se_aplica"

_FONTE_CATEGORIAS = Path(__file__).parent / "categorias_fonte.json"


def motivacoes() -> list[dict]:
    """Categorias de motivação, lidas da mesma fonte que gera o prompt da LLM,
    para que o rótulo do corpus e a classe do modelo nunca divirjam."""
    dados = json.loads(_FONTE_CATEGORIAS.read_text(encoding="utf-8"))
    return dados["categorias"]


def slugs_motivacao() -> list[str]:
    return [categoria["nome"] for categoria in motivacoes()]
