"""Esquema e acesso ao SQLite.

Duas tabelas centrais, ligadas por id: `ocorrencia` e `vitima`
(vitima.ocorrencia_id -> ocorrencia.id). As demais tabelas apoiam o fluxo:
`noticia` guarda os textos colados, `vitima_historico` registra cada alteração
e `alerta_duplicidade` guarda a checagem paralela de registros repetidos.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from vocabulario import CIRCUNSTANCIA_PADRAO, ROTULOS_CIRCUNSTANCIA

CAMINHO_BANCO = Path(__file__).parent / "classificador.db"

ESQUEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS ocorrencia (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    data_fato              TEXT NOT NULL,
    cidade                 TEXT NOT NULL,
    bairro                 TEXT NOT NULL,
    localidade             TEXT,
    analista               TEXT NOT NULL,
    criado_em              TEXT NOT NULL,
    atualizado_em          TEXT NOT NULL,
    status                 TEXT NOT NULL DEFAULT 'nao_aprovado',
    motivo_principal       TEXT,
    motivos_complementares TEXT,
    indicadores            TEXT,
    indicadores_texto      TEXT,
    divergencias           TEXT,
    aprovado_por           TEXT,
    aprovado_em            TEXT,
    observacao_aprovacao   TEXT,
    unificada_em           INTEGER REFERENCES ocorrencia(id)
);

-- A busca de duplicidade é sempre por data + cidade + bairro.
CREATE INDEX IF NOT EXISTS idx_ocorrencia_duplicidade
    ON ocorrencia(data_fato, cidade, bairro);

CREATE TABLE IF NOT EXISTS noticia (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ocorrencia_id INTEGER NOT NULL REFERENCES ocorrencia(id) ON DELETE CASCADE,
    link          TEXT,
    texto         TEXT NOT NULL,
    texto_processado TEXT,
    criado_em     TEXT NOT NULL,
    -- preenchido só quando a notícia veio de outra ocorrência numa unificação,
    -- para que o "desfazer" saiba para onde devolvê-la
    ocorrencia_origem_id INTEGER REFERENCES ocorrencia(id)
);

CREATE INDEX IF NOT EXISTS idx_noticia_ocorrencia ON noticia(ocorrencia_id);

CREATE TABLE IF NOT EXISTS vitima (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ocorrencia_id       INTEGER NOT NULL REFERENCES ocorrencia(id) ON DELETE CASCADE,
    chave               TEXT,
    nome                TEXT,
    idade               INTEGER,
    genero              TEXT,
    tipo_vitima         TEXT,
    situacao            TEXT,
    data_morte          TEXT,
    circunstancia       TEXT,
    cargo_politico      TEXT,
    fonte               TEXT,
    divergencia         TEXT,
    editado_manualmente INTEGER NOT NULL DEFAULT 0,
    -- versão sugerida pela LLM quando ela discorda de uma correção manual
    -- (ex.: ferida que passou a morta numa notícia nova), à espera do analista
    proposta            TEXT,
    criado_em           TEXT NOT NULL,
    atualizado_em       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vitima_ocorrencia ON vitima(ocorrencia_id);

CREATE TABLE IF NOT EXISTS vitima_historico (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    vitima_id      INTEGER NOT NULL REFERENCES vitima(id) ON DELETE CASCADE,
    ocorrencia_id  INTEGER NOT NULL REFERENCES ocorrencia(id) ON DELETE CASCADE,
    campo          TEXT NOT NULL,
    valor_anterior TEXT,
    valor_novo     TEXT,
    alterado_por   TEXT,
    alterado_em    TEXT NOT NULL,
    fonte          TEXT
);

CREATE INDEX IF NOT EXISTS idx_historico_vitima ON vitima_historico(vitima_id);

CREATE TABLE IF NOT EXISTS alerta_duplicidade (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    ocorrencia_id         INTEGER NOT NULL REFERENCES ocorrencia(id) ON DELETE CASCADE,
    ocorrencia_similar_id INTEGER NOT NULL REFERENCES ocorrencia(id) ON DELETE CASCADE,
    status                TEXT NOT NULL DEFAULT 'aberto',
    criado_em             TEXT NOT NULL,
    resolvido_por         TEXT,
    resolvido_em          TEXT,
    motivo                TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerta_ocorrencia ON alerta_duplicidade(ocorrencia_id);
"""


def agora() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def conectar() -> sqlite3.Connection:
    con = sqlite3.connect(CAMINHO_BANCO)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def iniciar() -> None:
    with conectar() as con:
        con.executescript(ESQUEMA)
        _migrar(con)


def _migrar(con) -> None:
    """Ajusta bancos criados antes de uma mudança de esquema."""
    colunas = {linha[1] for linha in con.execute("PRAGMA table_info(vitima)")}

    # "bala perdida" virou um dos valores de `circunstancia`; manter a coluna
    # booleana permitiria que os dois campos se contradissessem.
    if "bala_perdida" in colunas:
        con.execute(
            "UPDATE vitima SET circunstancia = 'bala_perdida' WHERE bala_perdida = 1"
        )
        con.execute("ALTER TABLE vitima DROP COLUMN bala_perdida")

    # `circunstancia` era texto livre e passou a ser lista fechada.
    marcadores = ",".join("?" for _ in ROTULOS_CIRCUNSTANCIA)
    con.execute(
        "UPDATE vitima SET circunstancia = ?"
        f" WHERE circunstancia IS NULL OR circunstancia NOT IN ({marcadores})",
        (CIRCUNSTANCIA_PADRAO, *ROTULOS_CIRCUNSTANCIA),
    )

    # Vítima ilesa deixou de existir: só entra na lista quem foi atingido.
    con.execute("DELETE FROM vitima WHERE situacao = 'ilesa'")


def json_carregar(valor, padrao):
    if not valor:
        return padrao
    try:
        return json.loads(valor)
    except (json.JSONDecodeError, TypeError):
        return padrao
