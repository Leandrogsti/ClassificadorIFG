"""Esquema e acesso ao SQLite, no formato do Dicionário de Dados.

SÃO TRÊS GRUPOS DE TABELAS, com propósitos que não se misturam:

1. PRODUÇÃO — o fato registrado. A espinha são `ocorrencia` (o caso) e `vitima`,
   ligadas por `vitima.ocorrencia_id`. Em volta ficam as complementares, que só
   recebem linha quando o fenômeno correspondente é observado:

    ocorrencia ─┬─ noticia                 (links e textos da ocorrência)
                ├─ acao_policial           (1:1, se o fenômeno foi observado)
                ├─ crime_organizado        (1:1, se o fenômeno foi observado)
                ├─ alerta_duplicidade
                └─ vitima ─┬─ vitimizacao_policial  (1:1, se a vítima é agente)
                           ├─ politico_baleado      (1:1, se a vítima é política)
                           └─ vitima_historico

2. TRIAGEM — `recebida`: a notícia ANTES de virar ocorrência, com link, a data
   que o usuário digitou e o texto. É o que permite guardar a notícia que não
   vira registro nenhum, e essa é justamente o exemplo negativo que ensina o
   classificador a recusar o que está fora do escopo. A coluna `estado` carrega
   a faixa de concordância. São SEIS: `na_fila` (ninguém olhou), `arquivo`
   (0 votos; acervo que não vai à LLM e não conta como pendência), `alerta`
   (abaixo do corte), `com_fenomeno` e `sem_fenomeno` (pessoa conferiu) e
   `virou_ocorrencia`.

3. TREINO E AUDITORIA — `corpus_noticia` (exemplo rotulado, com a partição
   treino/validação/teste fixada), `treino_execucao` (métricas de cada treino),
   `entrada` e `entrada_classificacao` (o texto que chegou e o que o BERT marcou
   nele, gravados ANTES de a LLM ver — sem isso o erro do classificador
   desaparece assim que a LLM o corrige), e `classificacao_bert`.

Uma notícia vive em até três lugares ao mesmo tempo, e isso é intencional: em
`recebida` como item de fila, em `noticia` como fonte de um fato, e em
`corpus_noticia` como exemplo de treino. Os três guardam o mesmo texto com
propósitos diferentes, e a coluna `impressao` (sha1 do texto normalizado) é o
que permite reconhecer que são a mesma notícia sem compará-la caractere a
caractere.

Separar as complementares em tabelas próprias, em vez de pendurar as colunas
em `vitima` e `ocorrencia`, é o que evita dezenas de colunas nulas em toda
linha: corporação e patente só existem para agente, partido e mandato só para
político.

Campos de marcação múltipla (fenômenos observados, circunstâncias, tipo de
vítima, contextos) são guardados como JSON de lista. Os de marcação única são
texto validado contra as listas fechadas de `vocabulario`.

`ID_CASO` e `ID_VÍTIMA` são gerados por gatilho a partir do id interno, nunca
em código: assim nenhum caminho de inserção pode esquecer de preenchê-los.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

CAMINHO_BANCO = Path(__file__).parent / "classificador.db"

# Prefixo dos códigos públicos. O dicionário distingue o que foi registrado
# antes da reestruturação (PRE) do que veio depois (POS); esta base nasce
# depois, então tudo que ela gera é POS, escrito direto nos dois gatilhos de
# `id_caso` e `id_vitima`. Havia uma constante aqui, sem nenhum uso: mudá-la não
# mudava nada, o que é pior do que não existir.

ESQUEMA = """
PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------------ CASO --
CREATE TABLE IF NOT EXISTS ocorrencia (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    -- ID_CASO: preenchido pelo gatilho abaixo, nunca pela aplicação.
    id_caso       TEXT UNIQUE,
    data_registro TEXT NOT NULL,
    -- DATA DA NOTÍCIA: não aparece no formulário, vem da coleta.
    data_noticia  TEXT,
    data_caso     TEXT NOT NULL,
    hora_caso     TEXT,
    faixa_horaria TEXT NOT NULL DEFAULT 'nao_identificado',
    estado        TEXT NOT NULL,
    cidade        TEXT NOT NULL,
    bairro        TEXT NOT NULL DEFAULT 'nao_identificado',
    localidade    TEXT NOT NULL DEFAULT 'nao_identificado',
    latitude      REAL,
    longitude     REAL,
    -- FENÔMENO OBSERVADO: JSON de lista, marcação múltipla.
    fenomenos     TEXT NOT NULL DEFAULT '[]',
    -- O que o BERT marcou antes da confirmação pela LLM, guardado separado
    -- para que se possa medir depois se a LLM corrigiu o modelo ou o repetiu.
    fenomenos_bert TEXT,
    -- Classes que o BERT marcou e para as quais a LLM não achou base no texto.
    -- Elas CONTINUAM em `fenomenos` — quem revisa o classificador é o analista.
    -- Esta lista explica por que as tabelas complementares delas estão em
    -- branco: foi de propósito, não por falha da chamada.
    nao_sustentados TEXT,
    -- Classes que a LLM tentou marcar por conta própria e que o sistema
    -- recusou, porque quem decide as categorias é o BERT. Não vira
    -- classificação, mas é o melhor sinal de falso negativo do classificador —
    -- e portanto de notícia que vale anotar no corpus.
    sugeridos_pela_llm TEXT,
    -- VÍTIMAS BALEADAS?: 'S' abre a tabela de vítimas, 'N' a mantém vazia.
    vitimas_baleadas TEXT NOT NULL DEFAULT 'N',

    -- MOTIVAÇÃO: fora de uso desde 06/10/2026. O sistema registra FENÔMENO
    -- OBSERVADO e CIRCUNSTÂNCIA, e nada mais. As duas colunas continuam aqui
    -- para não perder o que bases antigas já gravaram; nada escreve nelas e
    -- nada as lê.
    motivo_principal       TEXT,
    motivos_complementares TEXT,
    divergencias           TEXT,

    -- Fluxo de trabalho da redação.
    analista             TEXT NOT NULL,
    status               TEXT NOT NULL DEFAULT 'rascunho',
    aprovado_por         TEXT,
    aprovado_em          TEXT,
    observacao_aprovacao TEXT,
    unificada_em         INTEGER REFERENCES ocorrencia(id),
    atualizado_em        TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS gera_id_caso
AFTER INSERT ON ocorrencia
WHEN NEW.id_caso IS NULL
BEGIN
    UPDATE ocorrencia SET id_caso = 'POS' || printf('%08d', NEW.id)
     WHERE id = NEW.id;
END;

-- A busca de duplicidade é sempre por data + cidade + bairro.
CREATE INDEX IF NOT EXISTS idx_ocorrencia_duplicidade
    ON ocorrencia(data_caso, cidade, bairro);

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

-- ---------------------------------------------------------------- VÍTIMA --
CREATE TABLE IF NOT EXISTS vitima (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    id_vitima     TEXT UNIQUE,
    ocorrencia_id INTEGER NOT NULL REFERENCES ocorrencia(id) ON DELETE CASCADE,
    nome          TEXT,
    status        TEXT NOT NULL,
    genero        TEXT NOT NULL DEFAULT 'nao_identificado',
    cor_raca      TEXT NOT NULL DEFAULT 'nao_identificado',
    grupo_etario  TEXT NOT NULL DEFAULT 'nao_identificado',
    idade         INTEGER,
    -- TIPO DE VÍTIMA e CIRCUNSTÂNCIA: JSON de lista, marcação múltipla.
    tipo_vitima   TEXT NOT NULL DEFAULT '["civil"]',
    circunstancias TEXT NOT NULL DEFAULT '["nao_se_aplica"]',
    local         TEXT NOT NULL DEFAULT 'nao_identificado',
    descricao     TEXT,
    data_morte    TEXT,

    -- Conciliação entre notícias e com as correções do analista.
    chave               TEXT,
    fonte               TEXT,
    divergencia         TEXT,
    editado_manualmente INTEGER NOT NULL DEFAULT 0,
    -- versão sugerida pela LLM quando ela discorda de uma correção manual
    -- (ex.: ferida que passou a morta numa notícia nova), à espera do analista
    proposta            TEXT,
    criado_em           TEXT NOT NULL,
    atualizado_em       TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS gera_id_vitima
AFTER INSERT ON vitima
WHEN NEW.id_vitima IS NULL
BEGIN
    UPDATE vitima SET id_vitima = 'POS' || printf('%08d', NEW.id)
     WHERE id = NEW.id;
END;

CREATE INDEX IF NOT EXISTS idx_vitima_ocorrencia ON vitima(ocorrencia_id);

-- ----------------------------------------------- TABELAS COMPLEMENTARES --
-- Uma linha por vítima que é agente de segurança.
CREATE TABLE IF NOT EXISTS vitimizacao_policial (
    vitima_id     INTEGER PRIMARY KEY REFERENCES vitima(id) ON DELETE CASCADE,
    corporacao    TEXT NOT NULL DEFAULT 'nao_identificado',
    cargo         TEXT NOT NULL DEFAULT 'nao_identificado',
    status_agente TEXT NOT NULL DEFAULT 'nao_identificado',
    unidade       TEXT
);

-- Uma linha por vítima que é política.
CREATE TABLE IF NOT EXISTS politico_baleado (
    vitima_id       INTEGER PRIMARY KEY REFERENCES vitima(id) ON DELETE CASCADE,
    partido         TEXT NOT NULL DEFAULT 'Não identificado',
    cargo           TEXT,
    status_politico TEXT
);

-- Uma linha por ocorrência com o fenômeno ação policial.
CREATE TABLE IF NOT EXISTS acao_policial (
    ocorrencia_id         INTEGER PRIMARY KEY REFERENCES ocorrencia(id) ON DELETE CASCADE,
    contexto              TEXT NOT NULL DEFAULT 'acao_de_rotina',
    corporacoes_presentes TEXT NOT NULL DEFAULT '["nao_identificado"]',
    unidades_presentes    TEXT
);

-- Uma linha por ocorrência com o fenômeno crime organizado.
CREATE TABLE IF NOT EXISTS crime_organizado (
    ocorrencia_id  INTEGER PRIMARY KEY REFERENCES ocorrencia(id) ON DELETE CASCADE,
    contextos      TEXT NOT NULL DEFAULT '[]',
    grupos_citados TEXT NOT NULL DEFAULT '[]'
);

-- ---------------------------------------------------- APOIO AO PROCESSO --
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

-- ------------------------------------------------- CORPUS E AVALIAÇÃO ----
-- ------------------------------------------------------------- RECEBIDAS --
-- A fila de entrada. A notícia chega AQUI antes de existir ocorrência: só o
-- que o analista tem na mão quando cola o texto — link, a data que ele digitou
-- e o texto. Cidade e bairro não são pedidos aqui de propósito; eles só fazem
-- sentido depois que alguém leu a notícia e decidiu que ela vira um registro.
--
-- Por que esta tabela existe, e não serve a `entrada`: `entrada` é log de
-- auditoria, escrito DURANTE a classificação de uma ocorrência que já existe.
-- Esta é fila de trabalho, e é o que permite guardar a notícia que não vira
-- ocorrência nenhuma — que é justamente o exemplo negativo que ensina o
-- classificador a recusar.
CREATE TABLE IF NOT EXISTS recebida (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    link          TEXT,
    -- A data que o USUÁRIO informou para a notícia, não a de cadastro.
    data_noticia  TEXT,
    texto         TEXT NOT NULL,
    -- sha1 do texto normalizado: a mesma notícia não entra duas vezes na fila.
    impressao     TEXT,
    -- Os SEIS estados (a lista viva está em recebidas.ESTADOS):
    -- na_fila        — ninguém olhou ainda
    -- arquivo        — 0 votos do cluster; acervo, não vai à LLM e não é
    --                  pendência de ninguém
    -- alerta         — votos abaixo do corte: um voto só, vai a uma pessoa
    -- com_fenomeno   — tem indicador: pode virar ocorrência
    -- sem_fenomeno   — lida e sem indicador nenhum; fica guardada como
    --                  exemplo negativo, não vira ocorrência
    -- virou_ocorrencia
    estado        TEXT NOT NULL DEFAULT 'na_fila',
    -- O que foi marcado, por quem e como: JSON de lista.
    indicadores   TEXT NOT NULL DEFAULT '[]',
    -- 'bert' quando o modelo marcou, 'analista' quando foi pessoa.
    marcado_por   TEXT,
    modelo        TEXT,
    ocorrencia_id INTEGER REFERENCES ocorrencia(id) ON DELETE SET NULL,
    -- O exemplo que esta notícia virou no corpus, para não mandar duas vezes.
    corpus_id     INTEGER REFERENCES corpus_noticia(id) ON DELETE SET NULL,
    recebido_por  TEXT,
    recebido_em   TEXT NOT NULL,
    atualizado_em TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recebida_estado ON recebida(estado);
CREATE INDEX IF NOT EXISTS idx_recebida_impressao ON recebida(impressao);

-- Corpus de treino, separado das ocorrências de produção: aqui a notícia é um
-- exemplo rotulado à mão, não um fato registrado.
--
-- O rótulo é um só: as classes do BERT que se aplicam ao texto, em
-- multirrótulo. Não se anota motivação nem dado de vítima — isso é trabalho
-- da LLM, depois.
CREATE TABLE IF NOT EXISTS corpus_noticia (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    link          TEXT,
    texto         TEXT NOT NULL,
    -- JSON de lista. Lista vazia é um rótulo VÁLIDO e necessário: é o exemplo
    -- negativo que ensina o modelo a não marcar nada.
    indicadores   TEXT NOT NULL DEFAULT '[]',
    -- Por isso existe esta coluna: sem ela, "nenhuma classe se aplica" e
    -- "ainda não foi anotada" seriam a mesma linha no banco.
    rotulada      INTEGER NOT NULL DEFAULT 0,
    anotador      TEXT,
    -- APROVAÇÃO DO RÓTULO, que é diferente de ter rótulo. Quem cadastra a
    -- ocorrência é o analista júnior; quem revisa é o sênior, e é a revisão
    -- dele que autoriza o exemplo a treinar o modelo. Sem esta coluna, o
    -- rótulo do júnior ia direto para o treino: um erro dele não apareceria em
    -- tela nenhuma, só num modelo pior errando em notícia que ninguém olha.
    -- Nasce 0 de propósito, inclusive nas linhas que já existiam.
    aprovada      INTEGER NOT NULL DEFAULT 0,
    aprovada_por  TEXT,
    aprovada_em   TEXT,
    -- treino/validacao/teste, fixada no primeiro sorteio para que a divisão
    -- não mude entre execuções e os resultados sejam comparáveis.
    particao      TEXT,
    criado_em     TEXT NOT NULL,
    atualizado_em TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_corpus_rotulada ON corpus_noticia(rotulada);

-- ------------------------------------------------- ENTRADAS RECEBIDAS ----
-- TODO texto que entra no sistema ganha uma linha aqui, ANTES de qualquer
-- classificação e mesmo quando não existe modelo treinado.
--
-- É o que garante que nenhuma notícia submetida se perca. Sem esta tabela, dois
-- casos não deixavam rastro nenhum: o texto que não recebe indicador algum, e o
-- texto que chega antes do primeiro treino — ali `inferencia.marcar` devolve
-- None e nada era gravado, embora a notícia já tivesse sido enviada à LLM.
CREATE TABLE IF NOT EXISTS entrada (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    texto         TEXT NOT NULL,
    link          TEXT,
    -- de onde o texto veio: ocorrência de produção, playground, importação
    origem        TEXT NOT NULL DEFAULT 'producao',
    noticia_id    INTEGER REFERENCES noticia(id) ON DELETE SET NULL,
    ocorrencia_id INTEGER REFERENCES ocorrencia(id) ON DELETE SET NULL,
    -- 0 quando não havia modelo treinado no momento: a entrada existe, mas
    -- ninguém a classificou, e ela fica à espera do primeiro treino. Distingue
    -- "classificada e sem nenhum indicador" de "nunca passou pelo classificador".
    classificado  INTEGER NOT NULL DEFAULT 0,
    modelo        TEXT,
    estrategia    TEXT,
    criado_em     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entrada_classificado
    ON entrada(classificado, id DESC);

-- Uma linha por (entrada, indicador). É esta tabela que realiza "uma notícia
-- pode possuir várias classificações": a mesma entrada tem tantas linhas quantos
-- indicadores o classificador encontrou, e nenhuma quando não encontrou nenhum.
--
-- `classificacao_bert` guarda o mesmo fato como lista JSON, para auditoria da
-- chamada inteira (com a confiança de TODAS as classes, inclusive as que não
-- passaram). Aqui o fato é consultável por indicador, sem abrir o JSON.
CREATE TABLE IF NOT EXISTS entrada_classificacao (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    entrada_id   INTEGER NOT NULL REFERENCES entrada(id) ON DELETE CASCADE,
    indicador    TEXT NOT NULL,
    confianca    REAL,
    limiar       REAL,
    -- quantos modelos concordaram, quando a estratégia for de cluster
    concordancia INTEGER,
    criado_em    TEXT NOT NULL,
    UNIQUE (entrada_id, indicador)
);

CREATE INDEX IF NOT EXISTS idx_entrada_classificacao_indicador
    ON entrada_classificacao(indicador);

-- Toda marcação que o BERT faz fica registrada aqui, com o link, o texto e as
-- confianças, antes de a LLM ver qualquer coisa. É o que permite voltar depois
-- e conferir o que o modelo errou: sem este registro, o erro desaparece assim
-- que a LLM o corrige, e nunca se saberia que o classificador precisa de mais
-- exemplos daquela classe.
--
-- Revisar uma linha não altera a ocorrência — ela já foi classificada. O que a
-- revisão produz é exemplo de treino: `enviada_ao_corpus` guarda para qual
-- notícia do corpus os rótulos corrigidos foram.
CREATE TABLE IF NOT EXISTS classificacao_bert (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    noticia_id    INTEGER REFERENCES noticia(id) ON DELETE SET NULL,
    ocorrencia_id INTEGER REFERENCES ocorrencia(id) ON DELETE SET NULL,
    link          TEXT,
    texto         TEXT NOT NULL,
    -- O que o BERT marcou, e a confiança de CADA classe, não só das marcadas:
    -- ao revisar um erro, saber que a classe certa ficou em 0,48 aponta o
    -- limiar, e saber que ficou em 0,02 aponta falta de exemplo.
    classes       TEXT NOT NULL DEFAULT '[]',
    confiancas    TEXT,
    -- Um corte só, quando todos os indicadores usam o mesmo; NULL quando cada
    -- um tem o seu, porque aí nenhum número único descreveria a decisão.
    limiar        REAL,
    -- O corte de CADA indicador, como JSON. É ele que explica a marcação: uma
    -- classe com 0,42 marcada não é erro de leitura, é um indicador cujo limiar
    -- calibrado é 0,30.
    limiares      TEXT,
    modelo        TEXT,
    estrategia    TEXT,
    origem        TEXT NOT NULL DEFAULT 'producao',

    revisado           INTEGER NOT NULL DEFAULT 0,
    classes_corrigidas TEXT,
    revisor            TEXT,
    revisado_em        TEXT,
    observacao         TEXT,
    enviada_ao_corpus  INTEGER REFERENCES corpus_noticia(id) ON DELETE SET NULL,
    criado_em          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_classificacao_revisado
    ON classificacao_bert(revisado, id DESC);

-- Uma linha por (estratégia, modelo, hiperparâmetros) treinado, com as métricas
-- medidas no conjunto de teste.
--
-- `estrategia` é o que torna a tabela uma comparação e não uma lista: `unico` é
-- uma rede com uma saída sigmoide por indicador, `por_indicador` são N redes
-- binárias reunidas num vetor só. As duas são medidas na MESMA partição de
-- teste, então as linhas são diretamente comparáveis.
CREATE TABLE IF NOT EXISTS treino_execucao (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    tarefa           TEXT NOT NULL,
    estrategia       TEXT NOT NULL DEFAULT 'unico',
    modelo           TEXT NOT NULL,
    hiperparametros  TEXT NOT NULL,
    metricas         TEXT,
    matriz_confusao  TEXT,
    rotulos          TEXT,
    -- O limiar calibrado de cada indicador nesta execução. Sem ele a métrica
    -- não seria reprodutível: a mesma rede com outros cortes dá outro F1.
    limiares         TEXT,
    historico        TEXT,
    n_treino         INTEGER,
    n_validacao      INTEGER,
    n_teste          INTEGER,
    -- Quantas redes a execução ajustou: 1 em `unico`, uma por indicador em
    -- `por_indicador`. É o custo da estratégia, ao lado do resultado dela.
    n_redes          INTEGER,
    duracao_s        REAL,
    dispositivo      TEXT,
    caminho_modelo   TEXT,
    erro             TEXT,
    criado_em        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_treino_tarefa ON treino_execucao(tarefa);
"""

def agora() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def conectar() -> sqlite3.Connection:
    con = sqlite3.connect(CAMINHO_BANCO)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


# Colunas acrescentadas depois que bases já existiam. `CREATE TABLE IF NOT
# EXISTS` não mexe em tabela criada, então sem isto a base de quem já rodou a
# aplicação ficaria sem as colunas novas e a gravação falharia.
COLUNAS_NOVAS = {
    "treino_execucao": [
        ("estrategia", "TEXT NOT NULL DEFAULT 'unico'"),
        ("limiares", "TEXT"),
        ("n_redes", "INTEGER"),
    ],
    "classificacao_bert": [
        ("limiares", "TEXT"),
        ("estrategia", "TEXT"),
        ("entrada_id", "INTEGER REFERENCES entrada(id) ON DELETE SET NULL"),
    ],
    "ocorrencia": [
        ("nao_sustentados", "TEXT"),
        ("sugeridos_pela_llm", "TEXT"),
    ],
    # Impressão digital do texto, para não gravar a mesma notícia duas vezes.
    # Sem ela, alimentar o corpus a cada aprovação repetiria o mesmo exemplo a
    # cada reprocessamento — e a mesma notícia em treino e em teste é
    # vazamento, que infla a métrica sem melhorar o modelo.
    "corpus_noticia": [
        ("impressao", "TEXT"),
        # Bancos que já existiam ganham a coluna em 0: nada do que está lá foi
        # revisado por um sênior, e presumir que foi seria exatamente o
        # problema que a coluna existe para resolver. O que já estava no corpus
        # aparece em /corpus esperando revisão.
        ("aprovada", "INTEGER NOT NULL DEFAULT 0"),
        ("aprovada_por", "TEXT"),
        ("aprovada_em", "TEXT"),
    ],
    # Quantos modelos marcaram cada indicador, e de quantos. Sem isto a dúvida
    # do cluster era CALCULADA, devolvida pela inferência e jogada fora na
    # gravação: uma notícia com 2 de 3 votos em duas classes era arquivada como
    # se não tivesse nada.
    "recebida": [
        ("concordancia", "TEXT"),
        ("n_modelos", "INTEGER"),
    ],
}


def _acrescentar_colunas(con) -> None:
    for tabela, colunas in COLUNAS_NOVAS.items():
        existentes = {
            linha["name"] for linha in con.execute(f"PRAGMA table_info({tabela})")
        }
        if not existentes:
            continue
        for nome, definicao in colunas:
            if nome not in existentes:
                con.execute(f"ALTER TABLE {tabela} ADD COLUMN {nome} {definicao}")


def _preencher_impressoes(con) -> None:
    """Calcula a impressão das notícias do corpus que ainda não a têm.

    A coluna é acrescentada vazia numa base que já existia, e impressão vazia
    não casa com nada: sem este passo, a primeira aprovação regravaria no
    corpus notícias que já estavam lá.
    """
    pendentes = con.execute(
        "SELECT id, texto FROM corpus_noticia WHERE impressao IS NULL"
    ).fetchall()
    if not pendentes:
        return
    import corpus

    for linha in pendentes:
        con.execute(
            "UPDATE corpus_noticia SET impressao = ? WHERE id = ?",
            (corpus.impressao(linha["texto"]), linha["id"]),
        )


def iniciar() -> None:
    with conectar() as con:
        con.executescript(ESQUEMA)
        _acrescentar_colunas(con)
        _preencher_impressoes(con)


def json_carregar(valor, padrao):
    if not valor:
        return padrao
    try:
        return json.loads(valor)
    except (json.JSONDecodeError, TypeError):
        return padrao


def json_gravar(valor) -> str:
    return json.dumps(valor, ensure_ascii=False)
