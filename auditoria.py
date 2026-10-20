"""Banco das classificações do BERT, para conferir depois o que ele errou.

Toda marcação é registrada aqui no momento em que acontece — com o link, o
texto e a confiança de cada classe — antes de a LLM ver qualquer coisa. Sem
esse registro o erro do classificador desaparece assim que a LLM o corrige, e
nunca se saberia que aquela classe precisa de mais exemplos.

A revisão não mexe na ocorrência: ela já foi classificada e, se estiver errada,
corrige-se lá. O que a revisão produz é **exemplo de treino** — a correção vira
uma notícia rotulada no corpus, e é assim que o erro de hoje vira o acerto do
próximo treino.
"""

import indicadores
from db import agora, json_carregar, json_gravar


def registrar_entrada(con, texto: str, marcacao: dict | None, *,
                      link: str | None = None, origem: str = "producao",
                      noticia_id: int | None = None,
                      ocorrencia_id: int | None = None) -> int:
    """Grava o texto recebido e, numa linha por indicador, o que o BERT marcou.

    Chamada SEMPRE, inclusive com `marcacao is None` — e é justamente esse o caso
    que não deixava rastro antes: sem modelo treinado, o texto entrava, seguia
    para a LLM e desaparecia. Aqui ele fica com `classificado = 0`, à espera do
    primeiro treino.

    Texto sem nenhum indicador também é preservado: ele ganha a linha de entrada
    e nenhuma linha de classificação, que é a diferença entre "o classificador
    leu e não achou nada" e "ninguém leu".
    """
    momento = agora()
    cursor = con.execute(
        "INSERT INTO entrada (texto, link, origem, noticia_id, ocorrencia_id,"
        " classificado, modelo, estrategia, criado_em) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            texto, link, origem, noticia_id, ocorrencia_id,
            0 if marcacao is None else 1,
            (marcacao or {}).get("modelo"),
            (marcacao or {}).get("estrategia"),
            momento,
        ),
    )
    entrada_id = cursor.lastrowid

    if marcacao:
        confiancas = marcacao.get("confianca") or {}
        limiares = marcacao.get("limiares") or {}
        concordancias = marcacao.get("concordancia") or {}
        for indicador in marcacao.get("previsto") or []:
            con.execute(
                "INSERT OR IGNORE INTO entrada_classificacao (entrada_id,"
                " indicador, confianca, limiar, concordancia, criado_em)"
                " VALUES (?,?,?,?,?,?)",
                (
                    entrada_id, indicador,
                    confiancas.get(indicador),
                    limiares.get(indicador, marcacao.get("limiar")),
                    concordancias.get(indicador),
                    momento,
                ),
            )
    return entrada_id


def classificacoes_da_entrada(con, entrada_id: int) -> list[dict]:
    return [
        {chave: linha[chave] for chave in linha.keys()}
        for linha in con.execute(
            "SELECT * FROM entrada_classificacao WHERE entrada_id = ?"
            " ORDER BY id", (entrada_id,),
        )
    ]


def entradas(con, limite: int = 50, deslocamento: int = 0,
             sem_classificacao: bool = False) -> list[dict]:
    """As entradas recebidas, com os indicadores de cada uma reunidos.

    `sem_classificacao` filtra as que não receberam indicador nenhum — as que o
    classificador leu e descartou, mais as que chegaram antes de haver modelo.
    """
    onde = ""
    if sem_classificacao:
        onde = (
            " WHERE NOT EXISTS (SELECT 1 FROM entrada_classificacao c"
            " WHERE c.entrada_id = e.id)"
        )
    linhas = con.execute(
        f"SELECT * FROM entrada e{onde} ORDER BY e.id DESC LIMIT ? OFFSET ?",
        (limite, deslocamento),
    ).fetchall()
    return [
        {
            **{chave: linha[chave] for chave in linha.keys()},
            "classificacoes": classificacoes_da_entrada(con, linha["id"]),
        }
        for linha in linhas
    ]


def estatisticas_de_entrada(con) -> dict:
    """Quanto entrou, quanto foi classificado e quanto está à espera de treino."""
    def um(consulta: str) -> int:
        return con.execute(consulta).fetchone()[0]

    total = um("SELECT COUNT(*) FROM entrada")
    sem_modelo = um("SELECT COUNT(*) FROM entrada WHERE classificado = 0")
    sem_indicador = um(
        "SELECT COUNT(*) FROM entrada e WHERE classificado = 1 AND NOT EXISTS"
        " (SELECT 1 FROM entrada_classificacao c WHERE c.entrada_id = e.id)"
    )
    por_indicador = {
        linha["indicador"]: linha["n"]
        for linha in con.execute(
            "SELECT indicador, COUNT(*) AS n FROM entrada_classificacao"
            " GROUP BY indicador ORDER BY n DESC"
        )
    }
    return {
        "total": total,
        # Entrou antes de existir modelo: o texto está guardado, mas ninguém o
        # classificou. É a fila que o primeiro treino vai poder processar.
        "sem_modelo": sem_modelo,
        # O classificador leu e não achou nenhum indicador. É uma resposta
        # legítima, e é também o exemplo negativo de que o corpus precisa.
        "sem_indicador": sem_indicador,
        "classificadas": total - sem_modelo - sem_indicador,
        "por_indicador": por_indicador,
        "rotulos": indicadores.rotulos(),
    }


def _para_dict(linha) -> dict:
    registro = {chave: linha[chave] for chave in linha.keys()}
    registro["classes"] = json_carregar(registro.get("classes"), [])
    registro["confiancas"] = json_carregar(registro.get("confiancas"), {})
    registro["limiares"] = json_carregar(registro.get("limiares"), {})
    registro["classes_corrigidas"] = json_carregar(
        registro.get("classes_corrigidas"), None
    )
    return registro


def registrar(con, resultado: dict, texto: str, *, link: str | None = None,
              noticia_id: int | None = None, ocorrencia_id: int | None = None,
              origem: str = "producao", entrada_id: int | None = None) -> int:
    """Grava uma marcação. `resultado` é o que `inferencia.marcar` devolveu.

    Guarda as classes marcadas, a confiança de todas elas e o LIMIAR DE CADA
    UMA. O limiar por indicador é o que torna a revisão possível: sem ele, uma
    classe marcada com 0,42 parece erro de leitura, quando pode ser um
    indicador cujo corte calibrado é 0,30.
    """
    cursor = con.execute(
        "INSERT INTO classificacao_bert (noticia_id, ocorrencia_id, entrada_id,"
        " link, texto, classes, confiancas, limiar, limiares, modelo,"
        " estrategia, origem, criado_em) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            noticia_id, ocorrencia_id, entrada_id, link, texto,
            json_gravar(resultado.get("previsto") or []),
            json_gravar(resultado.get("confianca") or {}),
            resultado.get("limiar"),
            json_gravar(resultado.get("limiares") or {}),
            resultado.get("modelo"),
            resultado.get("estrategia"),
            origem,
            agora(),
        ),
    )
    return cursor.lastrowid


def _clausulas(filtros: dict) -> tuple[str, list]:
    condicoes = ["1=1"]
    parametros: list = []

    revisao = (filtros.get("revisado") or "").strip()
    if revisao == "sim":
        condicoes.append("revisado = 1")
    elif revisao == "nao":
        condicoes.append("revisado = 0")
    elif revisao == "errada":
        # Revisadas em que a correção difere do que o modelo marcou: é esta a
        # lista que diz onde o classificador está falhando.
        condicoes.append(
            "revisado = 1 AND classes_corrigidas IS NOT NULL"
            " AND classes_corrigidas <> classes"
        )

    classe = (filtros.get("classe") or "").strip()
    if classe in indicadores.todos_os_nomes():
        condicoes.append("classes LIKE ?")
        parametros.append(f'%"{classe}"%')

    if (filtros.get("sem_classe") or "").strip():
        condicoes.append("(classes IS NULL OR classes = '[]')")

    origem = (filtros.get("origem") or "").strip()
    if origem:
        condicoes.append("origem = ?")
        parametros.append(origem)

    busca = (filtros.get("busca") or "").strip()
    if busca:
        condicoes.append("(texto LIKE ? OR link LIKE ?)")
        parametros.extend([f"%{busca}%"] * 2)

    return " AND ".join(condicoes), parametros


def contar(con, filtros: dict) -> int:
    onde, parametros = _clausulas(filtros)
    return con.execute(
        f"SELECT COUNT(*) AS total FROM classificacao_bert WHERE {onde}", parametros
    ).fetchone()["total"]


def listar(con, filtros: dict, limite=None, deslocamento=0) -> list[dict]:
    onde, parametros = _clausulas(filtros)
    consulta = (
        f"SELECT * FROM classificacao_bert WHERE {onde} ORDER BY id DESC"
    )
    if limite is not None:
        consulta += " LIMIT ? OFFSET ?"
        parametros = [*parametros, limite, deslocamento]
    return [_para_dict(linha) for linha in con.execute(consulta, parametros)]


def buscar(con, registro_id: int) -> dict | None:
    linha = con.execute(
        "SELECT * FROM classificacao_bert WHERE id = ?", (registro_id,)
    ).fetchone()
    return _para_dict(linha) if linha else None


def revisar(con, registro_id: int, classes: list[str], revisor: str,
            observacao: str = "") -> dict:
    """Grava a correção do analista. Não altera a ocorrência: ela já foi
    classificada, e o que esta correção alimenta é o próximo treino."""
    validas = [n for n in indicadores.nomes() if n in set(classes or [])]
    con.execute(
        "UPDATE classificacao_bert SET revisado = 1, classes_corrigidas = ?,"
        " revisor = ?, revisado_em = ?, observacao = ? WHERE id = ?",
        (
            json_gravar(validas), revisor, agora(),
            (observacao or "").strip() or None, registro_id,
        ),
    )
    return buscar(con, registro_id)


def enviar_ao_corpus(con, registro_id: int, analista: str) -> int:
    """Leva a classificação revisada para o corpus, como exemplo rotulado.

    É aqui que o ciclo fecha: o erro visto na revisão vira treino. Usa a
    correção quando existe e a marcação do modelo quando o analista a
    confirmou sem mudar nada.

    TAMBÉM ACEITA REGISTRO NÃO REVISADO, e isso é deliberado: ele entra com
    `rotulada = ''`, ou seja, cai na fila de anotação de /corpus em vez de
    virar exemplo pronto. O que vai junto é o texto e a marcação do modelo
    como ponto de partida; quem decide continua sendo uma pessoa. Só o
    registro revisado entra já rotulado.
    """
    import corpus

    registro = buscar(con, registro_id)
    if registro is None:
        raise ValueError(f"Classificação #{registro_id} não encontrada.")
    if registro["enviada_ao_corpus"]:
        return registro["enviada_ao_corpus"]

    rotulos = registro["classes_corrigidas"]
    if rotulos is None:
        rotulos = registro["classes"]

    novo = corpus.adicionar(
        con,
        {
            "link": registro["link"],
            "texto": registro["texto"],
            "indicadores": rotulos,
            # Mesmo sem nenhuma classe, a notícia já foi decidida por uma
            # pessoa: é um exemplo negativo pronto, não uma pendência.
            "rotulada": "1" if registro["revisado"] else "",
        },
        analista,
    )
    con.execute(
        "UPDATE classificacao_bert SET enviada_ao_corpus = ? WHERE id = ?",
        (novo, registro_id),
    )
    return novo


def estatisticas(con) -> dict:
    """Quanto já foi conferido e onde o modelo erra mais."""
    total = con.execute(
        "SELECT COUNT(*) AS t FROM classificacao_bert"
    ).fetchone()["t"]
    revisadas = con.execute(
        "SELECT COUNT(*) AS t FROM classificacao_bert WHERE revisado = 1"
    ).fetchone()["t"]

    erros: dict[str, dict] = {
        nome: {"faltou": 0, "sobrou": 0} for nome in indicadores.nomes()
    }
    divergentes = 0
    for linha in con.execute(
        "SELECT classes, classes_corrigidas FROM classificacao_bert"
        " WHERE revisado = 1 AND classes_corrigidas IS NOT NULL"
    ):
        marcadas = set(json_carregar(linha["classes"], []))
        corretas = set(json_carregar(linha["classes_corrigidas"], []))
        if marcadas == corretas:
            continue
        divergentes += 1
        for nome in corretas - marcadas:
            erros.setdefault(nome, {"faltou": 0, "sobrou": 0})["faltou"] += 1
        for nome in marcadas - corretas:
            erros.setdefault(nome, {"faltou": 0, "sobrou": 0})["sobrou"] += 1

    return {
        "total": total,
        "revisadas": revisadas,
        "pendentes": total - revisadas,
        "divergentes": divergentes,
        "erros": erros,
        "rotulos": indicadores.rotulos(),
    }
