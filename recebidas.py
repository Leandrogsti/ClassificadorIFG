"""A fila de entrada: a notícia antes de virar ocorrência.

POR QUE ESTA FILA EXISTE. O fluxo antigo começava pela ocorrência — o analista
digitava data e endereço e só depois colava o texto. Isso tinha duas
consequências ruins. A primeira é que a notícia que NÃO vira ocorrência não
deixava nada atrás de si, e ela é justamente o exemplo negativo que ensina o
classificador a recusar o que está fora do escopo. A segunda é que todo texto
ia para a LLM com o catálogo inteiro de indicadores, porque não havia como
saber de antemão quais blocos interessavam.

Com a fila, a ordem se inverte: a notícia entra, o classificador olha, e só
então ela vira (ou não) um registro. Quando o BERT marca, a chamada à LLM leva
só os blocos marcados — medido: ~5.100 tokens de entrada em vez de ~6.775, que
é a folga de que o modelo precisa para raciocinar e devolver a lista de vítimas
de forma estável.

ENQUANTO NÃO HÁ MODELO TREINADO, tudo cai na fila do analista. É lento no
começo e é esse o trabalho que constrói o corpus: cada rotulagem aqui vira
exemplo de treino, inclusive — e principalmente — a que conclui que não há
fenômeno nenhum.

ESTADOS (seis; a lista viva é `ESTADOS`, logo abaixo)
    na_fila          ninguém olhou ainda
    arquivo          0 votos: acervo, não vai à LLM e não conta como pendência
    alerta           votos abaixo do corte; a dúvida vai a uma pessoa
    com_fenomeno     tem indicador; pode virar ocorrência
    sem_fenomeno     lida e sem indicador; fica guardada, não vira ocorrência
    virou_ocorrencia já é um registro de produção
"""
import json

import corpus
import indicadores
import inferencia
import preprocessar_noticia
from db import agora, json_carregar

ESTADOS = {
    "na_fila": "Na fila",
    "arquivo": "Arquivo (nenhum indicador)",
    "alerta": "Alerta (abaixo do corte)",
    "com_fenomeno": "Com fenômeno",
    "sem_fenomeno": "Sem fenômeno (conferido)",
    "virou_ocorrencia": "Virou ocorrência",
}

# A cor de cada estado na tela. As três faixas que a usuária definiu em
# 2026-10-07: vermelho revisa, laranja alerta, verde segue.
CORES = {
    "na_fila": "faixa-fila",
    "arquivo": "faixa-revisao",
    "alerta": "faixa-alerta",
    "com_fenomeno": "faixa-ok",
    "sem_fenomeno": "faixa-conferida",
    "virou_ocorrencia": "faixa-pronta",
}

# Os estados em que a notícia ainda espera uma PESSOA.
#
# `arquivo` NÃO entra, e essa é a mudança: zero indicador não é pendência, é
# acervo. A notícia fica guardada, fora da fila e fora da LLM, até que um
# modelo novo ou um indicador novo mude a resposta. Contá-la como pendente
# emperrava o trabalho com o caso mais comum que existe.
#
# `sem_fenomeno` também não entra: ali uma pessoa já olhou e decidiu.
PENDENTES = ("na_fila", "alerta")

# Os estados que podem ser reavaliados quando houver modelo novo. O arquivo é o
# principal: é para isso que ele existe.
REPROCESSAVEIS = ("arquivo", "na_fila")


def faixa(concordancia: dict[str, int] | None, n_modelos: int) -> str:
    """Em que faixa esta notícia cai, pelo indicador de MAIOR concordância.

        0 votos em tudo        -> `arquivo` (acervo; não vai à LLM)
        1 voto, abaixo do corte -> `alerta`  (laranja; pessoa verifica)
        no corte ou acima      -> `com_fenomeno`, segue para a LLM

    O corte é fração, não contagem: 2 de 3 no cluster, 1 de 1 nas estratégias de
    uma rede só. Ver `treinamento.votos_para_prosseguir`.

    A faixa é da NOTÍCIA e sai do indicador mais bem votado dela. Uma notícia
    com uma classe acima do corte segue, mesmo que as outras estejam em dúvida —
    as em dúvida viajam como alerta e o analista as vê na tela da ocorrência.
    """
    import treinamento

    votos = max((concordancia or {}).values(), default=0)
    if votos <= 0:
        return "arquivo"
    return (
        "com_fenomeno"
        if votos >= treinamento.votos_para_prosseguir(n_modelos)
        else "alerta"
    )


def _para_dict(linha) -> dict:
    registro = {chave: linha[chave] for chave in linha.keys()}
    if "indicadores" in registro:
        registro["indicadores"] = json_carregar(registro["indicadores"], [])
    if "concordancia" in registro:
        registro["concordancia"] = json_carregar(registro["concordancia"], {})
    registro["cor"] = CORES.get(registro.get("estado"), "")
    # Os indicadores que ficaram abaixo do corte: é o que a tela precisa
    # mostrar em laranja, e o que a LLM não recebe sem alguém confirmar.
    corte = registro.get("n_modelos") or 1
    import treinamento

    precisa = treinamento.votos_para_prosseguir(corte)
    registro["duvidosos"] = sorted(
        nome for nome, votos in (registro.get("concordancia") or {}).items()
        if 0 < votos < precisa
    )
    registro["votos_para_prosseguir"] = precisa
    return registro


def ja_na_fila(con, texto: str) -> int | None:
    """O id da recebida que tem este mesmo texto, se houver.

    Usa a mesma impressão do corpus: a notícia colada de duas URLs diferentes é
    a mesma notícia, e o link é a parte que muda.
    """
    linha = con.execute(
        "SELECT id FROM recebida WHERE impressao = ?", (corpus.impressao(texto),)
    ).fetchone()
    return linha["id"] if linha else None


def adicionar(con, dados: dict, analista: str) -> dict:
    """Põe a notícia na fila e deixa o classificador olhá-la, se houver modelo.

    Devolve `{"id": ..., "repetida": bool, "marcados": [...]}`. Texto repetido
    não cria linha nova: devolve a que já existe, porque receber a mesma
    notícia duas vezes é engano comum e duplicar a fila só dá trabalho a mais.
    """
    texto = (dados.get("texto") or "").strip()
    if not texto:
        raise ValueError("A notícia precisa de texto.")

    repetida = ja_na_fila(con, texto)
    if repetida is not None:
        return {"id": repetida, "repetida": True, "marcados": []}

    # O classificador olha ANTES de gravar, para que o estado já nasça na faixa
    # certa. Sem modelo treinado, `marcar` devolve None e a notícia fica
    # `na_fila`: é a pessoa que decide, e é dela que o primeiro modelo aprende.
    #
    # Lê o texto PODADO, o mesmo que a LLM vai receber depois. Antes lia o
    # texto cru, e isso era divergência silenciosa: a triagem prometia "duas
    # classes marcadas" a partir de um texto, e o portão do prompt era decidido
    # de novo a partir de outro. Ler o mesmo texto nos dois pontos faz os dois
    # concordarem por construção. A poda só remove linha sem marcador de fato
    # — ver `preprocessar_noticia`.
    marcacao = inferencia.marcar(preprocessar_noticia.resumir(texto))

    if marcacao is None:
        estado, marcado_por = "na_fila", None
        marcados, concordancia, n_modelos = [], {}, None
    else:
        marcados = sorted(marcacao.get("previsto") or [])
        concordancia = marcacao.get("concordancia") or {}
        n_modelos = int(marcacao.get("n_modelos") or 1)
        # A faixa sai da CONCORDÂNCIA, não de `previsto` estar vazio. Era esse o
        # defeito: uma notícia com 2 de 3 votos em duas classes tinha `previsto`
        # vazio e era arquivada como se não tivesse nada, enterrando justamente
        # a dúvida que precisa de pessoa.
        estado, marcado_por = faixa(concordancia, n_modelos), "bert"

    momento = agora()
    cursor = con.execute(
        "INSERT INTO recebida (link, data_noticia, texto, impressao, estado,"
        " indicadores, concordancia, n_modelos, marcado_por, modelo,"
        " recebido_por, recebido_em, atualizado_em)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            (dados.get("link") or "").strip() or None,
            (dados.get("data_noticia") or "").strip() or None,
            texto,
            corpus.impressao(texto),
            estado,
            json.dumps(marcados, ensure_ascii=False),
            json.dumps(concordancia, ensure_ascii=False),
            n_modelos,
            marcado_por,
            (marcacao or {}).get("modelo"),
            analista,
            momento,
            momento,
        ),
    )
    duvidosos = sorted(
        nome for nome, votos in concordancia.items()
        if 0 < votos < (
            __import__("treinamento").votos_para_prosseguir(n_modelos or 1)
        )
    )
    return {
        "id": cursor.lastrowid, "repetida": False, "marcados": marcados,
        "estado": estado, "duvidosos": duvidosos,
    }


def rotular(con, recebida_id: int, escolhidos: list[str], analista: str) -> dict:
    """O analista decide quais indicadores a notícia sustenta.

    É AQUI QUE O CLASSIFICADOR APRENDE. A rotulagem vai direto para o corpus,
    com `rotulada = 1` mesmo quando a lista é vazia — "li e não há fenômeno" é
    uma decisão tomada, não uma pendência, e é o exemplo que ensina o modelo a
    não marcar nada.
    """
    registro = buscar(con, recebida_id)
    if registro is None:
        raise ValueError(f"Notícia recebida #{recebida_id} não encontrada.")

    validos = [n for n in indicadores.nomes() if n in set(escolhidos or [])]

    corpus_id = registro.get("corpus_id")
    if corpus_id is None:
        corpus_id = corpus.adicionar(
            con,
            {
                "link": registro["link"],
                "texto": registro["texto"],
                "indicadores": validos,
                "rotulada": "1",
            },
            analista,
        )
    else:
        # Já estava no corpus: corrigir aqui corrige lá, senão o treino usaria
        # o rótulo antigo e a correção não teria servido para nada.
        corpus.atualizar(
            con, corpus_id,
            {
                "link": registro["link"],
                "texto": registro["texto"],
                "indicadores": validos,
                "rotulada": "1",
            },
            analista,
        )

    # Decisão de pessoa fecha a faixa: `com_fenomeno` quando marcou algo,
    # `sem_fenomeno` quando leu e concluiu que não há. Nenhum dos dois volta
    # para a fila de pendentes — ao contrário de `alerta`, que é hipótese do
    # modelo esperando confirmação, e de `arquivo`, que é acervo.
    con.execute(
        "UPDATE recebida SET indicadores = ?, estado = ?, marcado_por = ?,"
        " corpus_id = ?, atualizado_em = ? WHERE id = ?",
        (
            json.dumps(validos, ensure_ascii=False),
            "com_fenomeno" if validos else "sem_fenomeno",
            "analista",
            corpus_id,
            agora(),
            recebida_id,
        ),
    )
    return {"indicadores": validos, "corpus_id": corpus_id}


def marcar_como_ocorrencia(con, recebida_id: int, ocorrencia_id: int) -> None:
    con.execute(
        "UPDATE recebida SET estado = 'virou_ocorrencia', ocorrencia_id = ?,"
        " atualizado_em = ? WHERE id = ?",
        (ocorrencia_id, agora(), recebida_id),
    )


def buscar(con, recebida_id: int) -> dict | None:
    linha = con.execute(
        "SELECT * FROM recebida WHERE id = ?", (recebida_id,)
    ).fetchone()
    return _para_dict(linha) if linha else None


def _clausulas(filtros: dict) -> tuple[str, list]:
    condicoes, parametros = [], []
    if filtros.get("estado") in ESTADOS:
        condicoes.append("estado = ?")
        parametros.append(filtros["estado"])
    if filtros.get("busca"):
        condicoes.append("(texto LIKE ? OR link LIKE ?)")
        alvo = f"%{filtros['busca']}%"
        parametros += [alvo, alvo]
    return (" WHERE " + " AND ".join(condicoes) if condicoes else "", parametros)


def contar(con, filtros: dict | None = None) -> int:
    onde, parametros = _clausulas(filtros or {})
    return con.execute(
        f"SELECT COUNT(*) AS total FROM recebida{onde}", parametros
    ).fetchone()["total"]


def listar(con, filtros: dict | None = None, limite=None,
           deslocamento: int = 0) -> list[dict]:
    onde, parametros = _clausulas(filtros or {})
    # Ordem por PENDÊNCIA, não por data: primeiro o que espera uma pessoa —
    # `alerta` (voto isolado, a dúvida do modelo) e depois `na_fila` (ninguém
    # olhou). Em seguida o que já foi resolvido, e por último `arquivo`, que é
    # acervo: fica acessível para reavaliação, mas não disputa a atenção de
    # quem abre a tela para trabalhar a fila.
    consulta = (
        f"SELECT * FROM recebida{onde}"
        " ORDER BY CASE estado"
        "   WHEN 'alerta' THEN 0"
        "   WHEN 'na_fila' THEN 1"
        "   WHEN 'com_fenomeno' THEN 2"
        "   WHEN 'arquivo' THEN 3"
        "   ELSE 4 END,"
        " recebido_em DESC"
    )
    if limite:
        consulta += " LIMIT ? OFFSET ?"
        parametros = parametros + [limite, deslocamento]
    return [_para_dict(linha) for linha in con.execute(consulta, parametros)]


def reprocessar(con, estados: tuple[str, ...] = REPROCESSAVEIS) -> dict:
    """Passa o classificador de novo sobre o que ele não soube classificar.

    É a razão de o arquivo existir. Treinar um modelo novo, ou acrescentar um
    indicador ao catálogo, muda a resposta para notícias que já estão aqui — e
    sem este passo elas ficariam paradas com o veredito de um modelo que não
    existe mais.

    Não toca no que uma PESSOA decidiu (`sem_fenomeno`) nem no que já virou
    registro: a decisão humana vale mais que a releitura da máquina.
    """
    import treinamento

    movidas = {novo: 0 for novo in ESTADOS}
    lidas = 0
    for linha in con.execute(
        f"SELECT id, texto, estado FROM recebida"
        f" WHERE estado IN ({','.join('?' * len(estados))})",
        estados,
    ).fetchall():
        marcacao = inferencia.marcar(preprocessar_noticia.resumir(linha["texto"]))
        if marcacao is None:
            continue  # sem modelo não há o que reavaliar
        lidas += 1
        concordancia = marcacao.get("concordancia") or {}
        n_modelos = int(marcacao.get("n_modelos") or 1)
        novo = faixa(concordancia, n_modelos)
        if novo == linha["estado"]:
            continue
        con.execute(
            "UPDATE recebida SET estado = ?, indicadores = ?, concordancia = ?,"
            " n_modelos = ?, marcado_por = 'bert', modelo = ?,"
            " atualizado_em = ? WHERE id = ?",
            (
                novo,
                json.dumps(sorted(marcacao.get("previsto") or []), ensure_ascii=False),
                json.dumps(concordancia, ensure_ascii=False),
                n_modelos,
                marcacao.get("modelo"),
                agora(),
                linha["id"],
            ),
        )
        movidas[novo] += 1
    return {"lidas": lidas, "movidas": {k: v for k, v in movidas.items() if v}}


def estatisticas(con) -> dict:
    """Quantas em cada estado, e quantas esperam uma pessoa."""
    por_estado = {estado: 0 for estado in ESTADOS}
    for linha in con.execute(
        "SELECT estado, COUNT(*) AS total FROM recebida GROUP BY estado"
    ):
        if linha["estado"] in por_estado:
            por_estado[linha["estado"]] = linha["total"]
    return {
        "por_estado": por_estado,
        "total": sum(por_estado.values()),
        "pendentes": sum(por_estado[e] for e in PENDENTES),
        "arquivadas": por_estado.get("arquivo", 0),
        "reprocessaveis": sum(por_estado.get(e, 0) for e in REPROCESSAVEIS),
        "modelo_treinado": inferencia.treinado(),
    }
