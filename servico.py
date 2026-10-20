"""Operações de domínio: reprocessamento, edição de vítimas, duplicidade e
aprovação. As rotas em app.py só traduzem HTTP para estas funções.

O QUE ACONTECE EM `reprocessar`, que é o coração do arquivo: todas as notícias
da ocorrência são relidas JUNTAS e a lista de vítimas é refeita do zero. Custa
mais tokens do que processar só a notícia nova, e é o que impede a soma
indevida — se um jornal diz "um homem baleado" e outro diz "dois homens
baleados", o total é 2, nunca 3.

CORREÇÃO MANUAL PREVALECE. A vítima que o analista editou não é sobrescrita pela
releitura. Quando a LLM discorda dela — um ferido que a notícia nova diz morto —
a divergência fica guardada em `proposta`, para o analista aceitar ou recusar, em
vez de o sistema decidir sozinho qual das duas versões vale.

`enviar_ao_corpus` é o ciclo de treino fechando: ao aprovar uma ocorrência, cada
notícia dela vira exemplo rotulado com os fenômenos que o analista confirmou. Vai
na MESMA transação da aprovação, para que não exista registro aprovado sem o
exemplo correspondente.

Boa parte do cuidado aqui é com os campos de marcação múltipla. Eles ficam no
banco como JSON de lista, mas circulam pelo código como lista Python: a
conversão acontece só na borda (`_vitima_do_banco` ao ler, `json_gravar` ao
escrever), para que nenhuma comparação acabe feita entre uma lista e o texto
dela.
"""
from datetime import date, timedelta

import classificador
import indicadores
import preprocessar_noticia
import vocabulario
from db import agora, json_carregar, json_gravar

# Campos simples da vítima, editáveis pelo analista.
CAMPOS_VITIMA = [
    "nome", "status", "genero", "cor_raca", "grupo_etario", "idade",
    "local", "descricao", "data_morte", "fonte",
]

# Campos de marcação múltipla: comparados e gravados como JSON.
CAMPOS_VITIMA_LISTA = ["tipo_vitima", "circunstancias"]

TODOS_CAMPOS_VITIMA = CAMPOS_VITIMA + CAMPOS_VITIMA_LISTA

# "fonte" muda a cada passagem da LLM sem trazer informação nova sobre a
# pessoa, então não vira proposta para o analista decidir.
CAMPOS_PROPOSTA = [c for c in TODOS_CAMPOS_VITIMA if c != "fonte"]

# Tabelas complementares da vítima: a chave vinda da LLM, a tabela e as suas
# colunas, mais o tipo de vítima que faz a linha existir. Ficam juntas aqui
# para que acrescentar uma não exija caçar código.
COMPLEMENTOS_VITIMA = {
    "vitimizacao_policial": {
        "tabela": "vitimizacao_policial",
        "colunas": ["corporacao", "cargo", "status_agente", "unidade"],
        "exige_tipo": "agente",
    },
    "politico_baleado": {
        "tabela": "politico_baleado",
        "colunas": ["partido", "cargo", "status_politico"],
        "exige_tipo": "politico",
    },
}

DIAS_ACOMPANHAMENTO = 90


def _linha_para_dict(linha) -> dict:
    return {k: linha[k] for k in linha.keys()}


def _vitima_do_banco(linha) -> dict:
    vitima = _linha_para_dict(linha)
    for campo in CAMPOS_VITIMA_LISTA:
        if campo in vitima:
            vitima[campo] = json_carregar(vitima[campo], [])
    if "proposta" in vitima:
        vitima["proposta"] = json_carregar(vitima["proposta"], None)
    return vitima


def _com_complementos(con, vitima: dict) -> dict:
    """Pendura na vítima as tabelas complementares que ela tiver. Quem não é
    agente nem político volta sem nenhuma chave extra, e não com chaves
    vazias: a ausência é a informação."""
    for chave, config in COMPLEMENTOS_VITIMA.items():
        linha = con.execute(
            f"SELECT * FROM {config['tabela']} WHERE vitima_id = ?", (vitima["id"],)
        ).fetchone()
        if linha:
            vitima[chave] = {c: linha[c] for c in config["colunas"]}
    return vitima


def _gravar_complementos(con, vitima_id: int, vitima: dict) -> None:
    """Sincroniza as complementares com o tipo da vítima. Tirar `agente` do
    tipo tem de apagar a linha de vitimização policial, senão sobraria uma
    patente pendurada em civil."""
    tipos = set(vitima.get("tipo_vitima") or [])

    for chave, config in COMPLEMENTOS_VITIMA.items():
        tabela, colunas = config["tabela"], config["colunas"]
        if config["exige_tipo"] not in tipos:
            con.execute(f"DELETE FROM {tabela} WHERE vitima_id = ?", (vitima_id,))
            continue
        dados = vitima.get(chave) or {}
        # Só as colunas que REALMENTE vieram. As outras ficam de fora do
        # INSERT para o banco aplicar o DEFAULT delas — eram NOT NULL, e um
        # formulário que mandasse a tabela do agente pela metade (só o status
        # de serviço, por exemplo) derrubava a gravação com IntegrityError em
        # vez de preencher "não identificado".
        presentes = [c for c in colunas if dados.get(c) is not None]
        if not presentes:
            con.execute(
                f"INSERT INTO {tabela} (vitima_id) VALUES (?)"
                " ON CONFLICT(vitima_id) DO NOTHING",
                (vitima_id,),
            )
            continue
        marcadores = ",".join("?" for _ in presentes)
        con.execute(
            f"INSERT INTO {tabela} (vitima_id, {','.join(presentes)})"
            f" VALUES (?,{marcadores})"
            " ON CONFLICT(vitima_id) DO UPDATE SET "
            + ", ".join(f"{c} = excluded.{c}" for c in presentes),
            (vitima_id, *[dados[c] for c in presentes]),
        )


def listar_vitimas(con, ocorrencia_id: int) -> list[dict]:
    linhas = con.execute(
        "SELECT * FROM vitima WHERE ocorrencia_id = ? ORDER BY id", (ocorrencia_id,)
    ).fetchall()
    return [_com_complementos(con, _vitima_do_banco(linha)) for linha in linhas]


def adicionar_noticia(con, ocorrencia_id: int, link: str, texto: str) -> int:
    processado = preprocessar_noticia.resumir(texto)
    cursor = con.execute(
        "INSERT INTO noticia (ocorrencia_id, link, texto, texto_processado, criado_em)"
        " VALUES (?, ?, ?, ?, ?)",
        (ocorrencia_id, link or None, texto, processado, agora()),
    )
    return cursor.lastrowid


def _casar_manual(vitima: dict, disponiveis: list[dict]) -> dict | None:
    """Encontra, entre as vítimas já corrigidas pelo analista, aquela que é a
    mesma pessoa. Casamento por nome tem prioridade sobre o de perfil, e cada
    vítima só pode ser casada uma vez — é isso que impede a mesma pessoa de
    aparecer duas vezes quando uma notícia nova traz um dado a mais."""
    nome = classificador.nome_identificador(vitima)
    if nome:
        for candidato in disponiveis:
            if classificador.nome_identificador(candidato) == nome:
                return candidato
    for candidato in disponiveis:
        if classificador.mesma_pessoa(vitima, candidato):
            return candidato
    return None


# Valores que significam "não sei", e por isso nunca contradizem o que o
# analista preencheu à mão.
VAZIOS = {None, "", "nao_identificado"}


def _proposta_de(vitima: dict, existente: dict) -> dict:
    """Só vira proposta o que a LLM afirma e diverge da correção do analista.
    Campo que a LLM deixou vazio é ausência de informação, não contradição."""
    proposta = {}
    for campo in CAMPOS_PROPOSTA:
        novo = vitima.get(campo)
        if campo in CAMPOS_VITIMA_LISTA:
            # Lista só com o valor-padrão também é ausência de informação.
            concretos = [
                v for v in (novo or [])
                if v not in VAZIOS and v != vocabulario.CIRCUNSTANCIA_PADRAO
            ]
            if not concretos:
                continue
            # As derivadas ficam FORA da comparação. `chacina` é calculada da
            # contagem de vítimas e a LLM nunca a devolve, então compará-la
            # faria toda vítima que já a tem gerar uma proposta de removê-la —
            # divergência inventada pela comparação, não afirmada pela LLM.
            def _comparaveis(valores):
                return sorted(
                    v for v in (valores or [])
                    if v not in vocabulario.CIRCUNSTANCIAS_DERIVADAS
                )

            if _comparaveis(novo) != _comparaveis(existente.get(campo)):
                proposta[campo] = novo
            continue
        if novo in VAZIOS:
            continue
        if novo != existente.get(campo):
            proposta[campo] = novo
    return proposta


COLUNAS_INSERCAO_VITIMA = [
    "ocorrencia_id", "chave", "nome", "status", "genero", "cor_raca",
    "grupo_etario", "idade", "tipo_vitima", "circunstancias", "local",
    "descricao", "data_morte", "fonte", "divergencia", "criado_em",
    "atualizado_em",
]


def _inserir_vitima(con, ocorrencia_id: int, vitima: dict, momento: str,
                    manual: bool = False) -> int:
    valores = {
        "ocorrencia_id": ocorrencia_id,
        "chave": vitima.get("chave") or classificador.chave_vitima(vitima),
        "nome": vitima.get("nome"),
        "status": vitima.get("status") or "ferida",
        "genero": vitima.get("genero") or "nao_identificado",
        "cor_raca": vitima.get("cor_raca") or "nao_identificado",
        "grupo_etario": vitima.get("grupo_etario") or "nao_identificado",
        "idade": vitima.get("idade"),
        "tipo_vitima": json_gravar(vitima.get("tipo_vitima") or ["civil"]),
        "circunstancias": json_gravar(
            vitima.get("circunstancias") or [vocabulario.CIRCUNSTANCIA_PADRAO]
        ),
        "local": vitima.get("local") or "nao_identificado",
        "descricao": vitima.get("descricao"),
        "data_morte": vitima.get("data_morte"),
        "fonte": vitima.get("fonte"),
        "divergencia": vitima.get("divergencia"),
        "criado_em": momento,
        "atualizado_em": momento,
    }
    marcadores = ",".join("?" for _ in COLUNAS_INSERCAO_VITIMA)
    cursor = con.execute(
        f"INSERT INTO vitima ({','.join(COLUNAS_INSERCAO_VITIMA)},"
        f" editado_manualmente) VALUES ({marcadores}, ?)",
        (*[valores[c] for c in COLUNAS_INSERCAO_VITIMA], 1 if manual else 0),
    )
    _gravar_complementos(con, cursor.lastrowid, vitima)
    return cursor.lastrowid


def reprocessar(con, ocorrencia_id: int) -> dict:
    """Relê todas as notícias juntas e refaz a lista de vítimas.

    Correções manuais do analista prevalecem. Quando a LLM discorda de uma
    vítima já corrigida (um ferido que morreu, por exemplo), a divergência fica
    guardada como proposta para o analista aceitar, em vez de sobrescrever.
    """
    noticias = [
        _linha_para_dict(linha)
        for linha in con.execute(
            "SELECT * FROM noticia WHERE ocorrencia_id = ? ORDER BY id",
            (ocorrencia_id,),
        ).fetchall()
    ]
    resultado = classificador.classificar_ocorrencia(noticias, con, ocorrencia_id)

    manuais = [
        _com_complementos(con, _vitima_do_banco(linha))
        for linha in con.execute(
            "SELECT * FROM vitima WHERE ocorrencia_id = ? AND editado_manualmente = 1",
            (ocorrencia_id,),
        ).fetchall()
    ]

    con.execute(
        "DELETE FROM vitima WHERE ocorrencia_id = ? AND editado_manualmente = 0",
        (ocorrencia_id,),
    )

    momento = agora()
    disponiveis = list(manuais)
    for vitima in resultado["vitimas"]:
        existente = _casar_manual(vitima, disponiveis)
        if existente:
            disponiveis.remove(existente)
            divergentes = _proposta_de(vitima, existente)
            con.execute(
                "UPDATE vitima SET proposta = ?, atualizado_em = ? WHERE id = ?",
                (json_gravar(divergentes) if divergentes else None, momento,
                 existente["id"]),
            )
            continue
        _inserir_vitima(con, ocorrencia_id, vitima, momento)

    con.execute(
        "UPDATE ocorrencia SET"
        " fenomenos_bert = ?, nao_sustentados = ?, sugeridos_pela_llm = ?,"
        " divergencias = ?, vitimas_baleadas = ?,"
        " atualizado_em = ? WHERE id = ?",
        (
            json_gravar(resultado.get("fenomenos_bert")),
            # Por que uma tabela complementar ficou em branco, e o que a LLM
            # tentou marcar sem autorização: as duas coisas que o analista
            # precisa saber para revisar o classificador em vez de adivinhar.
            json_gravar(resultado.get("nao_sustentados") or []),
            json_gravar(resultado.get("sugeridos_pela_llm") or []),
            json_gravar(resultado["divergencias"]),
            resultado["vitimas_baleadas"],
            momento,
            ocorrencia_id,
        ),
    )

    _gravar_complementos_ocorrencia(con, ocorrencia_id, resultado)
    recalcular_fenomenos(con, ocorrencia_id, resultado["fenomenos"])
    return resultado


def _gravar_complementos_ocorrencia(con, ocorrencia_id: int, resultado: dict) -> None:
    """As complementares da ocorrência só existem enquanto o fenômeno existir.
    Quando a LLM descarta `acao_policial`, a linha some junto — deixá-la
    manteria contexto e corporações de um fenômeno que não foi observado."""
    acao = resultado.get("acao_policial")
    if acao:
        con.execute(
            "INSERT INTO acao_policial (ocorrencia_id, contexto,"
            " corporacoes_presentes, unidades_presentes) VALUES (?,?,?,?)"
            " ON CONFLICT(ocorrencia_id) DO UPDATE SET contexto = excluded.contexto,"
            " corporacoes_presentes = excluded.corporacoes_presentes,"
            " unidades_presentes = excluded.unidades_presentes",
            (ocorrencia_id, acao["contexto"],
             json_gravar(acao["corporacoes_presentes"]), acao["unidades_presentes"]),
        )
    else:
        con.execute(
            "DELETE FROM acao_policial WHERE ocorrencia_id = ?", (ocorrencia_id,)
        )

    crime = resultado.get("crime_organizado")
    if crime:
        con.execute(
            "INSERT INTO crime_organizado (ocorrencia_id, contextos, grupos_citados)"
            " VALUES (?,?,?) ON CONFLICT(ocorrencia_id) DO UPDATE SET"
            " contextos = excluded.contextos,"
            " grupos_citados = excluded.grupos_citados",
            (ocorrencia_id, json_gravar(crime["contextos"]),
             json_gravar(crime["grupos_citados"])),
        )
    else:
        con.execute(
            "DELETE FROM crime_organizado WHERE ocorrencia_id = ?", (ocorrencia_id,)
        )


def complementos_da_ocorrencia(con, ocorrencia_id: int) -> dict:
    """As duas tabelas complementares da ocorrência, para exibição."""
    resultado = {}

    linha = con.execute(
        "SELECT * FROM acao_policial WHERE ocorrencia_id = ?", (ocorrencia_id,)
    ).fetchone()
    if linha:
        resultado["acao_policial"] = {
            "contexto": linha["contexto"],
            "corporacoes_presentes": json_carregar(linha["corporacoes_presentes"], []),
            "unidades_presentes": linha["unidades_presentes"],
        }

    linha = con.execute(
        "SELECT * FROM crime_organizado WHERE ocorrencia_id = ?", (ocorrencia_id,)
    ).fetchone()
    if linha:
        resultado["crime_organizado"] = {
            "contextos": json_carregar(linha["contextos"], []),
            "grupos_citados": json_carregar(linha["grupos_citados"], []),
        }

    return resultado


def limpar_classificacao(con, ocorrencia_id: int) -> None:
    """Sem notícias não há o que classificar; deixar a lista antiga no lugar
    faria o registro mostrar vítimas sem nenhuma fonte que as sustente."""
    con.execute("DELETE FROM vitima WHERE ocorrencia_id = ?", (ocorrencia_id,))
    con.execute("DELETE FROM acao_policial WHERE ocorrencia_id = ?", (ocorrencia_id,))
    con.execute("DELETE FROM crime_organizado WHERE ocorrencia_id = ?", (ocorrencia_id,))
    con.execute(
        "UPDATE ocorrencia SET"
        " fenomenos = '[]', fenomenos_bert = NULL, nao_sustentados = NULL,"
        " sugeridos_pela_llm = NULL, divergencias = NULL,"
        " vitimas_baleadas = 'N', status = 'rascunho', atualizado_em = ?"
        " WHERE id = ?",
        (agora(), ocorrencia_id),
    )


def noticia_em_outra_ocorrencia(con, texto: str,
                                ocorrencia_id: int) -> list[dict]:
    """Onde mais este mesmo texto já está anexado.

    Existe porque aconteceu: a reportagem sobre o sargento executado em Vaz
    Lobo foi colada tanto na ocorrência dela quanto na do tiroteio da Ilha do
    Governador, que é outro fato. Os dois links só diferiam por um
    `#google_vignette` no fim. Ninguém percebeu, e no reprocessamento seguinte
    as vítimas de um fato entrariam na lista do outro.

    Compara pela impressão do texto, não pelo link, justamente porque o link é
    a parte que muda.
    """
    import corpus

    marca = corpus.impressao(texto)
    encontradas = []
    for linha in con.execute(
        "SELECT n.id, n.ocorrencia_id, n.texto, o.id_caso"
        " FROM noticia n JOIN ocorrencia o ON o.id = n.ocorrencia_id"
        " WHERE n.ocorrencia_id != ?",
        (ocorrencia_id,),
    ):
        if corpus.impressao(linha["texto"]) == marca:
            encontradas.append({
                "noticia_id": linha["id"],
                "ocorrencia_id": linha["ocorrencia_id"],
                "id_caso": linha["id_caso"],
            })
    return encontradas


def enviar_ao_corpus(con, ocorrencia_id: int, analista: str,
                     aprovar: bool = True) -> dict:
    """Cada notícia da ocorrência aprovada vira um exemplo de treino.

    É aqui que o ciclo fecha. Antes, aprovar só mudava o status: o trabalho de
    conferir a classificação não deixava nada atrás de si, e o corpus só crescia
    se alguém o alimentasse à mão. Agora a conferência que o analista já fez é o
    rótulo.

    O rótulo de cada notícia é o conjunto de fenômenos CONFIRMADO da ocorrência,
    reduzido às classes que o BERT aprende (`indicadores.nomes()`). Fica de fora
    o que só existe como regra sobre a lista de vítimas — hoje `chacina` —
    porque não se aprende do texto, e sim da contagem. `vitimizacao_policial`
    ENTRA: desde 09/10 ela é classe de BERT, com a regra ao lado.

    É A ÚNICA PORTA QUE JÁ ENTRA APROVADA, e é por isso que ela existe: quem
    aprova a ocorrência é o sênior que revisou o trabalho do júnior, então a
    decisão dele é a assinatura que o treino exige. `aprovar=False` grava o
    exemplo sem a assinatura — serve à notícia que está certa como registro mas
    é ruim como exemplo de treino (texto truncado, duas ocorrências misturadas,
    fonte duvidosa). Ela fica em /corpus esperando revisão, em vez de treinar o
    modelo ou de se perder.

    LIMITE CONHECIDO: o rótulo é da OCORRÊNCIA, e cada notícia recebe o conjunto
    inteiro. Numa ocorrência com várias fontes, uma delas pode não sustentar
    sozinha tudo o que o conjunto sustenta — uma notícia curta que não menciona
    a polícia receberia `acao_policial` assim mesmo. Por isso a notícia entra
    editável em /corpus, e não como verdade intocável.
    """
    import corpus

    fenomenos = json_carregar(
        con.execute(
            "SELECT fenomenos FROM ocorrencia WHERE id = ?", (ocorrencia_id,)
        ).fetchone()["fenomenos"],
        [],
    )
    do_bert = [nome for nome in indicadores.nomes() if nome in set(fenomenos)]

    noticias = con.execute(
        "SELECT id, link, texto FROM noticia WHERE ocorrencia_id = ? ORDER BY id",
        (ocorrencia_id,),
    ).fetchall()

    novas, repetidas = [], []
    for noticia in noticias:
        if corpus.ja_no_corpus(con, noticia["texto"]) is not None:
            repetidas.append(noticia["id"])
            continue
        novas.append(corpus.adicionar(
            con,
            {
                "link": noticia["link"],
                "texto": noticia["texto"],
                "indicadores": do_bert,
                # Lista vazia é rótulo válido: é o exemplo negativo. Marcar
                # `rotulada` explícito é o que impede que a ocorrência sem
                # nenhum fenômeno volte para a fila de anotação como pendência.
                "rotulada": "1",
                "aprovada": "1" if aprovar else "0",
            },
            analista,
        ))
    return {
        "novas": novas, "repetidas": repetidas, "rotulos": do_bert,
        "aprovadas": bool(aprovar),
    }


def _marcar_chacina(con, vitimas: list[dict]) -> int:
    """Põe e tira a circunstância `chacina` das vítimas, pela contagem.

    Roda ANTES do cálculo dos fenômenos, para que o fenômeno e a circunstância
    saiam da mesma leitura da lista. Devolve quantas vítimas mudaram.

    Tira também, e isso é metade do trabalho: quando um morto se revela ferido,
    ou quando o status de serviço do agente é corrigido, a ocorrência deixa de
    ser chacina e a marcação tem de sair de todas as vítimas. Marcação derivada
    que só sabe entrar vira sujeira permanente no banco.

    Não respeita `editado_manualmente`, de propósito: `chacina` é campo
    calculado, como o fenômeno da ocorrência. Quem discorda da marcação corrige
    o que a gera — o status da vítima ou o status de serviço do agente.
    """
    do_grupo = set(indicadores.vitimas_de_chacina(vitimas))
    mudaram = 0
    for posicao, vitima in enumerate(vitimas):
        atuais = list(vitima.get("circunstancias") or [])
        tinha = "chacina" in atuais
        deve = posicao in do_grupo
        if tinha == deve:
            continue

        if deve:
            # `nao_se_aplica` é a AUSÊNCIA de circunstância: não convive com
            # outra, então sai quando entra uma concreta.
            novas = [c for c in atuais if c != vocabulario.CIRCUNSTANCIA_PADRAO]
            novas.append("chacina")
        else:
            novas = [c for c in atuais if c != "chacina"]
            novas = novas or [vocabulario.CIRCUNSTANCIA_PADRAO]

        con.execute(
            "UPDATE vitima SET circunstancias = ?, atualizado_em = ?"
            " WHERE id = ?",
            (json_gravar(sorted(set(novas))), agora(), vitima["id"]),
        )
        vitima["circunstancias"] = sorted(set(novas))
        mudaram += 1
    return mudaram


def recalcular_fenomenos(con, ocorrencia_id: int,
                         do_texto: list[str] | None = None) -> list[str]:
    """Consolida FENÔMENO OBSERVADO: o que a LLM certificou mais o que a lista
    de vítimas comprova. Roda de novo a cada alteração numa vítima, porque um
    ferido que morre pode transformar a ocorrência em chacina."""
    if do_texto is None:
        linha = con.execute(
            "SELECT fenomenos FROM ocorrencia WHERE id = ?", (ocorrencia_id,)
        ).fetchone()
        do_texto = json_carregar(linha["fenomenos"], []) if linha else []
    vitimas = listar_vitimas(con, ocorrencia_id)
    _marcar_chacina(con, vitimas)
    calculados = indicadores.calcular(vitimas, do_texto)
    con.execute(
        "UPDATE ocorrencia SET fenomenos = ?, atualizado_em = ? WHERE id = ?",
        (json_gravar(calculados), agora(), ocorrencia_id),
    )
    return calculados


def _normalizar_valor(campo: str, valor):
    """Encaixa o que veio do formulário na lista fechada do campo. Um select
    adulterado no navegador não pode gravar valor fora da lista."""
    if campo == "idade":
        texto = str(valor or "").strip()
        return int(texto) if texto.isdigit() and 0 <= int(texto) <= 200 else None
    if campo == "status":
        return vocabulario.normalizar(valor, vocabulario.STATUS_VITIMA, "ferida")
    if campo == "genero":
        return vocabulario.normalizar(valor, vocabulario.GENEROS, "nao_identificado")
    if campo == "cor_raca":
        return vocabulario.normalizar(valor, vocabulario.CORES_RACA, "nao_identificado")
    if campo == "grupo_etario":
        return vocabulario.normalizar(
            valor, vocabulario.GRUPOS_ETARIOS, "nao_identificado"
        )
    if campo == "local":
        return vocabulario.normalizar(
            valor, vocabulario.LOCAIS_VITIMA, "nao_identificado"
        )
    if campo == "tipo_vitima":
        return vocabulario.normalizar_lista(valor, vocabulario.TIPOS_VITIMA, "civil")
    if campo == "circunstancias":
        marcadas = vocabulario.normalizar_lista(valor, vocabulario.CIRCUNSTANCIAS)
        concretas = [c for c in marcadas if c != vocabulario.CIRCUNSTANCIA_PADRAO]
        return concretas or [vocabulario.CIRCUNSTANCIA_PADRAO]
    if valor is None:
        return None
    return str(valor).strip() or None


def _complemento_do_formulario(chave: str, dados: dict) -> dict | None:
    """Lê os campos de uma tabela complementar vindos do formulário. Devolve
    None quando nenhum veio, para não apagar o que a LLM preencheu numa edição
    que nem mexeu nesses campos."""
    colunas = COMPLEMENTOS_VITIMA[chave]["colunas"]
    presentes = {c: dados[f"{chave}_{c}"] for c in colunas if f"{chave}_{c}" in dados}
    return presentes or None


def atualizar_vitima(con, vitima_id: int, dados: dict, analista: str, fonte: str) -> int:
    """Edição manual: grava o valor anterior de cada campo alterado."""
    atual = _vitima_do_banco(
        con.execute("SELECT * FROM vitima WHERE id = ?", (vitima_id,)).fetchone()
    )
    momento = agora()
    mudou = False

    for campo in TODOS_CAMPOS_VITIMA:
        if campo not in dados:
            continue
        novo = _normalizar_valor(campo, dados[campo])
        anterior = atual.get(campo)
        if campo == "circunstancias":
            # As DERIVADAS voltam para a lista. O formulário não as oferece
            # como caixa, então um POST normal vinha sem elas e as apagava — e
            # `recalcular_fenomenos` as repunha logo depois, deixando no
            # histórico uma alteração que ninguém fez. Preservá-las aqui é o
            # que mantém o histórico honesto.
            preservadas = [
                c for c in (anterior or [])
                if c in vocabulario.CIRCUNSTANCIAS_DERIVADAS
            ]
            if preservadas:
                concretas = [
                    c for c in [*novo, *preservadas]
                    if c != vocabulario.CIRCUNSTANCIA_PADRAO
                ]
                novo = sorted(set(concretas)) or [vocabulario.CIRCUNSTANCIA_PADRAO]
        if campo in CAMPOS_VITIMA_LISTA:
            if sorted(novo) == sorted(anterior or []):
                continue
            gravado = json_gravar(novo)
        else:
            if novo == anterior:
                continue
            gravado = novo

        mudou = True
        con.execute(
            "INSERT INTO vitima_historico (vitima_id, ocorrencia_id, campo,"
            " valor_anterior, valor_novo, alterado_por, alterado_em, fonte)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (
                vitima_id, atual["ocorrencia_id"], campo,
                None if anterior is None else str(anterior),
                None if novo is None else str(novo),
                analista, momento, fonte or None,
            ),
        )
        con.execute(f"UPDATE vitima SET {campo} = ? WHERE id = ?", (gravado, vitima_id))

    # A idade manda no grupo etário, mesmo quando o formulário também mandou o
    # grupo: deixar os dois se contradizerem faria a mesma vítima cair num
    # grupo na tabela e em outro no gráfico. Sem idade, vale o que foi enviado.
    if "idade" in dados:
        idade = _normalizar_valor("idade", dados["idade"])
        if idade is not None:
            derivado = vocabulario.grupo_etario(idade)
            atualizado = con.execute(
                "SELECT grupo_etario FROM vitima WHERE id = ?", (vitima_id,)
            ).fetchone()["grupo_etario"]
            if derivado != atualizado:
                mudou = True
                con.execute(
                    "UPDATE vitima SET grupo_etario = ? WHERE id = ?",
                    (derivado, vitima_id),
                )

    complementos = {
        chave: _complemento_do_formulario(chave, dados) or dados.get(chave)
        for chave in COMPLEMENTOS_VITIMA
    }
    if any(complementos.values()):
        mudou = True

    if mudou:
        atualizada = _com_complementos(
            con,
            _vitima_do_banco(
                con.execute("SELECT * FROM vitima WHERE id = ?", (vitima_id,)).fetchone()
            ),
        )
        for chave, enviado in complementos.items():
            if enviado:
                atualizada[chave] = {**(atualizada.get(chave) or {}), **enviado}
        _gravar_complementos(con, vitima_id, atualizada)

        con.execute(
            "UPDATE vitima SET chave = ?, editado_manualmente = 1, proposta = NULL,"
            " atualizado_em = ? WHERE id = ?",
            (classificador.chave_vitima(atualizada), momento, vitima_id),
        )
        recalcular_fenomenos(con, atual["ocorrencia_id"])
        marcar_para_reaprovacao(con, atual["ocorrencia_id"])

    return atual["ocorrencia_id"]


def aceitar_proposta(con, vitima_id: int, analista: str) -> int:
    linha = con.execute("SELECT * FROM vitima WHERE id = ?", (vitima_id,)).fetchone()
    proposta = json_carregar(linha["proposta"], None)
    if not proposta:
        return linha["ocorrencia_id"]
    return atualizar_vitima(
        con, vitima_id, proposta, analista, "Proposta da LLM sobre notícia adicionada"
    )


def descartar_proposta(con, vitima_id: int) -> int:
    linha = con.execute(
        "SELECT ocorrencia_id FROM vitima WHERE id = ?", (vitima_id,)
    ).fetchone()
    con.execute(
        "UPDATE vitima SET proposta = NULL, atualizado_em = ? WHERE id = ?",
        (agora(), vitima_id),
    )
    return linha["ocorrencia_id"]


def criar_vitima_manual(con, ocorrencia_id: int, dados: dict, analista: str) -> int:
    momento = agora()
    vitima = {
        campo: _normalizar_valor(campo, dados.get(campo))
        for campo in TODOS_CAMPOS_VITIMA
    }
    if vitima.get("idade") is not None:
        vitima["grupo_etario"] = vocabulario.grupo_etario(vitima["idade"])
    for chave in COMPLEMENTOS_VITIMA:
        enviado = _complemento_do_formulario(chave, dados) or dados.get(chave)
        if enviado:
            vitima[chave] = enviado

    vitima_id = _inserir_vitima(con, ocorrencia_id, vitima, momento, manual=True)
    con.execute(
        "INSERT INTO vitima_historico (vitima_id, ocorrencia_id, campo, valor_anterior,"
        " valor_novo, alterado_por, alterado_em, fonte) VALUES (?,?,?,?,?,?,?,?)",
        (vitima_id, ocorrencia_id, "vitima", None, "incluída manualmente",
         analista, momento, vitima.get("fonte")),
    )
    recalcular_fenomenos(con, ocorrencia_id)
    marcar_para_reaprovacao(con, ocorrencia_id)
    return vitima_id


def remover_vitima(con, vitima_id: int, analista: str) -> int:
    linha = con.execute("SELECT * FROM vitima WHERE id = ?", (vitima_id,)).fetchone()
    ocorrencia_id = linha["ocorrencia_id"]
    con.execute(
        "INSERT INTO vitima_historico (vitima_id, ocorrencia_id, campo, valor_anterior,"
        " valor_novo, alterado_por, alterado_em, fonte) VALUES (?,?,?,?,?,?,?,?)",
        (vitima_id, ocorrencia_id, "vitima", linha["nome"] or "sem nome", "removida",
         analista, agora(), None),
    )
    con.execute("DELETE FROM vitima WHERE id = ?", (vitima_id,))
    recalcular_fenomenos(con, ocorrencia_id)
    marcar_para_reaprovacao(con, ocorrencia_id)
    return ocorrencia_id


def marcar_para_reaprovacao(con, ocorrencia_id: int) -> None:
    """Qualquer alteração derruba a aprovação; a versão alterada só chega à
    comunicação depois de passar de novo pelo responsável."""
    con.execute(
        "UPDATE ocorrencia SET status = 'nao_aprovado', aprovado_por = NULL,"
        " aprovado_em = NULL, atualizado_em = ? WHERE id = ? AND status = 'aprovado'",
        (agora(), ocorrencia_id),
    )


def checar_duplicidade(con, ocorrencia_id: int) -> list[int]:
    """Roda depois de gravar, sem travar o cadastro: mesmo dia e bairro da
    mesma cidade. Usa o índice (data_caso, cidade, bairro)."""
    atual = con.execute(
        "SELECT * FROM ocorrencia WHERE id = ?", (ocorrencia_id,)
    ).fetchone()
    similares = con.execute(
        "SELECT id FROM ocorrencia WHERE data_caso = ? AND cidade = ? AND bairro = ?"
        " AND id != ? AND unificada_em IS NULL AND status != 'rascunho'",
        (atual["data_caso"], atual["cidade"], atual["bairro"], ocorrencia_id),
    ).fetchall()

    criados = []
    for similar in similares:
        ja_existe = con.execute(
            "SELECT 1 FROM alerta_duplicidade WHERE status = 'aberto' AND"
            " ((ocorrencia_id = ? AND ocorrencia_similar_id = ?)"
            " OR (ocorrencia_id = ? AND ocorrencia_similar_id = ?))",
            (ocorrencia_id, similar["id"], similar["id"], ocorrencia_id),
        ).fetchone()
        if ja_existe:
            continue
        con.execute(
            "INSERT INTO alerta_duplicidade (ocorrencia_id, ocorrencia_similar_id,"
            " criado_em) VALUES (?, ?, ?)",
            (ocorrencia_id, similar["id"], agora()),
        )
        criados.append(similar["id"])
    return criados


def unificar(con, alerta_id: int, analista: str, motivo: str) -> int:
    """Junta os registros movendo as notícias para a ocorrência mais antiga e
    reprocessando. Nada é apagado: a origem fica marcada em unificada_em e cada
    notícia guarda de onde veio, o que permite desfazer."""
    alerta = con.execute(
        "SELECT * FROM alerta_duplicidade WHERE id = ?", (alerta_id,)
    ).fetchone()
    destino, origem = sorted([alerta["ocorrencia_id"], alerta["ocorrencia_similar_id"]])

    con.execute(
        "UPDATE noticia SET ocorrencia_id = ?, ocorrencia_origem_id = ?"
        " WHERE ocorrencia_id = ?",
        (destino, origem, origem),
    )
    con.execute(
        "UPDATE ocorrencia SET unificada_em = ?, status = 'unificada', atualizado_em = ?"
        " WHERE id = ?",
        (destino, agora(), origem),
    )
    con.execute(
        "UPDATE alerta_duplicidade SET status = 'unificado', resolvido_por = ?,"
        " resolvido_em = ?, motivo = ? WHERE id = ?",
        (analista, agora(), motivo or None, alerta_id),
    )

    reprocessar(con, destino)
    con.execute(
        "UPDATE ocorrencia SET status = 'nao_aprovado', aprovado_por = NULL,"
        " aprovado_em = NULL, atualizado_em = ? WHERE id = ?",
        (agora(), destino),
    )
    return destino


def desfazer_unificacao(con, alerta_id: int, analista: str) -> int:
    alerta = con.execute(
        "SELECT * FROM alerta_duplicidade WHERE id = ?", (alerta_id,)
    ).fetchone()
    destino, origem = sorted([alerta["ocorrencia_id"], alerta["ocorrencia_similar_id"]])

    con.execute(
        "UPDATE noticia SET ocorrencia_id = ocorrencia_origem_id,"
        " ocorrencia_origem_id = NULL WHERE ocorrencia_origem_id = ?",
        (origem,),
    )
    con.execute(
        "UPDATE ocorrencia SET unificada_em = NULL, status = 'nao_aprovado',"
        " atualizado_em = ? WHERE id = ?",
        (agora(), origem),
    )
    con.execute(
        "UPDATE alerta_duplicidade SET status = 'aberto', resolvido_por = ?,"
        " resolvido_em = NULL, motivo = NULL WHERE id = ?",
        (analista, alerta_id),
    )
    for ocorrencia_id in (destino, origem):
        if con.execute(
            "SELECT 1 FROM noticia WHERE ocorrencia_id = ?", (ocorrencia_id,)
        ).fetchone():
            reprocessar(con, ocorrencia_id)
    return destino


def descartar_alerta(con, alerta_id: int, analista: str, motivo: str) -> None:
    con.execute(
        "UPDATE alerta_duplicidade SET status = 'descartado', resolvido_por = ?,"
        " resolvido_em = ?, motivo = ? WHERE id = ?",
        (analista, agora(), motivo or None, alerta_id),
    )


# A vítima só faz sentido junto do lugar onde o fato aconteceu, então toda
# listagem e toda planilha de vítimas trazem a localização da ocorrência.
CONSULTA_VITIMAS = (
    "SELECT v.*, o.id_caso, o.data_caso, o.hora_caso, o.estado, o.cidade,"
    " o.bairro, o.localidade, o.status AS status_ocorrencia FROM vitima v"
    " JOIN ocorrencia o ON o.id = v.ocorrencia_id"
)


def _clausulas_vitimas(filtros: dict) -> tuple[str, list]:
    condicoes = ["o.unificada_em IS NULL"]
    parametros: list = []

    for chave, condicao in (
        ("de", "o.data_caso >= ?"),
        ("ate", "o.data_caso <= ?"),
        ("estado", "o.estado = ?"),
        ("cidade", "o.cidade = ?"),
        ("bairro", "o.bairro = ?"),
        ("status", "v.status = ?"),
        ("genero", "v.genero = ?"),
        ("cor_raca", "v.cor_raca = ?"),
        ("grupo_etario", "v.grupo_etario = ?"),
    ):
        valor = (filtros.get(chave) or "").strip()
        if valor:
            condicoes.append(condicao)
            parametros.append(valor)

    # Campos de marcação múltipla são JSON: casar com as aspas evita que
    # `agente` traga também um futuro `agente_socioeducativo`.
    for chave, coluna in (
        ("circunstancia", "v.circunstancias"),
        ("tipo_vitima", "v.tipo_vitima"),
    ):
        valor = (filtros.get(chave) or "").strip()
        if valor:
            condicoes.append(f"{coluna} LIKE ?")
            parametros.append(f'%"{valor}"%')

    # Localidade é texto livre digitado pelo analista, então a busca é parcial.
    localidade = (filtros.get("localidade") or "").strip()
    if localidade:
        condicoes.append("o.localidade LIKE ?")
        parametros.append(f"%{localidade}%")

    return " AND ".join(condicoes), parametros


def contar_vitimas(con, filtros: dict) -> int:
    onde, parametros = _clausulas_vitimas(filtros)
    return con.execute(
        "SELECT COUNT(*) AS total FROM vitima v"
        f" JOIN ocorrencia o ON o.id = v.ocorrencia_id WHERE {onde}",
        parametros,
    ).fetchone()["total"]


def filtrar_vitimas(con, filtros: dict, limite=None, deslocamento=0) -> list[dict]:
    onde, parametros = _clausulas_vitimas(filtros)
    consulta = f"{CONSULTA_VITIMAS} WHERE {onde} ORDER BY o.data_caso DESC, v.id"
    if limite is not None:
        consulta += " LIMIT ? OFFSET ?"
        parametros = [*parametros, limite, deslocamento]
    return [
        _com_complementos(con, _vitima_do_banco(linha))
        for linha in con.execute(consulta, parametros)
    ]


def inicio_do_acompanhamento() -> str:
    return (date.today() - timedelta(days=DIAS_ACOMPANHAMENTO)).isoformat()


def filtros_de_acompanhamento(filtros: dict | None = None) -> dict:
    """Feridos ficam em checagem periódica por 90 dias, porque a situação pode
    mudar semanas depois. Passado o prazo, a edição continua possível — por
    isso a busca por data pode olhar para fora da janela padrão."""
    filtros = dict(filtros or {})
    filtros["status"] = "ferida"
    if not filtros.get("de") and not filtros.get("ate"):
        filtros["de"] = inicio_do_acompanhamento()
    return filtros
