"""Operações de domínio: reprocessamento, edição de vítimas, duplicidade e
aprovação. As rotas em app.py só traduzem HTTP para estas funções.
"""
import json
from datetime import date, timedelta

import classificador
import indicadores
import preprocessar_noticia
from db import agora, json_carregar

CAMPOS_VITIMA = [
    "nome", "idade", "genero", "tipo_vitima", "situacao", "data_morte",
    "cargo_politico", "circunstancia", "fonte",
]

# "fonte" muda a cada passagem da LLM sem trazer informação nova sobre a
# pessoa, então não vira proposta para o analista decidir.
CAMPOS_PROPOSTA = [c for c in CAMPOS_VITIMA if c != "fonte"]

DIAS_ACOMPANHAMENTO = 90


def _linha_para_dict(linha) -> dict:
    return {k: linha[k] for k in linha.keys()}


def listar_vitimas(con, ocorrencia_id: int) -> list[dict]:
    linhas = con.execute(
        "SELECT * FROM vitima WHERE ocorrencia_id = ? ORDER BY id", (ocorrencia_id,)
    ).fetchall()
    vitimas = []
    for linha in linhas:
        vitima = _linha_para_dict(linha)
        vitima["proposta"] = json_carregar(linha["proposta"], None)
        vitimas.append(vitima)
    return vitimas


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


def _proposta_de(vitima: dict, existente: dict) -> dict:
    """Só vira proposta o que a LLM afirma e diverge da correção do analista.
    Campo que a LLM deixou vazio é ausência de informação, não contradição."""
    proposta = {}
    for campo in CAMPOS_PROPOSTA:
        novo = vitima.get(campo)
        if novo in (None, "", "nao_informado"):
            continue
        if novo != existente.get(campo):
            proposta[campo] = novo
    return proposta


def reprocessar(con, ocorrencia_id: int) -> dict:
    """Etapas 3 a 5: relê todas as notícias juntas e refaz a lista de vítimas.

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
    resultado = classificador.classificar_ocorrencia(noticias)

    manuais = [
        _linha_para_dict(linha)
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
                (
                    json.dumps(divergentes, ensure_ascii=False) if divergentes else None,
                    momento,
                    existente["id"],
                ),
            )
            continue

        con.execute(
            "INSERT INTO vitima (ocorrencia_id, chave, nome, idade, genero,"
            " tipo_vitima, situacao, data_morte, circunstancia,"
            " cargo_politico, fonte, divergencia, criado_em, atualizado_em)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ocorrencia_id, vitima["chave"], vitima["nome"], vitima["idade"],
                vitima["genero"], vitima["tipo_vitima"], vitima["situacao"],
                vitima["data_morte"], vitima["circunstancia"],
                vitima["cargo_politico"], vitima["fonte"], vitima["divergencia"],
                momento, momento,
            ),
        )

    con.execute(
        "UPDATE ocorrencia SET motivo_principal = ?, motivos_complementares = ?,"
        " indicadores_texto = ?, divergencias = ?, atualizado_em = ? WHERE id = ?",
        (
            resultado["motivo_principal"],
            json.dumps(resultado["motivos_complementares"], ensure_ascii=False),
            json.dumps(resultado["indicadores_texto"], ensure_ascii=False),
            json.dumps(resultado["divergencias"], ensure_ascii=False),
            momento,
            ocorrencia_id,
        ),
    )
    recalcular_indicadores(con, ocorrencia_id)
    return resultado


def limpar_classificacao(con, ocorrencia_id: int) -> None:
    """Sem notícias não há o que classificar; deixar a lista antiga no lugar
    faria o registro mostrar vítimas sem nenhuma fonte que as sustente."""
    con.execute("DELETE FROM vitima WHERE ocorrencia_id = ?", (ocorrencia_id,))
    con.execute(
        "UPDATE ocorrencia SET motivo_principal = NULL, motivos_complementares = NULL,"
        " indicadores = NULL, indicadores_texto = NULL, divergencias = NULL,"
        " status = 'rascunho', atualizado_em = ? WHERE id = ?",
        (agora(), ocorrencia_id),
    )


def recalcular_indicadores(con, ocorrencia_id: int) -> list[str]:
    """Etapa 7: indicadores de regra são sempre recalculados sobre a lista
    final de vítimas, então mudar uma vítima pode criar ou remover uma chacina.
    """
    linha = con.execute(
        "SELECT indicadores_texto FROM ocorrencia WHERE id = ?", (ocorrencia_id,)
    ).fetchone()
    vitimas = listar_vitimas(con, ocorrencia_id)
    calculados = indicadores.calcular(
        vitimas, json_carregar(linha["indicadores_texto"], [])
    )
    con.execute(
        "UPDATE ocorrencia SET indicadores = ?, atualizado_em = ? WHERE id = ?",
        (json.dumps(calculados, ensure_ascii=False), agora(), ocorrencia_id),
    )
    return calculados


def _normalizar_valor(campo: str, valor):
    if valor is None:
        return None
    if campo == "idade":
        texto = str(valor).strip()
        return int(texto) if texto.isdigit() else None
    texto = str(valor).strip()
    return texto or None


def atualizar_vitima(con, vitima_id: int, dados: dict, analista: str, fonte: str) -> int:
    """Etapa 6 / edição manual: grava o valor anterior de cada campo alterado."""
    atual = con.execute("SELECT * FROM vitima WHERE id = ?", (vitima_id,)).fetchone()
    momento = agora()
    mudou = False

    for campo in CAMPOS_VITIMA:
        if campo not in dados:
            continue
        novo = _normalizar_valor(campo, dados[campo])
        anterior = atual[campo]
        if novo == anterior:
            continue
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
        con.execute(f"UPDATE vitima SET {campo} = ? WHERE id = ?", (novo, vitima_id))

    if mudou:
        atualizada = con.execute(
            "SELECT * FROM vitima WHERE id = ?", (vitima_id,)
        ).fetchone()
        con.execute(
            "UPDATE vitima SET chave = ?, editado_manualmente = 1, proposta = NULL,"
            " atualizado_em = ? WHERE id = ?",
            (
                classificador.chave_vitima(_linha_para_dict(atualizada)),
                momento,
                vitima_id,
            ),
        )
        recalcular_indicadores(con, atual["ocorrencia_id"])
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
    valores = {campo: _normalizar_valor(campo, dados.get(campo)) for campo in CAMPOS_VITIMA}
    valores["genero"] = valores["genero"] or "nao_informado"
    valores["tipo_vitima"] = valores["tipo_vitima"] or "nao_informado"
    valores["situacao"] = valores["situacao"] or "ferida"
    valores["circunstancia"] = valores["circunstancia"] or "nao_se_aplica"

    cursor = con.execute(
        "INSERT INTO vitima (ocorrencia_id, chave, nome, idade, genero, tipo_vitima,"
        " situacao, data_morte, cargo_politico, circunstancia, fonte,"
        " editado_manualmente, criado_em, atualizado_em)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,1,?,?)",
        (
            ocorrencia_id, classificador.chave_vitima(valores), valores["nome"],
            valores["idade"], valores["genero"], valores["tipo_vitima"],
            valores["situacao"], valores["data_morte"], valores["cargo_politico"],
            valores["circunstancia"], valores["fonte"],
            momento, momento,
        ),
    )
    con.execute(
        "INSERT INTO vitima_historico (vitima_id, ocorrencia_id, campo, valor_anterior,"
        " valor_novo, alterado_por, alterado_em, fonte) VALUES (?,?,?,?,?,?,?,?)",
        (
            cursor.lastrowid, ocorrencia_id, "vitima", None, "incluída manualmente",
            analista, momento, valores["fonte"],
        ),
    )
    recalcular_indicadores(con, ocorrencia_id)
    marcar_para_reaprovacao(con, ocorrencia_id)
    return cursor.lastrowid


def remover_vitima(con, vitima_id: int, analista: str) -> int:
    linha = con.execute("SELECT * FROM vitima WHERE id = ?", (vitima_id,)).fetchone()
    ocorrencia_id = linha["ocorrencia_id"]
    con.execute(
        "INSERT INTO vitima_historico (vitima_id, ocorrencia_id, campo, valor_anterior,"
        " valor_novo, alterado_por, alterado_em, fonte) VALUES (?,?,?,?,?,?,?,?)",
        (
            vitima_id, ocorrencia_id, "vitima",
            linha["nome"] or "sem nome", "removida", analista, agora(), None,
        ),
    )
    con.execute("DELETE FROM vitima WHERE id = ?", (vitima_id,))
    recalcular_indicadores(con, ocorrencia_id)
    marcar_para_reaprovacao(con, ocorrencia_id)
    return ocorrencia_id


def marcar_para_reaprovacao(con, ocorrencia_id: int) -> None:
    """Etapa 10: qualquer alteração derruba a aprovação; a versão alterada só
    chega à comunicação depois de passar de novo pelo responsável."""
    con.execute(
        "UPDATE ocorrencia SET status = 'nao_aprovado', aprovado_por = NULL,"
        " aprovado_em = NULL, atualizado_em = ? WHERE id = ? AND status = 'aprovado'",
        (agora(), ocorrencia_id),
    )


def checar_duplicidade(con, ocorrencia_id: int) -> list[int]:
    """Roda depois de gravar, sem travar o cadastro: mesmo dia e bairro da
    mesma cidade. Usa o índice (data_fato, cidade, bairro)."""
    atual = con.execute(
        "SELECT * FROM ocorrencia WHERE id = ?", (ocorrencia_id,)
    ).fetchone()
    similares = con.execute(
        "SELECT id FROM ocorrencia WHERE data_fato = ? AND cidade = ? AND bairro = ?"
        " AND id != ? AND unificada_em IS NULL AND status != 'rascunho'",
        (atual["data_fato"], atual["cidade"], atual["bairro"], ocorrencia_id),
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


def vitimas_em_acompanhamento(con) -> list[dict]:
    """Feridos ficam em checagem periódica por 90 dias, porque a situação pode
    mudar semanas depois. Passado o prazo, a edição continua possível."""
    limite = (date.today() - timedelta(days=DIAS_ACOMPANHAMENTO)).isoformat()
    linhas = con.execute(
        "SELECT v.*, o.data_fato, o.cidade, o.bairro FROM vitima v"
        " JOIN ocorrencia o ON o.id = v.ocorrencia_id"
        " WHERE v.situacao = 'ferida' AND o.data_fato >= ? AND o.unificada_em IS NULL"
        " ORDER BY o.data_fato DESC",
        (limite,),
    ).fetchall()
    return [_linha_para_dict(linha) for linha in linhas]
