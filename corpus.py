"""Corpus de treino: notícias rotuladas à mão, com link e texto preservados.

Separado das ocorrências de produção de propósito. Ali uma notícia é a fonte de
um fato registrado; aqui é um exemplo de treino, e o que importa é o rótulo e a
sua procedência — sem o link, nenhum resultado de treino é auditável numa banca.

São duas tarefas encadeadas:

1. `violencia` — binária, a premissa de tudo: a notícia relata DISPARO DE ARMA
   DE FOGO? Precisa dos dois lados. Um corpus só com casos positivos não ensina
   o modelo a recusar nada, e a matriz de confusão fica sem as colunas que
   mostram o erro que importa.
2. `motivacao` e `indicadores` — só sobre as notícias positivas, usando as
   mesmas categorias que alimentam o prompt da LLM.
"""
import json
import random

import indicadores
import vocabulario
from db import agora, json_carregar

TAREFAS = {
    "violencia": "Violência armada (disparo de arma de fogo)",
    "motivacao": "Motivação principal",
    "indicadores": "Indicadores de texto",
}

PARTICOES = ["treino", "validacao", "teste"]
PROPORCAO_PADRAO = (0.70, 0.15, 0.15)

# Abaixo disso a métrica de uma classe é ruído: com 4 exemplos no teste, acertar
# 3 ou 4 muda o recall de 0,75 para 1,00. O número não trava nada, só avisa.
MINIMO_POR_CLASSE = 30


def _para_dict(linha) -> dict:
    registro = {chave: linha[chave] for chave in linha.keys()}
    if "indicadores" in registro:
        registro["indicadores"] = json_carregar(registro["indicadores"], [])
    return registro


def adicionar(con, dados: dict, anotador: str) -> int:
    momento = agora()
    cursor = con.execute(
        "INSERT INTO corpus_noticia (link, titulo, texto, violencia_armada,"
        " motivo_principal, indicadores, observacao, anotador, criado_em,"
        " atualizado_em) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            (dados.get("link") or "").strip() or None,
            (dados.get("titulo") or "").strip() or None,
            dados["texto"].strip(),
            _rotulo_binario(dados.get("violencia_armada")),
            _motivo_valido(dados),
            json.dumps(_indicadores_validos(dados), ensure_ascii=False),
            (dados.get("observacao") or "").strip() or None,
            anotador,
            momento,
            momento,
        ),
    )
    return cursor.lastrowid


def atualizar(con, noticia_id: int, dados: dict, anotador: str) -> None:
    con.execute(
        "UPDATE corpus_noticia SET link = ?, titulo = ?, texto = ?,"
        " violencia_armada = ?, motivo_principal = ?, indicadores = ?,"
        " observacao = ?, anotador = ?, atualizado_em = ? WHERE id = ?",
        (
            (dados.get("link") or "").strip() or None,
            (dados.get("titulo") or "").strip() or None,
            dados["texto"].strip(),
            _rotulo_binario(dados.get("violencia_armada")),
            _motivo_valido(dados),
            json.dumps(_indicadores_validos(dados), ensure_ascii=False),
            (dados.get("observacao") or "").strip() or None,
            anotador,
            agora(),
            noticia_id,
        ),
    )


def remover(con, noticia_id: int) -> None:
    con.execute("DELETE FROM corpus_noticia WHERE id = ?", (noticia_id,))


def _rotulo_binario(valor):
    if valor in (None, "", "nao_rotulado"):
        return None
    return 1 if valor in (1, "1", True, "sim", "true", "on") else 0


def _motivo_valido(dados: dict):
    """Motivação só existe para notícia positiva: rotular a motivação de uma
    notícia que não é violência armada contaminaria a segunda tarefa."""
    if _rotulo_binario(dados.get("violencia_armada")) != 1:
        return None
    motivo = (dados.get("motivo_principal") or "").strip()
    return motivo if motivo in vocabulario.slugs_motivacao() else None


def _indicadores_validos(dados: dict) -> list[str]:
    if _rotulo_binario(dados.get("violencia_armada")) != 1:
        return []
    disponiveis = {i["nome"] for i in indicadores.carregar() if i["tipo"] == "texto"}
    escolhidos = dados.get("indicadores") or []
    if isinstance(escolhidos, str):
        escolhidos = [escolhidos]
    return [nome for nome in escolhidos if nome in disponiveis]


def _clausulas(filtros: dict) -> tuple[str, list]:
    condicoes = ["1=1"]
    parametros: list = []

    rotulo = (filtros.get("violencia_armada") or "").strip()
    if rotulo == "sim":
        condicoes.append("violencia_armada = 1")
    elif rotulo == "nao":
        condicoes.append("violencia_armada = 0")
    elif rotulo == "nao_rotulado":
        condicoes.append("violencia_armada IS NULL")

    motivo = (filtros.get("motivo_principal") or "").strip()
    if motivo == "sem_motivo":
        condicoes.append("violencia_armada = 1 AND motivo_principal IS NULL")
    elif motivo:
        condicoes.append("motivo_principal = ?")
        parametros.append(motivo)

    busca = (filtros.get("busca") or "").strip()
    if busca:
        condicoes.append("(texto LIKE ? OR titulo LIKE ? OR link LIKE ?)")
        parametros.extend([f"%{busca}%"] * 3)

    particao = (filtros.get("particao") or "").strip()
    if particao:
        condicoes.append("particao = ?")
        parametros.append(particao)

    return " AND ".join(condicoes), parametros


def contar(con, filtros: dict) -> int:
    onde, parametros = _clausulas(filtros)
    return con.execute(
        f"SELECT COUNT(*) AS total FROM corpus_noticia WHERE {onde}", parametros
    ).fetchone()["total"]


def listar(con, filtros: dict, limite=None, deslocamento=0) -> list[dict]:
    onde, parametros = _clausulas(filtros)
    consulta = f"SELECT * FROM corpus_noticia WHERE {onde} ORDER BY id DESC"
    if limite is not None:
        consulta += " LIMIT ? OFFSET ?"
        parametros = [*parametros, limite, deslocamento]
    return [_para_dict(linha) for linha in con.execute(consulta, parametros)]


def buscar(con, noticia_id: int) -> dict | None:
    linha = con.execute(
        "SELECT * FROM corpus_noticia WHERE id = ?", (noticia_id,)
    ).fetchone()
    return _para_dict(linha) if linha else None


def estatisticas(con) -> dict:
    """Diz, por tarefa, se o corpus já sustenta uma medição honesta."""
    total = con.execute("SELECT COUNT(*) AS t FROM corpus_noticia").fetchone()["t"]
    positivos = con.execute(
        "SELECT COUNT(*) AS t FROM corpus_noticia WHERE violencia_armada = 1"
    ).fetchone()["t"]
    negativos = con.execute(
        "SELECT COUNT(*) AS t FROM corpus_noticia WHERE violencia_armada = 0"
    ).fetchone()["t"]
    nao_rotulados = total - positivos - negativos

    por_motivo = {
        linha["motivo_principal"]: linha["t"]
        for linha in con.execute(
            "SELECT motivo_principal, COUNT(*) AS t FROM corpus_noticia"
            " WHERE violencia_armada = 1 AND motivo_principal IS NOT NULL"
            " GROUP BY motivo_principal"
        )
    }
    motivos = {slug: por_motivo.get(slug, 0) for slug in vocabulario.slugs_motivacao()}

    por_indicador = {}
    for registro in con.execute(
        "SELECT indicadores FROM corpus_noticia WHERE violencia_armada = 1"
    ):
        for nome in json_carregar(registro["indicadores"], []):
            por_indicador[nome] = por_indicador.get(nome, 0) + 1
    nomes_indicador = [
        i["nome"] for i in indicadores.carregar() if i["tipo"] == "texto"
    ]
    indicadores_contagem = {
        nome: por_indicador.get(nome, 0) for nome in nomes_indicador
    }

    return {
        "total": total,
        "positivos": positivos,
        "negativos": negativos,
        "nao_rotulados": nao_rotulados,
        "motivos": motivos,
        "indicadores": indicadores_contagem,
        "sem_motivo": positivos - sum(motivos.values()),
        "prontidao": {
            "violencia": _prontidao([positivos, negativos]),
            "motivacao": _prontidao(
                [n for n in motivos.values() if n], minimo_classes=2
            ),
            "indicadores": _prontidao(
                [n for n in indicadores_contagem.values() if n], minimo_classes=1
            ),
        },
        "minimo_por_classe": MINIMO_POR_CLASSE,
    }


def _prontidao(contagens: list[int], minimo_classes: int = 2) -> dict:
    presentes = [n for n in contagens if n > 0]
    if len(presentes) < minimo_classes:
        return {
            "pode_treinar": False,
            "motivo": f"faltam classes com exemplo (mínimo {minimo_classes})",
        }
    menor = min(presentes)
    if menor < 3:
        return {
            "pode_treinar": False,
            "motivo": f"a classe mais rara tem {menor} exemplo(s); são precisos 3"
            " para dividir em treino, validação e teste",
        }
    if menor < MINIMO_POR_CLASSE:
        return {
            "pode_treinar": True,
            "motivo": f"treina, mas a classe mais rara tem {menor} exemplo(s):"
            f" abaixo de {MINIMO_POR_CLASSE} a métrica dela é instável",
        }
    return {"pode_treinar": True, "motivo": "ok"}


def dividir(con, proporcao=PROPORCAO_PADRAO, semente: int = 42) -> dict:
    """Sorteia treino/validação/teste de forma estratificada e grava a partição.

    A divisão é estratificada por (violência, motivação) para que nenhuma classe
    rara fique inteira de um lado só, e fica GRAVADA no banco: refazer o sorteio
    a cada treino mudaria o conjunto de teste e tornaria os modelos
    incomparáveis entre si.
    """
    aleatorio = random.Random(semente)

    # Agrupa pelo rótulo binário e, dentro dele, pela motivação.
    grupos: dict[int, dict] = {}
    for linha in con.execute(
        "SELECT id, violencia_armada, motivo_principal FROM corpus_noticia"
        " WHERE violencia_armada IS NOT NULL ORDER BY id"
    ):
        por_motivo = grupos.setdefault(linha["violencia_armada"], {})
        por_motivo.setdefault(linha["motivo_principal"], []).append(linha["id"])

    contagem = {particao: 0 for particao in PARTICOES}
    for por_motivo in grupos.values():
        # Intercala as motivações numa só sequência em vez de fatiar cada uma
        # separadamente. Estratificar pela motivação parece mais correto, mas com
        # 15 categorias quase todo estrato fica com um ou dois exemplos, e aí a
        # regra "estrato pequeno vai inteiro para treino" esvazia a validação e o
        # teste justamente da classe positiva. Intercalando, a proporção do
        # rótulo binário é exata e as motivações se espalham entre as partições.
        listas = []
        for ids in por_motivo.values():
            embaralhado = list(ids)
            aleatorio.shuffle(embaralhado)
            listas.append(embaralhado)
        aleatorio.shuffle(listas)

        sequencia = []
        for posicao in range(max((len(l) for l in listas), default=0)):
            for lista in listas:
                if posicao < len(lista):
                    sequencia.append(lista[posicao])

        total = len(sequencia)
        # Treino primeiro: como a sequência começa com o primeiro exemplo de cada
        # motivação, nenhuma categoria fica sem exemplo de treino.
        n_treino = min(total, max(1, round(total * proporcao[0]))) if total else 0
        n_validacao = min(total - n_treino, round(total * proporcao[1]))
        fatias = {
            "treino": sequencia[:n_treino],
            "validacao": sequencia[n_treino:n_treino + n_validacao],
            "teste": sequencia[n_treino + n_validacao:],
        }

        for particao, lista in fatias.items():
            for noticia_id in lista:
                con.execute(
                    "UPDATE corpus_noticia SET particao = ? WHERE id = ?",
                    (particao, noticia_id),
                )
                contagem[particao] += 1

    return contagem


def exemplos(con, tarefa: str, particao: str | None = None) -> list[dict]:
    """Devolve (texto, rótulo) para treino. Cada tarefa enxerga um subconjunto:
    `violencia` usa todo o corpus rotulado; as outras, só as positivas."""
    condicoes = ["violencia_armada IS NOT NULL"]
    if tarefa == "motivacao":
        condicoes = ["violencia_armada = 1", "motivo_principal IS NOT NULL"]
    elif tarefa == "indicadores":
        condicoes = ["violencia_armada = 1"]
    if particao:
        condicoes.append("particao = ?")

    parametros = [particao] if particao else []
    linhas = con.execute(
        "SELECT id, titulo, texto, violencia_armada, motivo_principal, indicadores"
        f" FROM corpus_noticia WHERE {' AND '.join(condicoes)} ORDER BY id",
        parametros,
    ).fetchall()

    resultado = []
    for linha in linhas:
        registro = _para_dict(linha)
        texto = registro["texto"]
        if registro.get("titulo"):
            texto = f"{registro['titulo']}\n{texto}"
        if tarefa == "violencia":
            rotulo = registro["violencia_armada"]
        elif tarefa == "motivacao":
            rotulo = registro["motivo_principal"]
        else:
            rotulo = registro["indicadores"]
        resultado.append({"id": registro["id"], "texto": texto, "rotulo": rotulo})
    return resultado


def rotulos_da_tarefa(tarefa: str) -> list[str]:
    if tarefa == "violencia":
        return ["nao_violencia_armada", "violencia_armada"]
    if tarefa == "motivacao":
        return vocabulario.slugs_motivacao()
    return [i["nome"] for i in indicadores.carregar() if i["tipo"] == "texto"]
