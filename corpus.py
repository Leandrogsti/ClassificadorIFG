"""Corpus de treino: notícias rotuladas à mão, com link e texto preservados.

Separado das ocorrências de produção de propósito. Ali uma notícia é a fonte de
um fato registrado; aqui é um exemplo de treino, e o que importa é o rótulo e a
sua procedência — sem o link, nenhum resultado de treino é auditável numa banca.

É UMA tarefa só, multirrótulo: quais dos oito indicadores a notícia
sustenta.
Não há mais filtro binário de "houve disparo?" — uma notícia fora do escopo é
simplesmente a que não recebe indicador nenhum, e essas são tão necessárias
quanto as positivas: sem elas o modelo nunca aprende a não marcar nada, e a
matriz de confusão fica sem as colunas que mostram o erro que importa.

NÃO se anota aqui nada que a LLM preencha: ficha de vítima, circunstância,
tabela complementar. O rótulo é só a lista de indicadores, porque é só isso que
o BERT aprende do texto.

DE ONDE VÊM OS EXEMPLOS. Três portas, e nenhuma delas pede trabalho extra —
todas aproveitam decisão que alguém já tomou por outro motivo:

1. a rotulagem na fila de `recebidas.py`, inclusive a que conclui que não há
   fenômeno nenhum, que é o exemplo negativo;
2. a aprovação de uma ocorrência (`servico.enviar_ao_corpus`), em que os
   fenômenos confirmados pelo analista viram o rótulo das notícias dela;
3. a importação de planilha, para corpus que já existia fora daqui.

ESTAR NO CORPUS NÃO É ESTAR NO TREINO. São duas perguntas, e duas colunas:

- `rotulada` — alguém leu e decidiu. Pode ter sido o analista júnior que
  cadastrou a ocorrência.
- `aprovada` — um sênior conferiu a decisão e assumiu que ela está certa.

**Só o aprovado treina o modelo** (`TREINAVEL`). O motivo é a divisão de
trabalho real: quem cadastra é júnior, quem revisa é sênior, e um rótulo errado
no corpus não produz um erro visível — produz um modelo pior, que erra em
notícia que ninguém está olhando. É o tipo de defeito que não aparece em
nenhuma tela, e por isso a trava é aqui e não na confiança.

O que não está aprovado continua no corpus, visível e editável em `/corpus`,
esperando a revisão. Não é descartado: é exatamente a fila de trabalho do
sênior.

Texto repetido não entra duas vezes: `impressao` é o sha1 do texto normalizado,
e a mesma notícia em treino e em teste seria vazamento — métrica que sobe sem o
modelo melhorar.
"""
import hashlib
import json
import random
import re

import indicadores
from db import agora, json_carregar

# A CONDIÇÃO DE TREINO, escrita UMA vez. Toda consulta que decide o que o
# modelo vê usa esta constante: sorteio de partição, contagem por partição,
# leitura dos exemplos. Espalhar `rotulada = 1` por sete consultas foi o que
# permitiu, antes, que o rótulo do júnior entrasse no treino sem passar por
# ninguém — basta esquecer uma delas.
TREINAVEL = "rotulada = 1 AND aprovada = 1"

TAREFAS = {
    "indicadores": "Indicadores (multirrótulo)",
}

PARTICOES = ["treino", "validacao", "teste"]
PROPORCAO_PADRAO = (0.70, 0.15, 0.15)

# POSITIVOS APROVADOS EXIGIDOS DE CADA CLASSE PARA O TREINO RODAR.
#
# Abaixo disso a métrica de uma classe é ruído: com 4 exemplos no teste,
# acertar 3 ou 4 muda o recall de 0,75 para 1,00. Até 14/10/2026 este número
# só avisava; agora TRAVA, por decisão da usuária. O que o aviso deixava
# passar era pior que o incômodo da trava — uma classe com 2 exemplos, ou um
# indicador recém-criado com zero, treinava para dizer sempre não e saía com
# F1 0, que parece defeito do modelo e é falta de dado.
#
# A trava vale para TODAS as classes do catálogo. Criar um indicador novo
# portanto barra o treino até ele ter 30 positivos aprovados — é o
# comportamento pretendido, e a saída é `rodar_treino.py --classe`, que treina
# só o subconjunto já pronto.
MINIMO_POR_CLASSE = 30

# A notícia sem nenhum indicador entra na estratificação como se fosse uma
# classe. Sem isso ela só seria distribuída no fim, pela vaga que sobrasse, e as
# negativas — que são o que ensina o modelo a não marcar nada — ficariam
# desproporcionais entre as partições. O \x00 garante que nenhum slug de
# indicador possa colidir com esta chave.
SEM_INDICADOR = "\x00sem_indicador"


def impressao(texto: str) -> str:
    """Identidade do texto para fins de repetição.

    Normaliza espaço e caixa antes de resumir em sha1: o mesmo artigo colado de
    duas URLs diferentes (uma com `#google_vignette` no fim, por exemplo) chega
    com o texto igual e tem de ser reconhecido como o mesmo exemplo. Não vai
    além disso de propósito — tirar acento ou pontuação começaria a juntar
    notícias que são de fatos parecidos mas distintos.
    """
    normalizado = re.sub(r"\s+", " ", (texto or "")).strip().lower()
    return hashlib.sha1(normalizado.encode("utf-8")).hexdigest()


def ja_no_corpus(con, texto: str) -> int | None:
    """O id da notícia do corpus que tem este mesmo texto, se houver."""
    linha = con.execute(
        "SELECT id FROM corpus_noticia WHERE impressao = ?", (impressao(texto),)
    ).fetchone()
    return linha["id"] if linha else None


def pendentes_de_particao(con) -> int:
    """Quantas notícias anotadas ainda não têm partição.

    NÃO se sorteia a partição de uma notícia na inserção. Tentei, e medi o
    resultado: sorteando uma por uma, sem saber o que já foi distribuído, três
    das classes ficaram sem nenhum exemplo na validação e o conjunto de
    teste ficou sem nenhuma negativa. A estratificação só funciona em lote,
    porque ela precisa olhar o conjunto todo e começar pela classe mais rara.

    O que impede a notícia nova de ficar invisível não é o sorteio imediato: é
    `dividir` ser chamada antes de treinar, com o lote inteiro do que está
    pendente.
    """
    return con.execute(
        "SELECT COUNT(*) AS total FROM corpus_noticia"
        f" WHERE {TREINAVEL} AND particao IS NULL"
    ).fetchone()["total"]


def _para_dict(linha) -> dict:
    registro = {chave: linha[chave] for chave in linha.keys()}
    if "indicadores" in registro:
        registro["indicadores"] = json_carregar(registro["indicadores"], [])
    return registro


def _indicadores_validos(dados: dict) -> list[str]:
    """Descarta nome que não existe mais no catálogo, para que um indicador
    renomeado em /gestao não vire uma classe fantasma no treino."""
    disponiveis = indicadores.nomes()
    escolhidos = dados.get("indicadores") or []
    if isinstance(escolhidos, str):
        escolhidos = [escolhidos]
    return [nome for nome in disponiveis if nome in set(escolhidos)]


def _rotulada(dados: dict, marcados: list[str]) -> int:
    """Marcar um indicador já é anotar. O formulário de anotação manda
    `rotulada` explícito para que "nenhum indicador se aplica" também conte
    como decisão tomada, em vez de devolver a notícia para a fila."""
    if dados.get("rotulada") in (1, "1", True, "sim", "on", "true"):
        return 1
    return 1 if marcados else 0


def adicionar(con, dados: dict, anotador: str) -> int:
    """Grava a notícia no corpus e devolve o id.

    Texto repetido NÃO cria linha nova: devolve o id da que já existe. É o que
    permite alimentar o corpus a cada aprovação sem que reprocessar a mesma
    ocorrência duplique os exemplos.

    A partição fica NULL: quem a atribui é `dividir`, em lote, e o treino a
    chama sozinho antes de começar. Ver `pendentes_de_particao` para o porquê.

    `aprovada` fica 0 a menos que quem chama passe `aprovada` explicitamente.
    Só a aprovação de uma ocorrência passa — ali um sênior já conferiu o
    rótulo. A rotulagem na fila e a importação de planilha não passam, e o
    exemplo espera revisão em /corpus antes de treinar.
    """
    texto = dados["texto"].strip()
    repetida = ja_no_corpus(con, texto)
    if repetida is not None:
        return repetida

    momento = agora()
    marcados = _indicadores_validos(dados)
    rotulada = _rotulada(dados, marcados)
    aprovada = 1 if dados.get("aprovada") in (1, "1", True, "sim", "on") else 0
    cursor = con.execute(
        "INSERT INTO corpus_noticia (link, texto, indicadores, rotulada,"
        " anotador, impressao, aprovada, aprovada_por, aprovada_em,"
        " criado_em, atualizado_em)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            (dados.get("link") or "").strip() or None,
            texto,
            json.dumps(marcados, ensure_ascii=False),
            rotulada,
            anotador,
            impressao(texto),
            aprovada,
            anotador if aprovada else None,
            momento if aprovada else None,
            momento,
            momento,
        ),
    )
    return cursor.lastrowid


def a_reanotar(con) -> dict[str, int]:
    """Quantas notícias foram anotadas ANTES de cada indicador existir.

    Esse é o buraco que criar um indicador abre, e ele é silencioso: a notícia
    anotada antes nunca foi lida com a pergunta nova, mas no treino ela conta
    como exemplo NEGATIVO da classe nova. Se trinta das suas notícias antigas
    de fato sustentam o indicador recém-criado, você está ensinando ao modelo
    que não sustentam — e o sintoma é um indicador que o modelo nunca marca,
    que parece defeito do modelo e é defeito do rótulo.

    A comparação é com `atualizado_em`, não com `criado_em` da notícia: quem
    editou a linha depois do indicador nascer viu a caixa dele na tela e
    decidiu, mesmo que tenha decidido não marcar.

    Indicador sem `criado_em` é tratado como se sempre tivesse existido —
    são os do catálogo original, anteriores a este controle, e não há data
    com que comparar.
    """
    pendentes: dict[str, int] = {}
    for indicador in indicadores.carregar():
        nascimento = (indicador.get("criado_em") or "").strip()
        if not nascimento or indicador.get("decidido_por") != "bert":
            continue
        n = con.execute(
            "SELECT COUNT(*) AS t FROM corpus_noticia"
            " WHERE rotulada = 1 AND atualizado_em < ?"
            "   AND indicadores NOT LIKE ?",
            (nascimento, f'%"{indicador["nome"]}"%'),
        ).fetchone()["t"]
        if n:
            pendentes[indicador["nome"]] = n
    return pendentes


def aprovar(con, noticia_id: int, revisor: str, aprovada: bool = True) -> None:
    """O sênior assume o rótulo — ou o devolve para revisão.

    APROVAR NÃO MEXE NA PARTIÇÃO, e desaprovar a limpa. Um exemplo que sai do
    treino não pertence a partição nenhuma; se ele voltar, `dividir` o sorteia
    de novo com o lote pendente. Manter a partição de quem saiu deixaria vaga
    reservada para exemplo que não vai ser usado, e era isso que desequilibrava
    a estratificação.
    """
    momento = agora()
    if aprovada:
        con.execute(
            "UPDATE corpus_noticia SET aprovada = 1, aprovada_por = ?,"
            " aprovada_em = ?, atualizado_em = ? WHERE id = ?",
            (revisor, momento, momento, noticia_id),
        )
    else:
        con.execute(
            "UPDATE corpus_noticia SET aprovada = 0, aprovada_por = NULL,"
            " aprovada_em = NULL, particao = NULL, atualizado_em = ?"
            " WHERE id = ?",
            (momento, noticia_id),
        )


def atualizar(con, noticia_id: int, dados: dict, anotador: str) -> None:
    """Corrige o rótulo de um exemplo.

    MUDAR O RÓTULO DERRUBA A APROVAÇÃO. Rótulo novo é afirmação nova, e quem
    assinou a anterior não assinou esta. Quem chama pode reaprovar no mesmo
    envio passando `aprovada` — é o que o formulário de /corpus faz, para que
    corrigir e assumir a correção sejam um clique só.
    """
    marcados = _indicadores_validos(dados)
    texto = dados["texto"].strip()
    marca = impressao(texto)
    rotulada = _rotulada(dados, marcados)
    # A partição já atribuída é PRESERVADA: mover um exemplo do teste para o
    # treino ao corrigir o rótulo dele invalidaria a comparação com as execuções
    # anteriores. Desanotar, sim, a zera — exemplo sem rótulo não pertence a
    # partição nenhuma.
    atual = con.execute(
        "SELECT particao, indicadores, aprovada FROM corpus_noticia WHERE id = ?",
        (noticia_id,),
    ).fetchone()
    particao = (atual["particao"] if atual else None) if rotulada else None

    antes = json_carregar(atual["indicadores"], []) if atual else []
    mudou_rotulo = sorted(antes) != sorted(marcados)
    if "aprovada" in dados:
        aprovada = 1 if dados.get("aprovada") in (1, "1", True, "sim", "on") else 0
    elif mudou_rotulo:
        aprovada = 0
    else:
        aprovada = int(atual["aprovada"]) if atual else 0

    momento = agora()
    con.execute(
        "UPDATE corpus_noticia SET link = ?, texto = ?, indicadores = ?,"
        " rotulada = ?, anotador = ?, impressao = ?, particao = ?,"
        " aprovada = ?, aprovada_por = ?, aprovada_em = ?,"
        " atualizado_em = ? WHERE id = ?",
        (
            (dados.get("link") or "").strip() or None,
            texto,
            json.dumps(marcados, ensure_ascii=False),
            rotulada,
            anotador,
            marca,
            # Correção de rótulo NÃO devolve a vaga de partição ao sorteio,
            # mesmo quando derruba a aprovação: reaprovar depois traria o
            # exemplo de volta em outra partição, e um exemplo que atravessa
            # do teste para o treino contamina a medição. Quem sai de vez sai
            # por `aprovar(..., False)`, e é lá que a partição é limpa.
            particao,
            aprovada,
            anotador if aprovada else None,
            momento if aprovada else None,
            momento,
            noticia_id,
        ),
    )


def remover(con, noticia_id: int) -> None:
    con.execute("DELETE FROM corpus_noticia WHERE id = ?", (noticia_id,))


def _clausulas(filtros: dict) -> tuple[str, list]:
    condicoes = ["1=1"]
    parametros: list = []

    situacao = (filtros.get("rotulada") or "").strip()
    if situacao == "sim":
        condicoes.append("rotulada = 1")
    elif situacao == "nao":
        condicoes.append("rotulada = 0")
    elif situacao == "sem_indicador":
        # Negativas: lidas, decididas, e nenhum indicador se aplica.
        condicoes.append("rotulada = 1 AND (indicadores IS NULL OR indicadores = '[]')")

    # As notícias que precisam ser relidas por causa de um indicador criado
    # depois delas. É o filtro que torna a fila de reanotação trabalhável:
    # sem ele, a pessoa saberia que há 40 pendentes e não quais.
    reanotar = (filtros.get("reanotar") or "").strip()
    if reanotar in indicadores.nomes():
        indicador_novo = indicadores.buscar(reanotar) or {}
        nascimento = (indicador_novo.get("criado_em") or "").strip()
        if nascimento:
            condicoes.append(
                "rotulada = 1 AND atualizado_em < ? AND indicadores NOT LIKE ?"
            )
            parametros.extend([nascimento, f'%"{reanotar}"%'])

    aprovacao = (filtros.get("aprovada") or "").strip()
    if aprovacao == "sim":
        condicoes.append("aprovada = 1")
    elif aprovacao == "nao":
        condicoes.append("aprovada = 0")
    elif aprovacao == "esperando":
        # A fila do sênior: o júnior já decidiu, ninguém conferiu ainda.
        condicoes.append("rotulada = 1 AND aprovada = 0")

    indicador = (filtros.get("indicador") or "").strip()
    if indicador in indicadores.nomes():
        # O JSON é uma lista de slugs entre aspas: casar com as aspas evita que
        # `acao_policial` traga também um futuro `acao_policial_indireta`.
        condicoes.append("indicadores LIKE ?")
        parametros.append(f'%"{indicador}"%')

    busca = (filtros.get("busca") or "").strip()
    if busca:
        condicoes.append("(texto LIKE ? OR link LIKE ?)")
        parametros.extend([f"%{busca}%"] * 2)

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


def estatisticas(con, classes: list[str] | None = None) -> dict:
    """Diz se o corpus já sustenta uma medição honesta.

    A PRONTIDÃO É CONTADA SOBRE O APROVADO, não sobre o rotulado. Contar o
    rotulado prometeria uma medição que o treino não vai fazer: a tela diria
    "pronto para treinar" com exemplos que a condição de treino exclui.
    `rotuladas_sem_aprovar` é a fila do sênior, e aparece ao lado para que a
    diferença entre os dois números seja visível em vez de surpresa.
    """
    total = con.execute("SELECT COUNT(*) AS t FROM corpus_noticia").fetchone()["t"]
    rotuladas = con.execute(
        "SELECT COUNT(*) AS t FROM corpus_noticia WHERE rotulada = 1"
    ).fetchone()["t"]
    treinaveis = con.execute(
        f"SELECT COUNT(*) AS t FROM corpus_noticia WHERE {TREINAVEL}"
    ).fetchone()["t"]
    pendentes = total - rotuladas

    por_indicador: dict[str, int] = {}
    negativas = 0
    for registro in con.execute(
        f"SELECT indicadores FROM corpus_noticia WHERE {TREINAVEL}"
    ):
        marcados = json_carregar(registro["indicadores"], [])
        if not marcados:
            negativas += 1
        for nome in marcados:
            por_indicador[nome] = por_indicador.get(nome, 0) + 1

    contagem = {nome: por_indicador.get(nome, 0) for nome in indicadores.nomes()}
    por_particao = _contagem_por_particao(con)

    # `classes` restringe a prontidão ao subconjunto que vai ser treinado. É o
    # que faz `rodar_treino.py --classe` continuar possível quando uma classe
    # rara trava o catálogo inteiro: a trava passa a valer só sobre o que foi
    # pedido. A contagem mostrada na tela continua sendo a de todas.
    do_treino = (
        contagem if not classes
        else {nome: contagem.get(nome, 0) for nome in classes}
    )

    return {
        "total": total,
        "rotuladas": rotuladas,
        "treinaveis": treinaveis,
        "rotuladas_sem_aprovar": rotuladas - treinaveis,
        "pendentes": pendentes,
        "negativas": negativas,
        "indicadores": contagem,
        "por_particao": por_particao,
        "rotulos": indicadores.rotulos(),
        "prontidao": {
            "indicadores": _prontidao(treinaveis, negativas, do_treino, por_particao)
        },
        "minimo_por_classe": MINIMO_POR_CLASSE,
    }


def _contagem_por_particao(con) -> dict[str, dict]:
    """Quantos exemplos de cada indicador há em cada partição.

    É o número que diz se um indicador pode sequer ser medido. Sem exemplo dele
    na VALIDAÇÃO não há como calibrar o limiar daquele indicador, e ele fica no
    corte neutro de 0,5 — que é justamente onde a classe rara desaparece. Sem
    exemplo no TESTE, o F1 dele sai 0 com suporte 0, o que parece erro do modelo
    e não é: é falta de exemplo onde o modelo é avaliado.
    """
    resumo = {
        particao: {
            "total": 0,
            "negativas": 0,
            "indicadores": {nome: 0 for nome in indicadores.nomes()},
        }
        for particao in PARTICOES
    }
    for linha in con.execute(
        "SELECT particao, indicadores FROM corpus_noticia"
        f" WHERE {TREINAVEL} AND particao IS NOT NULL"
    ):
        bloco = resumo.get(linha["particao"])
        if bloco is None:
            continue
        bloco["total"] += 1
        marcados = json_carregar(linha["indicadores"], [])
        if not marcados:
            bloco["negativas"] += 1
        for nome in marcados:
            if nome in bloco["indicadores"]:
                bloco["indicadores"][nome] += 1
    return resumo


def _prontidao(aprovadas: int, negativas: int, contagem: dict[str, int],
               por_particao: dict[str, dict] | None = None) -> dict:
    """Se o corpus sustenta um treino honesto. Conta o que está APROVADO, não o
    que está rotulado: é o aprovado que o treino vai ler, e prometer uma
    medição com exemplo que a condição de treino exclui seria mentir na tela."""
    presentes = {nome: n for nome, n in contagem.items() if n > 0}

    if aprovadas < 3:
        return {
            "pode_treinar": False,
            "motivo": f"só {aprovadas} notícia(s) aprovada(s) para treino; são"
            " precisas ao menos 3 para dividir em treino, validação e teste",
        }
    if not presentes:
        return {
            "pode_treinar": False,
            "motivo": "nenhum indicador tem exemplo: o modelo não teria o que aprender",
        }

    avisos = []
    if negativas == 0:
        avisos.append(
            "não há notícia aprovada sem nenhum indicador — sem exemplos"
            " negativos o modelo não aprende a recusar nada"
        )

    # A TRAVA DOS 30, decidida em 2026-10-14. Antes isto era só um aviso, e o
    # treino rodava com uma classe de 2 exemplos — ou de ZERO, no caso de um
    # indicador recém-criado, porque a classe sem nenhum positivo nem entrava
    # em `presentes` e escapava da conta. O resultado era uma classe que treina
    # para dizer sempre não e sai com F1 0, e esse zero parece defeito do
    # modelo quando é falta de dado.
    #
    # A conta é sobre TODAS as classes do catálogo, não só as que já têm
    # exemplo: é isso que faz criar um indicador novo barrar o treino até ele
    # ser anotado, que é o comportamento pedido.
    curtas = {
        nome: contagem.get(nome, 0)
        for nome in contagem
        if contagem.get(nome, 0) < MINIMO_POR_CLASSE
    }
    if curtas:
        detalhe = ", ".join(
            f"{nome} ({n}/{MINIMO_POR_CLASSE})"
            for nome, n in sorted(curtas.items(), key=lambda x: x[1])
        )
        return {
            "pode_treinar": False,
            "motivo": f"{len(curtas)} classe(s) ainda não têm"
            f" {MINIMO_POR_CLASSE} positivos aprovados: {detalhe}",
            "curtas": curtas,
            "avisos": avisos,
            # A saída quando uma classe rara trava o resto: treinar só as que
            # já estão prontas. Sem isto, um indicador difícil de encontrar
            # segura o treino do catálogo inteiro para sempre.
            "prontas": sorted(
                nome for nome, n in contagem.items() if n >= MINIMO_POR_CLASSE
            ),
        }

    # O que a divisão deixou de fora. Um indicador que não aparece na validação
    # nem no teste não é classificado pior: ele é INVISÍVEL para a calibração e
    # para a medição, e é esse o motivo mais comum de um indicador nunca ser
    # marcado em produção.
    if por_particao:
        vazias = [p for p in PARTICOES if not por_particao[p]["total"]]
        if len(vazias) == len(PARTICOES):
            # Nenhuma notícia em partição nenhuma: o sorteio nunca foi feito, ou
            # foi feito antes de estas notícias serem anotadas.
            return {
                "pode_treinar": False,
                "motivo": "a divisão em treino/validação/teste ainda não foi"
                " sorteada para estas notícias — sorteie antes de treinar",
            }
        if "teste" in vazias:
            return {
                "pode_treinar": False,
                "motivo": "a divisão deixou o conjunto de teste vazio: sem ele não"
                " há métrica para reportar. Anote mais notícias e sorteie a"
                " divisão de novo",
            }
        if vazias:
            avisos.append(f"sem nenhuma notícia em: {', '.join(vazias)}")
        for particao, consequencia in (
            ("validacao", "ficam no corte neutro de 0,5, onde a classe rara desaparece"),
            ("teste", "saem com F1 0 e suporte 0, que não é erro do modelo"),
        ):
            if not por_particao[particao]["total"]:
                continue
            ausentes = [
                nome for nome, n in por_particao[particao]["indicadores"].items()
                if n == 0 and contagem.get(nome, 0) > 0
            ]
            if ausentes:
                avisos.append(
                    f"sem exemplo na {particao}: {', '.join(ausentes)} —"
                    f" {consequencia}"
                )

    return {
        "pode_treinar": True,
        "motivo": "; ".join(avisos) if avisos else "ok",
    }


def _estratificar(exemplos: list[tuple[int, set[str]]], classes: list[str],
                  proporcao, aleatorio: random.Random) -> dict[str, list[int]]:
    """Estratificação iterativa para multirrótulo (Sechidis et al., 2011).

    Em multirrótulo não dá para fatiar classe por classe: a mesma notícia pode
    ter três indicadores e pertenceria a três estratos ao mesmo tempo. Tratar a
    COMBINAÇÃO de indicadores como estrato tampouco resolve — com oito
    indicadores existem 256 combinações possíveis, quase toda combinação tem um
    ou dois exemplos, e um estrato de um exemplo cai inteiro num lado só. O
    resultado prático disso é o pior possível: as classes raras, que vivem nas
    combinações raras, somem da validação e do teste, e aí não há como calibrar
    nem medir exatamente os indicadores que mais precisam disso.

    O método aqui é outro: distribui uma notícia por vez, e sempre começando
    pela classe com MENOS exemplos ainda por distribuir. Cada partição carrega
    uma cota por classe (`vagas_classe`) e uma cota geral (`vagas`), e a notícia
    vai para a partição mais necessitada da classe em questão. Como a classe
    mais rara é servida primeiro, os poucos exemplos dela são repartidos entre
    treino, validação e teste em vez de caírem todos no primeiro.

    A notícia sem nenhum indicador entra como se `SEM_INDICADOR` fosse uma
    classe dela. É o que mantém as negativas proporcionais: elas são o que
    ensina o modelo a não marcar nada, e deixá-las para o fim, pela vaga que
    sobrasse, faria uma partição ficar sem nenhuma.
    """
    marcas = {
        pid: (set(rotulos) & set(classes)) or {SEM_INDICADOR}
        for pid, rotulos in exemplos
    }

    total = len(exemplos)
    vagas = {p: total * r for p, r in zip(PARTICOES, proporcao)}

    de_classe: dict[str, int] = {}
    for conjunto in marcas.values():
        for classe in conjunto:
            de_classe[classe] = de_classe.get(classe, 0) + 1
    vagas_classe = {
        c: {p: n * r for p, r in zip(PARTICOES, proporcao)}
        for c, n in de_classe.items()
    }

    fatias: dict[str, list[int]] = {p: [] for p in PARTICOES}
    pendentes = dict(marcas)

    while pendentes:
        restam: dict[str, int] = {}
        for conjunto in pendentes.values():
            for classe in conjunto:
                restam[classe] = restam.get(classe, 0) + 1

        # A mais rara primeiro: é assim que os poucos exemplos dela sobram para
        # a validação e o teste em vez de caírem todos no treino. O desempate
        # pelo nome mantém o sorteio reprodutível quando duas classes têm a
        # mesma contagem.
        classe = min(restam, key=lambda c: (restam[c], c))
        lote = [pid for pid, conjunto in pendentes.items() if classe in conjunto]

        for pid in lote:
            # A partição que já recebeu a sua parte sai da disputa, senão o
            # treino — que tem a maior cota em toda classe — levaria também o
            # resto. Com três notícias anotadas, é o que garante uma em cada
            # partição em vez de três no treino. Se nenhuma tem vaga, todas
            # voltam a valer e a cota da classe decide sozinha.
            elegiveis = [p for p in PARTICOES if vagas[p] >= 1] or list(PARTICOES)
            # Mais necessitada NAQUELA classe; empate vai para quem tem mais
            # vaga geral; empate de novo, sorteio.
            destino = max(
                elegiveis,
                key=lambda p: (vagas_classe[classe][p], vagas[p], aleatorio.random()),
            )
            fatias[destino].append(pid)
            vagas[destino] -= 1
            # Desconta de TODAS as classes da notícia, não só da que a
            # escolheu: ela ocupa uma vaga de cada indicador que carrega.
            for marcada in pendentes.pop(pid):
                vagas_classe[marcada][destino] -= 1

    return fatias


def dividir(con, proporcao=PROPORCAO_PADRAO, semente: int = 42,
            refazer: bool = False) -> dict:
    """Sorteia treino/validação/teste de forma estratificada e grava a partição.

    A estratificação é a de `_estratificar`: por classe e uma notícia por vez,
    começando pela classe mais rara. É o que garante que cada indicador tenha
    exemplo nas três partições — sem exemplo na validação não há limiar para
    calibrar, e sem exemplo no teste não há métrica para reportar.

    POR PADRÃO SÓ DISTRIBUI O QUE AINDA NÃO TEM PARTIÇÃO. A versão anterior
    resorteava todas as linhas a cada chamada, contrariando o que ela própria
    prometia: com o corpus crescendo a cada aprovação, o conjunto de teste
    mudava toda vez e duas execuções de treino deixavam de ser comparáveis —
    um modelo medido na semana passada tinha sido avaliado num teste que já não
    existe. Congelar o que já foi sorteado é o que torna o placar honesto.

    `refazer=True` resorteia tudo, e é operação destrutiva do ponto de vista da
    medição: invalida a comparação com qualquer execução anterior.
    """
    aleatorio = random.Random(semente)

    condicao = TREINAVEL if refazer else f"{TREINAVEL} AND particao IS NULL"
    exemplos = [
        (linha["id"], set(json_carregar(linha["indicadores"], [])))
        for linha in con.execute(
            f"SELECT id, indicadores FROM corpus_noticia WHERE {condicao}"
            " ORDER BY id"
        )
    ]
    # Embaralha antes: sem isto, a ordem de inserção decidiria os empates e uma
    # leva de notícias anotadas no mesmo dia tenderia a cair junta.
    aleatorio.shuffle(exemplos)

    fatias = _estratificar(exemplos, indicadores.nomes(), proporcao, aleatorio)

    for particao, lista in fatias.items():
        for noticia_id in lista:
            con.execute(
                "UPDATE corpus_noticia SET particao = ? WHERE id = ?",
                (particao, noticia_id),
            )

    # A contagem devolvida é a do corpus INTEIRO, não a desta rodada: é o
    # número que a tela mostra, e mostrar só os recém-sorteados faria parecer
    # que o corpus encolheu.
    contagem = {particao: 0 for particao in PARTICOES}
    for linha in con.execute(
        "SELECT particao, COUNT(*) AS total FROM corpus_noticia"
        f" WHERE {TREINAVEL} AND particao IS NOT NULL GROUP BY particao"
    ):
        if linha["particao"] in contagem:
            contagem[linha["particao"]] = linha["total"]

    # Notícia ainda não anotada não pertence a partição nenhuma: se ficasse com
    # a etiqueta de um sorteio anterior, entraria no treino sem rótulo.
    con.execute("UPDATE corpus_noticia SET particao = NULL WHERE rotulada = 0")
    return contagem


# O código que o balão do botão "Sortear divisão" mostra na tela. Fica aqui, ao
# lado da função, porque um trecho copiado para o template envelhece calado.
CODIGO_DIVISAO = '''aleatorio = random.Random(semente)   # semente fixa: o mesmo sorteio sempre
aleatorio.shuffle(exemplos)

# A notícia sem indicador entra como se tivesse uma classe só dela: é o que
# mantém as negativas proporcionais nas tres partições
marcas = {pid: set(rotulos) or {SEM_INDICADOR} for pid, rotulos in exemplos}

# Cota de cada partição: no total e em CADA classe
vagas        = {p: len(exemplos) * r for p, r in zip(PARTICOES, (.70, .15, .15))}
vagas_classe = {c: {p: de_classe[c] * r for p, r in zip(PARTICOES, (.70, .15, .15))}
                for c in de_classe}

while pendentes:
    # 1. A classe com MENOS exemplos ainda por distribuir vai primeiro:
    #    é assim que os poucos exemplos dela sobram para validação e teste
    classe = min(restam, key=lambda c: (restam[c], c))
    lote   = [pid for pid, m in pendentes.items() if classe in m]

    # 2. Cada notícia vai para a partição mais necessitada DAQUELA classe
    for pid in lote:
        # quem já recebeu a sua parte sai da disputa, senão o treino levaria
        # o resto: ele tem a maior cota em toda classe
        elegiveis = [p for p in PARTICOES if vagas[p] >= 1] or list(PARTICOES)
        destino = max(elegiveis, key=lambda p: (vagas_classe[classe][p],
                                                vagas[p], aleatorio.random()))
        vagas[destino] -= 1
        for marcada in pendentes.pop(pid):      # desconta de TODAS as classes
            vagas_classe[marcada][destino] -= 1 # da notícia, não só da atual

# 3. A partição de cada notícia é GRAVADA no banco: refazer o sorteio a cada
#    treino mudaria o teste e tornaria os modelos incomparáveis'''


def exemplos(con, tarefa: str = "indicadores", particao: str | None = None) -> list[dict]:
    """Devolve (texto, rótulo) para treino. Só entram as notícias anotadas —
    incluindo as de lista vazia, que são os exemplos negativos."""
    condicoes = [TREINAVEL]
    parametros = []
    if particao:
        condicoes.append("particao = ?")
        parametros.append(particao)

    linhas = con.execute(
        "SELECT id, texto, indicadores FROM corpus_noticia"
        f" WHERE {' AND '.join(condicoes)} ORDER BY id",
        parametros,
    ).fetchall()

    return [
        {
            "id": registro["id"],
            "texto": registro["texto"],
            "rotulo": registro["indicadores"],
        }
        for registro in (_para_dict(linha) for linha in linhas)
    ]


def rotulos_da_tarefa(tarefa: str = "indicadores",
                      classes: list[str] | None = None) -> list[str]:
    """As classes do modelo, nesta ordem. `classes` restringe a um subconjunto,
    mantendo a ordem do catálogo: a ordem define as posições do vetor
    multirrótulo e precisa ser estável entre o treino e a inferência."""
    todas = indicadores.nomes()
    if not classes:
        return todas
    escolhidas = set(classes)
    return [nome for nome in todas if nome in escolhidas]
