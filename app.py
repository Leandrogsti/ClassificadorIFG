"""Aplicação Flask do classificador de violência armada.

Fluxo: abre a ocorrência com data e endereço, cola as notícias, o sistema
classifica e consolida as vítimas, o analista confere, conclui e o registro
entra na fila de aprovação.
"""
import io
import json
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask, flash, g, redirect, render_template, request, send_file, session, url_for
)

import corpus
import db
import indicadores
import inferencia
import servico
import treinamento
import vocabulario
from classificador import ErroClassificacao

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "chave-de-desenvolvimento")

LOCAIS = json.loads(
    (Path(__file__).parent / "locais.json").read_text(encoding="utf-8")
)["cidades"]

ROTULOS_STATUS = {
    "rascunho": "Em edição",
    "nao_aprovado": "Não aprovado",
    "aprovado": "Aprovado",
    "unificada": "Unificada",
}


def conexao():
    if "con" not in g:
        g.con = db.conectar()
    return g.con


@app.teardown_appcontext
def fechar_conexao(_erro):
    con = g.pop("con", None)
    if con is not None:
        con.close()


@app.context_processor
def variaveis_globais():
    return {
        "rotulos_indicadores": indicadores.rotulos(),
        "rotulos_status": ROTULOS_STATUS,
        "analista": session.get("analista", ""),
        "circunstancias": vocabulario.CIRCUNSTANCIAS,
        "rotulos_circunstancia": vocabulario.ROTULOS_CIRCUNSTANCIA,
        "situacoes": vocabulario.SITUACOES,
        "generos": vocabulario.GENEROS,
        "tipos_vitima": vocabulario.TIPOS_VITIMA,
    }


def analista_atual() -> str:
    return session.get("analista") or "não identificado"


def buscar_ocorrencia(ocorrencia_id: int):
    linha = conexao().execute(
        "SELECT * FROM ocorrencia WHERE id = ?", (ocorrencia_id,)
    ).fetchone()
    if linha is None:
        return None
    ocorrencia = {k: linha[k] for k in linha.keys()}
    ocorrencia["motivos_complementares"] = db.json_carregar(
        linha["motivos_complementares"], []
    )
    ocorrencia["indicadores"] = db.json_carregar(linha["indicadores"], [])
    ocorrencia["indicadores_texto"] = db.json_carregar(linha["indicadores_texto"], [])
    ocorrencia["divergencias"] = db.json_carregar(linha["divergencias"], [])
    return ocorrencia


POR_PAGINA = 20


@app.route("/")
def inicio():
    con = conexao()
    status = request.args.get("status", "")
    de = request.args.get("de", "")
    ate = request.args.get("ate", "")

    filtros = ["o.unificada_em IS NULL"]
    parametros = []
    if status:
        filtros.append("o.status = ?")
        parametros.append(status)
    if de:
        filtros.append("o.data_fato >= ?")
        parametros.append(de)
    if ate:
        filtros.append("o.data_fato <= ?")
        parametros.append(ate)
    onde = " AND ".join(filtros)

    total = con.execute(
        f"SELECT COUNT(*) AS total FROM ocorrencia o WHERE {onde}", parametros
    ).fetchone()["total"]

    paginas = max(1, -(-total // POR_PAGINA))
    try:
        pagina = min(max(1, int(request.args.get("pagina", 1))), paginas)
    except ValueError:
        pagina = 1

    linhas = con.execute(
        "SELECT o.*, (SELECT COUNT(*) FROM vitima v WHERE v.ocorrencia_id = o.id)"
        " AS total_vitimas, (SELECT COUNT(*) FROM noticia n WHERE n.ocorrencia_id = o.id)"
        f" AS total_noticias FROM ocorrencia o WHERE {onde}"
        " ORDER BY o.data_fato DESC, o.id DESC LIMIT ? OFFSET ?",
        [*parametros, POR_PAGINA, (pagina - 1) * POR_PAGINA],
    ).fetchall()

    ocorrencias = []
    for linha in linhas:
        registro = {k: linha[k] for k in linha.keys()}
        registro["indicadores"] = db.json_carregar(linha["indicadores"], [])
        ocorrencias.append(registro)

    alertas_abertos = con.execute(
        "SELECT COUNT(*) AS total FROM alerta_duplicidade WHERE status = 'aberto'"
    ).fetchone()["total"]

    return render_template(
        "inicio.html",
        ocorrencias=ocorrencias,
        status_filtro=status,
        de=de,
        ate=ate,
        alertas_abertos=alertas_abertos,
        total=total,
        pagina=pagina,
        paginas=paginas,
    )


TABELAS_EXPORTADAS = [
    "ocorrencia", "noticia", "vitima_historico", "alerta_duplicidade",
]

# A planilha de vítimas carrega a localização da ocorrência para que dê para
# buscar por cidade, bairro e localidade sem precisar cruzar as abas na mão.
COLUNAS_PLANILHA_VITIMA = [
    ("id", "ID da vítima"),
    ("ocorrencia_id", "Ocorrência"),
    ("data_fato", "Data do fato"),
    ("cidade", "Cidade"),
    ("bairro", "Bairro"),
    ("localidade", "Localidade"),
    ("nome", "Nome"),
    ("idade", "Idade"),
    ("genero", "Gênero"),
    ("tipo_vitima", "Tipo de vítima"),
    ("situacao", "Situação"),
    ("data_morte", "Data da morte"),
    ("circunstancia", "Circunstância"),
    ("cargo_politico", "Cargo político"),
    ("fonte", "Fonte"),
    ("status_ocorrencia", "Status da ocorrência"),
]


def _aba_de_vitimas(planilha, vitimas: list[dict]) -> None:
    aba = planilha.create_sheet("vitima")
    aba.append([rotulo for _, rotulo in COLUNAS_PLANILHA_VITIMA])
    for vitima in vitimas:
        aba.append([
            vocabulario.ROTULOS_CIRCUNSTANCIA.get(vitima["circunstancia"], vitima["circunstancia"])
            if campo == "circunstancia"
            else vitima.get(campo)
            for campo, _ in COLUNAS_PLANILHA_VITIMA
        ])


def _enviar_planilha(planilha, nome: str):
    buffer = io.BytesIO()
    planilha.save(buffer)
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{nome}-{date.today().isoformat()}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/exportar")
def exportar():
    """Baixa a base inteira como uma planilha, uma aba por tabela."""
    from openpyxl import Workbook

    con = conexao()
    planilha = Workbook()
    planilha.remove(planilha.active)

    for tabela in TABELAS_EXPORTADAS:
        aba = planilha.create_sheet(tabela)
        colunas = [c[1] for c in con.execute(f"PRAGMA table_info({tabela})")]
        aba.append(colunas)
        for linha in con.execute(f"SELECT * FROM {tabela}"):
            aba.append([linha[coluna] for coluna in colunas])

    _aba_de_vitimas(planilha, servico.filtrar_vitimas(con, {}))
    return _enviar_planilha(planilha, "base-classificador")


def _filtros_de_vitima() -> dict:
    return {
        chave: request.args.get(chave, "").strip()
        for chave in ("de", "ate", "cidade", "bairro", "localidade", "circunstancia", "situacao")
    }


@app.route("/vitimas")
def vitimas():
    con = conexao()
    filtros = _filtros_de_vitima()
    total = servico.contar_vitimas(con, filtros)

    paginas = max(1, -(-total // POR_PAGINA))
    try:
        pagina = min(max(1, int(request.args.get("pagina", 1))), paginas)
    except ValueError:
        pagina = 1

    return render_template(
        "vitimas.html",
        vitimas=servico.filtrar_vitimas(con, filtros, POR_PAGINA, (pagina - 1) * POR_PAGINA),
        filtros=filtros,
        # Sem os vazios, para não sujar as URLs de paginação e download.
        filtros_ativos={chave: valor for chave, valor in filtros.items() if valor},
        total=total,
        pagina=pagina,
        paginas=paginas,
        locais=LOCAIS,
    )


@app.route("/vitimas/exportar")
def exportar_vitimas():
    """Planilha só de vítimas, respeitando a busca ativa na tela."""
    from openpyxl import Workbook

    planilha = Workbook()
    planilha.remove(planilha.active)
    _aba_de_vitimas(planilha, servico.filtrar_vitimas(conexao(), _filtros_de_vitima()))
    return _enviar_planilha(planilha, "vitimas")


@app.route("/analista", methods=["POST"])
def definir_analista():
    session["analista"] = request.form.get("analista", "").strip()
    return redirect(request.referrer or url_for("inicio"))


@app.route("/ocorrencia/nova", methods=["GET", "POST"])
def nova_ocorrencia():
    if request.method == "POST":
        nome = request.form.get("analista", "").strip()
        if nome:
            session["analista"] = nome
        con = conexao()
        momento = db.agora()
        with con:
            cursor = con.execute(
                "INSERT INTO ocorrencia (data_fato, cidade, bairro, localidade,"
                " analista, criado_em, atualizado_em, status)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 'rascunho')",
                (
                    request.form["data_fato"],
                    request.form["cidade"],
                    request.form["bairro"],
                    request.form.get("localidade", "").strip() or None,
                    analista_atual(),
                    momento,
                    momento,
                ),
            )
        return redirect(url_for("ocorrencia", ocorrencia_id=cursor.lastrowid))

    return render_template("nova_ocorrencia.html", locais=LOCAIS)


@app.route("/ocorrencia/<int:ocorrencia_id>")
def ocorrencia(ocorrencia_id: int):
    con = conexao()
    registro = buscar_ocorrencia(ocorrencia_id)
    if registro is None:
        flash("Ocorrência não encontrada.", "erro")
        return redirect(url_for("inicio"))

    noticias = con.execute(
        "SELECT * FROM noticia WHERE ocorrencia_id = ? ORDER BY id", (ocorrencia_id,)
    ).fetchall()
    vitimas = servico.listar_vitimas(con, ocorrencia_id)
    historico = con.execute(
        "SELECT * FROM vitima_historico WHERE ocorrencia_id = ?"
        " ORDER BY alterado_em DESC, id DESC",
        (ocorrencia_id,),
    ).fetchall()
    alertas = con.execute(
        "SELECT * FROM alerta_duplicidade WHERE status = 'aberto'"
        " AND (ocorrencia_id = ? OR ocorrencia_similar_id = ?)",
        (ocorrencia_id, ocorrencia_id),
    ).fetchall()

    return render_template(
        "ocorrencia.html",
        ocorrencia=registro,
        noticias=noticias,
        vitimas=vitimas,
        historico=historico,
        alertas=alertas,
    )


@app.route("/ocorrencia/<int:ocorrencia_id>/noticia", methods=["POST"])
def adicionar_noticia(ocorrencia_id: int):
    texto = request.form.get("texto", "").strip()
    if not texto:
        flash("O texto da notícia é obrigatório.", "erro")
        return redirect(url_for("ocorrencia", ocorrencia_id=ocorrencia_id))

    con = conexao()
    with con:
        servico.adicionar_noticia(
            con, ocorrencia_id, request.form.get("link", "").strip(), texto
        )
    try:
        with con:
            servico.reprocessar(con, ocorrencia_id)
            servico.marcar_para_reaprovacao(con, ocorrencia_id)
        flash(
            "Notícia adicionada. Todas as notícias foram relidas juntas e a lista"
            " de vítimas foi refeita — confira antes de concluir.",
            "ok",
        )
    except ErroClassificacao as erro:
        flash(f"A notícia foi salva, mas a classificação falhou: {erro}", "erro")

    return redirect(url_for("ocorrencia", ocorrencia_id=ocorrencia_id))


@app.route("/ocorrencia/<int:ocorrencia_id>/reprocessar", methods=["POST"])
def reprocessar(ocorrencia_id: int):
    con = conexao()
    try:
        with con:
            servico.reprocessar(con, ocorrencia_id)
        flash("Ocorrência reprocessada.", "ok")
    except ErroClassificacao as erro:
        flash(str(erro), "erro")
    return redirect(url_for("ocorrencia", ocorrencia_id=ocorrencia_id))


@app.route("/noticia/<int:noticia_id>/excluir", methods=["POST"])
def excluir_noticia(noticia_id: int):
    con = conexao()
    linha = con.execute(
        "SELECT ocorrencia_id FROM noticia WHERE id = ?", (noticia_id,)
    ).fetchone()
    ocorrencia_id = linha["ocorrencia_id"]
    with con:
        con.execute("DELETE FROM noticia WHERE id = ?", (noticia_id,))
    restantes = con.execute(
        "SELECT COUNT(*) AS total FROM noticia WHERE ocorrencia_id = ?", (ocorrencia_id,)
    ).fetchone()["total"]
    if restantes:
        try:
            with con:
                servico.reprocessar(con, ocorrencia_id)
        except ErroClassificacao as erro:
            flash(str(erro), "erro")
    else:
        with con:
            servico.limpar_classificacao(con, ocorrencia_id)
        flash("Última notícia removida: a classificação e as vítimas foram limpas.", "aviso")
    return redirect(url_for("ocorrencia", ocorrencia_id=ocorrencia_id))


@app.route("/vitima/<int:vitima_id>", methods=["POST"])
def editar_vitima(vitima_id: int):
    con = conexao()
    dados = {campo: request.form.get(campo) for campo in servico.CAMPOS_VITIMA}
    with con:
        ocorrencia_id = servico.atualizar_vitima(
            con, vitima_id, dados, analista_atual(), request.form.get("fonte_alteracao", "")
        )
    flash("Vítima atualizada. Os indicadores foram recalculados.", "ok")
    return redirect(url_for("ocorrencia", ocorrencia_id=ocorrencia_id))


@app.route("/vitima/<int:vitima_id>/proposta/<acao>", methods=["POST"])
def resolver_proposta(vitima_id: int, acao: str):
    con = conexao()
    with con:
        if acao == "aceitar":
            ocorrencia_id = servico.aceitar_proposta(con, vitima_id, analista_atual())
        else:
            ocorrencia_id = servico.descartar_proposta(con, vitima_id)
    return redirect(url_for("ocorrencia", ocorrencia_id=ocorrencia_id))


@app.route("/ocorrencia/<int:ocorrencia_id>/vitima", methods=["POST"])
def criar_vitima(ocorrencia_id: int):
    con = conexao()
    dados = {campo: request.form.get(campo) for campo in servico.CAMPOS_VITIMA}
    with con:
        servico.criar_vitima_manual(con, ocorrencia_id, dados, analista_atual())
    flash("Vítima incluída manualmente.", "ok")
    return redirect(url_for("ocorrencia", ocorrencia_id=ocorrencia_id))


@app.route("/vitima/<int:vitima_id>/excluir", methods=["POST"])
def excluir_vitima(vitima_id: int):
    con = conexao()
    with con:
        ocorrencia_id = servico.remover_vitima(con, vitima_id, analista_atual())
    return redirect(url_for("ocorrencia", ocorrencia_id=ocorrencia_id))


@app.route("/ocorrencia/<int:ocorrencia_id>/concluir", methods=["POST"])
def concluir(ocorrencia_id: int):
    con = conexao()
    total = con.execute(
        "SELECT COUNT(*) AS total FROM noticia WHERE ocorrencia_id = ?", (ocorrencia_id,)
    ).fetchone()["total"]
    if not total:
        flash("Inclua ao menos uma notícia antes de concluir.", "erro")
        return redirect(url_for("ocorrencia", ocorrencia_id=ocorrencia_id))

    with con:
        servico.recalcular_indicadores(con, ocorrencia_id)
        con.execute(
            "UPDATE ocorrencia SET status = 'nao_aprovado', atualizado_em = ?"
            " WHERE id = ?",
            (db.agora(), ocorrencia_id),
        )
        similares = servico.checar_duplicidade(con, ocorrencia_id)

    if similares:
        flash(
            f"Registro gravado. Atenção: {len(similares)} possível(is) duplicidade(s)"
            " no mesmo dia e bairro — verifique em Alertas.",
            "aviso",
        )
    else:
        flash("Registro gravado como não aprovado e enviado para revisão.", "ok")
    return redirect(url_for("ocorrencia", ocorrencia_id=ocorrencia_id))


@app.route("/aprovacao")
def aprovacao():
    con = conexao()
    linhas = con.execute(
        "SELECT o.*, (SELECT COUNT(*) FROM alerta_duplicidade a WHERE a.status = 'aberto'"
        " AND (a.ocorrencia_id = o.id OR a.ocorrencia_similar_id = o.id)) AS alertas"
        " FROM ocorrencia o WHERE o.status = 'nao_aprovado' AND o.unificada_em IS NULL"
        " ORDER BY o.atualizado_em"
    ).fetchall()
    registros = []
    for linha in linhas:
        registro = {k: linha[k] for k in linha.keys()}
        registro["indicadores"] = db.json_carregar(linha["indicadores"], [])
        registros.append(registro)
    return render_template("aprovacao.html", ocorrencias=registros)


@app.route("/ocorrencia/<int:ocorrencia_id>/aprovar", methods=["POST"])
def aprovar(ocorrencia_id: int):
    con = conexao()
    aberto = con.execute(
        "SELECT COUNT(*) AS total FROM alerta_duplicidade WHERE status = 'aberto'"
        " AND (ocorrencia_id = ? OR ocorrencia_similar_id = ?)",
        (ocorrencia_id, ocorrencia_id),
    ).fetchone()["total"]
    if aberto:
        flash(
            "Existe alerta de duplicidade aberto para este registro. Resolva o"
            " alerta antes de aprovar.",
            "erro",
        )
        return redirect(url_for("ocorrencia", ocorrencia_id=ocorrencia_id))

    with con:
        con.execute(
            "UPDATE ocorrencia SET status = 'aprovado', aprovado_por = ?,"
            " aprovado_em = ?, observacao_aprovacao = ?, atualizado_em = ? WHERE id = ?",
            (
                analista_atual(), db.agora(),
                request.form.get("observacao", "").strip() or None,
                db.agora(), ocorrencia_id,
            ),
        )
    flash("Registro aprovado e liberado para a comunicação.", "ok")
    return redirect(url_for("aprovacao"))


@app.route("/ocorrencia/<int:ocorrencia_id>/reprovar", methods=["POST"])
def reprovar(ocorrencia_id: int):
    con = conexao()
    with con:
        con.execute(
            "UPDATE ocorrencia SET status = 'nao_aprovado', aprovado_por = NULL,"
            " aprovado_em = NULL, observacao_aprovacao = ?, atualizado_em = ?"
            " WHERE id = ?",
            (
                request.form.get("observacao", "").strip() or None,
                db.agora(), ocorrencia_id,
            ),
        )
    flash("Registro devolvido ao analista para ajuste.", "aviso")
    return redirect(url_for("aprovacao"))


@app.route("/alertas")
def alertas():
    con = conexao()
    linhas = con.execute(
        "SELECT a.*, o1.data_fato, o1.cidade, o1.bairro,"
        " o1.analista AS analista_a, o2.analista AS analista_b"
        " FROM alerta_duplicidade a"
        " JOIN ocorrencia o1 ON o1.id = a.ocorrencia_id"
        " JOIN ocorrencia o2 ON o2.id = a.ocorrencia_similar_id"
        " ORDER BY a.status = 'aberto' DESC, a.criado_em DESC"
    ).fetchall()
    return render_template("alertas.html", alertas=linhas)


@app.route("/alerta/<int:alerta_id>/<acao>", methods=["POST"])
def resolver_alerta(alerta_id: int, acao: str):
    con = conexao()
    motivo = request.form.get("motivo", "").strip()
    try:
        with con:
            if acao == "unificar":
                destino = servico.unificar(con, alerta_id, analista_atual(), motivo)
                flash(
                    f"Registros unificados na ocorrência #{destino}. As notícias"
                    " passaram de novo pela LLM e o registro voltou para aprovação.",
                    "ok",
                )
            elif acao == "desfazer":
                servico.desfazer_unificacao(con, alerta_id, analista_atual())
                flash("Unificação desfeita. Os registros voltaram a ser separados.", "ok")
            else:
                servico.descartar_alerta(con, alerta_id, analista_atual(), motivo)
                flash("Alerta descartado: os registros seguem separados.", "ok")
    except ErroClassificacao as erro:
        flash(str(erro), "erro")
    return redirect(url_for("alertas"))


@app.route("/acompanhamento")
def acompanhamento():
    de = request.args.get("de", "").strip()
    ate = request.args.get("ate", "").strip()
    return render_template(
        "acompanhamento.html",
        vitimas=servico.vitimas_em_acompanhamento(conexao(), {"de": de, "ate": ate}),
        dias=servico.DIAS_ACOMPANHAMENTO,
        de=de,
        ate=ate,
        inicio_padrao=servico.inicio_do_acompanhamento(),
    )


@app.route("/gestao", methods=["GET", "POST"])
def gestao():
    if request.method == "POST":
        lista = indicadores.carregar()
        nome = request.form.get("nome", "").strip().lower().replace(" ", "_")
        if not nome:
            flash("Informe o nome (slug) do indicador.", "erro")
            return redirect(url_for("gestao"))

        novo = {
            "nome": nome,
            "rotulo": request.form.get("rotulo", "").strip() or nome,
            "tipo": request.form.get("tipo", "texto"),
            "definicao": request.form.get("definicao", "").strip(),
        }
        if novo["tipo"] == "texto":
            novo["skill"] = request.form.get("skill", "").strip()
            exemplos = request.form.get("exemplos", "").strip()
            novo["exemplos"] = [l.strip() for l in exemplos.splitlines() if l.strip()]
        else:
            novo["regra"] = {
                "operacao": request.form.get("operacao", "campo_igual"),
                "campo": request.form.get("campo", "").strip(),
                "valor": request.form.get("valor", "").strip(),
            }
            minimo = request.form.get("minimo", "").strip()
            if minimo.isdigit():
                novo["regra"]["minimo"] = int(minimo)
            if novo["regra"]["operacao"] == "campo_menor_que":
                novo["regra"]["valor"] = int(novo["regra"]["valor"] or 0)

        lista = [i for i in lista if i["nome"] != nome] + [novo]
        indicadores.salvar(lista)
        aviso = (
            " Indicadores de texto novos também exigem exemplos rotulados para"
            " retreinar o BERTimbau."
            if novo["tipo"] == "texto"
            else ""
        )
        flash(f"Indicador '{nome}' salvo.{aviso}", "ok")
        return redirect(url_for("gestao"))

    return render_template("gestao.html", indicadores=indicadores.carregar())


@app.route("/gestao/<nome>/excluir", methods=["POST"])
def excluir_indicador(nome: str):
    indicadores.salvar([i for i in indicadores.carregar() if i["nome"] != nome])
    flash(f"Indicador '{nome}' removido.", "ok")
    return redirect(url_for("gestao"))


@app.route("/bairros/<cidade>")
def bairros(cidade: str):
    return {"bairros": LOCAIS.get(cidade, [])}


# --------------------------------------------------------------------------
# Corpus de treino, treino dos modelos e testes antes da camada de LLM
# --------------------------------------------------------------------------

def _filtros_de_corpus() -> dict:
    return {
        chave: request.args.get(chave, "").strip()
        for chave in ("violencia_armada", "motivo_principal", "busca", "particao")
    }


@app.route("/corpus")
def corpus_repositorio():
    con = conexao()
    filtros = _filtros_de_corpus()
    total = corpus.contar(con, filtros)

    paginas = max(1, -(-total // POR_PAGINA))
    try:
        pagina = min(max(1, int(request.args.get("pagina", 1))), paginas)
    except ValueError:
        pagina = 1

    return render_template(
        "corpus.html",
        noticias=corpus.listar(con, filtros, POR_PAGINA, (pagina - 1) * POR_PAGINA),
        filtros=filtros,
        filtros_ativos={c: v for c, v in filtros.items() if v},
        estatisticas=corpus.estatisticas(con),
        motivacoes=vocabulario.motivacoes(),
        indicadores_texto=[i for i in indicadores.carregar() if i["tipo"] == "texto"],
        total=total,
        pagina=pagina,
        paginas=paginas,
    )


@app.route("/corpus/nova", methods=["POST"])
def corpus_adicionar():
    texto = request.form.get("texto", "").strip()
    if not texto:
        flash("O texto da notícia é obrigatório.", "erro")
        return redirect(url_for("corpus_repositorio"))

    con = conexao()
    dados = {
        "link": request.form.get("link"),
        "titulo": request.form.get("titulo"),
        "texto": texto,
        "violencia_armada": request.form.get("violencia_armada"),
        "motivo_principal": request.form.get("motivo_principal"),
        "indicadores": request.form.getlist("indicadores"),
        "observacao": request.form.get("observacao"),
    }
    with con:
        novo = corpus.adicionar(con, dados, analista_atual())
    flash(f"Notícia #{novo} incluída no corpus.", "ok")

    if request.form.get("continuar"):
        return redirect(url_for("corpus_repositorio", _anchor="nova"))
    return redirect(url_for("corpus_noticia", noticia_id=novo))


@app.route("/corpus/<int:noticia_id>", methods=["GET", "POST"])
def corpus_noticia(noticia_id: int):
    con = conexao()

    if request.method == "POST":
        dados = {
            "link": request.form.get("link"),
            "titulo": request.form.get("titulo"),
            "texto": request.form.get("texto", "").strip(),
            "violencia_armada": request.form.get("violencia_armada"),
            "motivo_principal": request.form.get("motivo_principal"),
            "indicadores": request.form.getlist("indicadores"),
            "observacao": request.form.get("observacao"),
        }
        if not dados["texto"]:
            flash("O texto da notícia é obrigatório.", "erro")
            return redirect(url_for("corpus_noticia", noticia_id=noticia_id))
        with con:
            corpus.atualizar(con, noticia_id, dados, analista_atual())
        flash("Rótulos salvos.", "ok")
        proxima = request.form.get("proxima")
        if proxima:
            return redirect(url_for("corpus_anotar"))
        return redirect(url_for("corpus_noticia", noticia_id=noticia_id))

    registro = corpus.buscar(con, noticia_id)
    if registro is None:
        flash("Notícia não encontrada no corpus.", "erro")
        return redirect(url_for("corpus_repositorio"))

    return render_template(
        "corpus_noticia.html",
        noticia=registro,
        motivacoes=vocabulario.motivacoes(),
        indicadores_texto=[i for i in indicadores.carregar() if i["tipo"] == "texto"],
        restantes=corpus.contar(con, {"violencia_armada": "nao_rotulado"}),
    )


@app.route("/corpus/anotar")
def corpus_anotar():
    """Fila de anotação: entrega a próxima notícia sem rótulo."""
    con = conexao()
    pendentes = corpus.listar(con, {"violencia_armada": "nao_rotulado"}, 1)
    if not pendentes:
        flash("Nenhuma notícia sem rótulo. O corpus está todo anotado.", "ok")
        return redirect(url_for("corpus_repositorio"))
    return redirect(url_for("corpus_noticia", noticia_id=pendentes[0]["id"]))


@app.route("/corpus/<int:noticia_id>/excluir", methods=["POST"])
def corpus_excluir(noticia_id: int):
    con = conexao()
    with con:
        corpus.remover(con, noticia_id)
    flash(f"Notícia #{noticia_id} removida do corpus.", "ok")
    return redirect(url_for("corpus_repositorio"))


@app.route("/corpus/dividir", methods=["POST"])
def corpus_dividir():
    con = conexao()
    try:
        semente = int(request.form.get("semente", 42))
    except ValueError:
        semente = 42
    with con:
        contagem = corpus.dividir(con, semente=semente)
    flash(
        f"Divisão estratificada com semente {semente}: {contagem['treino']} treino,"
        f" {contagem['validacao']} validação, {contagem['teste']} teste.",
        "ok",
    )
    return redirect(url_for("corpus_repositorio"))


COLUNAS_CORPUS = [
    ("id", "ID"), ("link", "Link"), ("titulo", "Título"), ("texto", "Texto"),
    ("violencia_armada", "Violência armada"), ("motivo_principal", "Motivação"),
    ("indicadores", "Indicadores"), ("particao", "Partição"),
    ("observacao", "Observação"), ("anotador", "Anotador"), ("criado_em", "Criado em"),
]


@app.route("/corpus/exportar")
def corpus_exportar():
    """O corpus precisa sair do banco para ser anexado ao TCC e reproduzido."""
    from openpyxl import Workbook

    planilha = Workbook()
    planilha.remove(planilha.active)
    aba = planilha.create_sheet("corpus")
    aba.append([rotulo for _, rotulo in COLUNAS_CORPUS])
    for registro in corpus.listar(conexao(), _filtros_de_corpus()):
        aba.append([
            ", ".join(registro["indicadores"]) if campo == "indicadores"
            else registro.get(campo)
            for campo, _ in COLUNAS_CORPUS
        ])
    return _enviar_planilha(planilha, "corpus-treino")


@app.route("/corpus/importar", methods=["POST"])
def corpus_importar():
    """Importa um lote de notícias de planilha, para quando o corpus já existe
    fora da aplicação. Colunas aceitas: link, titulo, texto, violencia_armada,
    motivo_principal, indicadores."""
    from openpyxl import load_workbook

    arquivo = request.files.get("arquivo")
    if not arquivo or not arquivo.filename:
        flash("Selecione uma planilha .xlsx para importar.", "erro")
        return redirect(url_for("corpus_repositorio"))

    try:
        planilha = load_workbook(arquivo, read_only=True, data_only=True)
    except Exception as erro:
        flash(f"Não foi possível ler a planilha: {erro}", "erro")
        return redirect(url_for("corpus_repositorio"))

    aba = planilha[planilha.sheetnames[0]]
    linhas = aba.iter_rows(values_only=True)
    cabecalho = [str(c or "").strip().lower() for c in next(linhas, [])]
    if "texto" not in cabecalho:
        flash("A planilha precisa ter uma coluna chamada 'texto'.", "erro")
        return redirect(url_for("corpus_repositorio"))

    con = conexao()
    importadas = ignoradas = 0
    with con:
        for linha in linhas:
            registro = dict(zip(cabecalho, linha))
            texto = str(registro.get("texto") or "").strip()
            if not texto:
                ignoradas += 1
                continue
            bruto = registro.get("indicadores") or ""
            corpus.adicionar(con, {
                "link": registro.get("link"),
                "titulo": registro.get("titulo"),
                "texto": texto,
                "violencia_armada": registro.get("violencia_armada"),
                "motivo_principal": registro.get("motivo_principal"),
                "indicadores": [p.strip() for p in str(bruto).split(",") if p.strip()],
                "observacao": registro.get("observacao"),
            }, analista_atual())
            importadas += 1

    flash(
        f"{importadas} notícia(s) importada(s)."
        + (f" {ignoradas} linha(s) sem texto ignorada(s)." if ignoradas else ""),
        "ok",
    )
    return redirect(url_for("corpus_repositorio"))


@app.route("/modelos")
def modelos():
    """Resultados do treino: a tabela comparativa e as matrizes de confusão."""
    con = conexao()
    execucoes = {}
    for tarefa in corpus.TAREFAS:
        linhas = con.execute(
            "SELECT * FROM treino_execucao WHERE tarefa = ? ORDER BY id DESC",
            (tarefa,),
        ).fetchall()
        registros = []
        for linha in linhas:
            registro = {k: linha[k] for k in linha.keys()}
            registro["hiperparametros"] = db.json_carregar(linha["hiperparametros"], {})
            registro["metricas"] = db.json_carregar(linha["metricas"], {})
            registro["rotulos"] = db.json_carregar(linha["rotulos"], [])
            registro["historico"] = db.json_carregar(linha["historico"], [])
            registros.append(registro)
        execucoes[tarefa] = registros

    return render_template(
        "modelos.html",
        execucoes=execucoes,
        tarefas=corpus.TAREFAS,
        catalogo=treinamento.MODELOS,
        grade=treinamento.GRADE,
        ambiente=treinamento.descrever_ambiente(),
        estatisticas=corpus.estatisticas(con),
        treinados={t: inferencia.treinado(t) for t in corpus.TAREFAS},
    )


@app.route("/modelos/<int:execucao_id>")
def modelo_execucao(execucao_id: int):
    linha = conexao().execute(
        "SELECT * FROM treino_execucao WHERE id = ?", (execucao_id,)
    ).fetchone()
    if linha is None:
        flash("Execução não encontrada.", "erro")
        return redirect(url_for("modelos"))

    execucao = {k: linha[k] for k in linha.keys()}
    execucao["hiperparametros"] = db.json_carregar(linha["hiperparametros"], {})
    execucao["metricas"] = db.json_carregar(linha["metricas"], {})
    execucao["rotulos"] = db.json_carregar(linha["rotulos"], [])
    execucao["historico"] = db.json_carregar(linha["historico"], [])
    return render_template(
        "modelo_execucao.html", execucao=execucao, catalogo=treinamento.MODELOS
    )


@app.route("/testar", methods=["GET", "POST"])
def testar():
    """Playground: o que o BERT responde, antes de qualquer LLM."""
    etapa = request.args.get("etapa", "violencia")
    texto = request.form.get("texto", "") if request.method == "POST" else ""
    resultado = erro = None

    if request.method == "POST" and texto.strip():
        try:
            if etapa == "pipeline":
                resultado = inferencia.pipeline(texto)
            else:
                resultado = inferencia.prever(etapa, texto)
        except RuntimeError as falha:
            erro = str(falha)

    # O pipeline só precisa da etapa 1 para dar uma resposta útil: se a notícia
    # não é violência armada, as outras nem rodam.
    necessarias = ["violencia"] if etapa == "pipeline" else [etapa]

    return render_template(
        "testar.html",
        etapa=etapa,
        texto=texto,
        resultado=resultado,
        erro=erro,
        tarefas=corpus.TAREFAS,
        treinados={t: inferencia.treinado(t) for t in corpus.TAREFAS},
        faltando=[t for t in necessarias if not inferencia.treinado(t)],
        motivacoes={c["nome"]: c for c in vocabulario.motivacoes()},
    )


@app.route("/testar/salvar", methods=["POST"])
def testar_salvar():
    """Leva um texto do playground para o corpus. É assim que um erro visto no
    teste vira exemplo de treino em vez de só uma anotação perdida."""
    texto = request.form.get("texto", "").strip()
    if not texto:
        flash("Nada para salvar.", "erro")
        return redirect(url_for("testar"))

    con = conexao()
    with con:
        novo = corpus.adicionar(con, {
            "link": request.form.get("link"),
            "texto": texto,
            "observacao": "Enviada do playground de testes",
        }, analista_atual())
    flash(f"Notícia #{novo} enviada ao corpus. Agora defina os rótulos.", "ok")
    return redirect(url_for("corpus_noticia", noticia_id=novo))


if __name__ == "__main__":
    db.iniciar()
    app.run(debug=True, port=5000)
