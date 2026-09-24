"""Aplicação Flask do classificador de violência armada.

Fluxo: abre a ocorrência com data e endereço, cola as notícias, o sistema
classifica e consolida as vítimas, o analista confere, conclui e o registro
entra na fila de aprovação.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask, flash, g, redirect, render_template, request, session, url_for
)

import db
import indicadores
import servico
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


@app.route("/")
def inicio():
    con = conexao()
    status = request.args.get("status", "")
    consulta = (
        "SELECT o.*, (SELECT COUNT(*) FROM vitima v WHERE v.ocorrencia_id = o.id)"
        " AS total_vitimas, (SELECT COUNT(*) FROM noticia n WHERE n.ocorrencia_id = o.id)"
        " AS total_noticias FROM ocorrencia o WHERE o.unificada_em IS NULL"
    )
    parametros = []
    if status:
        consulta += " AND o.status = ?"
        parametros.append(status)
    consulta += " ORDER BY o.data_fato DESC, o.id DESC"

    ocorrencias = []
    for linha in con.execute(consulta, parametros).fetchall():
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
        alertas_abertos=alertas_abertos,
    )


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
    dados["bala_perdida"] = request.form.get("bala_perdida", "0")
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
    return render_template(
        "acompanhamento.html",
        vitimas=servico.vitimas_em_acompanhamento(conexao()),
        dias=servico.DIAS_ACOMPANHAMENTO,
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


if __name__ == "__main__":
    db.iniciar()
    app.run(debug=True, port=5000)
