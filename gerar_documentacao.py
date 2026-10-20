"""Gera a documentação de TCC em .docx a partir das fontes reais do projeto.

POR QUE GERAR EM VEZ DE ESCREVER. As tabelas de apêndice saem de
`indicadores_fonte.json` e do catálogo de modelos de `treinamento.py`, não de
texto copiado à mão. Alterar um indicador e regerar mantém o documento em dia
com o sistema que ele descreve, em vez de deixar os dois divergirem em silêncio
— que é o destino de todo anexo copiado.

    python gerar_documentacao.py

ESTRUTURA DO DOCUMENTO: resumo, introdução, justificativa, objetivos,
fundamentação, método (corpus, anotação, treino, avaliação), resultados
esperados, cronograma e apêndices. As funções de formatação (`sombrear`,
`legenda`, `tabela`, `quebra`, `sumario`) existem porque python-docx não tem
estilos de alto nível: cada tabela da ABNT é montada célula por célula.

O QUE O DOCUMENTO DESCREVE, para não divergir do código: o sistema classifica
FENÔMENO OBSERVADO da ocorrência e CIRCUNSTÂNCIA da vítima. Não há atribuição de
motivação — ela saiu do sistema em 2026-10-06, e o texto deste arquivo foi
revisado para refletir isso. Se você acrescentar uma seção aqui, confira contra
`indicadores_fonte.json` e `prompts.py`, que são o que de fato roda.
"""
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PROJETO = Path(__file__).parent
SAIDA = PROJETO / "docs"
SAIDA.mkdir(exist_ok=True)

import corpus as mod_corpus
import treinamento as mod_treino
import indicadores as mod_ind

INDICADORES = json.loads((PROJETO / "indicadores_fonte.json").read_text(encoding="utf-8"))["indicadores"]
# Só os que o classificador aprende do texto. Os demais do catálogo são
# derivados de regra sobre a lista de vítimas (vitimização policial, chacina) ou
# avaliados pelo modelo de linguagem (violência de gênero), e por isso não
# entram na contagem de classes do treino.
INDICADORES_BERT = [i for i in INDICADORES if i.get("decidido_por") == "bert"]

CINZA = RGBColor(0x44, 0x44, 0x44)

doc = Document()

# ---------------------------------------------------------------- formatacao
secao = doc.sections[0]
secao.page_width = Cm(21.0)
secao.page_height = Cm(29.7)
secao.left_margin = Cm(3)
secao.top_margin = Cm(3)
secao.right_margin = Cm(2)
secao.bottom_margin = Cm(2)

def fixar_fonte(estilo, nome="Times New Roman"):
    """Os estilos internos do Word apontam para a fonte do TEMA, e o atributo de
    tema vence o w:ascii que o python-docx escreve. Sem remover o tema, os
    titulos saem na fonte sem serifa do tema padrao."""
    rpr = estilo.element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    for atributo in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        chave = qn("w:" + atributo)
        if chave in rfonts.attrib:
            del rfonts.attrib[chave]
    for atributo in ("ascii", "hAnsi", "cs", "eastAsia"):
        rfonts.set(qn("w:" + atributo), nome)


normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
fixar_fonte(normal)
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_after = Pt(0)
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

for nome_estilo in ("List Bullet", "Caption"):
    try:
        fixar_fonte(doc.styles[nome_estilo])
    except KeyError:
        pass

for nivel, tamanho in [(1, 14), (2, 13), (3, 12)]:
    estilo = doc.styles[f"Heading {nivel}"]
    estilo.font.name = "Times New Roman"
    fixar_fonte(estilo)
    estilo.font.size = Pt(tamanho)
    estilo.font.bold = True
    estilo.font.color.rgb = RGBColor(0, 0, 0)
    estilo.paragraph_format.space_before = Pt(18)
    estilo.paragraph_format.space_after = Pt(10)
    estilo.paragraph_format.line_spacing = 1.5
    estilo.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    estilo.paragraph_format.keep_with_next = True


def p(texto="", recuo=True, centro=False, italico=False, negrito=False, tamanho=None, espaco=6):
    par = doc.add_paragraph()
    pf = par.paragraph_format
    if centro:
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif recuo:
        pf.first_line_indent = Cm(1.25)
    pf.space_after = Pt(espaco)
    if texto:
        run = par.add_run(texto)
        run.italic = italico
        run.bold = negrito
        if tamanho:
            run.font.size = Pt(tamanho)
    return par


def h(nivel, texto):
    return doc.add_heading(texto, level=nivel)


def bullet(texto, negrito_ate=None):
    par = doc.add_paragraph(style="List Bullet")
    par.paragraph_format.line_spacing = 1.5
    par.paragraph_format.space_after = Pt(4)
    par.paragraph_format.left_indent = Cm(1.25)
    if negrito_ate:
        par.add_run(negrito_ate).bold = True
        par.add_run(texto)
    else:
        par.add_run(texto)
    return par


def codigo(linhas):
    par = doc.add_paragraph()
    pf = par.paragraph_format
    pf.left_indent = Cm(1.25)
    pf.space_before = Pt(6)
    pf.space_after = Pt(10)
    pf.line_spacing = 1.0
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for i, linha in enumerate(linhas):
        if i:
            par.add_run().add_break()
        run = par.add_run(linha)
        run.font.name = "Consolas"
        run.font.size = Pt(10)
    sombrear(par, "F2F3F5")
    return par


def sombrear(par_ou_celula, cor_hex):
    elemento = par_ou_celula._p if hasattr(par_ou_celula, "_p") else par_ou_celula._tc
    pr = elemento.get_or_add_pPr() if hasattr(par_ou_celula, "_p") else elemento.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), cor_hex)
    pr.append(shd)


def legenda(texto, acima=False):
    par = doc.add_paragraph()
    par.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    par.paragraph_format.space_before = Pt(10 if acima else 2)
    par.paragraph_format.space_after = Pt(4 if acima else 12)
    par.paragraph_format.line_spacing = 1.0
    run = par.add_run(texto)
    run.font.size = Pt(10)
    run.font.color.rgb = CINZA
    return par


def tabela(cabecalho, linhas, larguras_cm):
    tab = doc.add_table(rows=1, cols=len(cabecalho))
    tab.style = "Table Grid"
    tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    tab.autofit = False

    def preencher(celula, texto, largura, negrito=False):
        celula.width = Cm(largura)
        celula.text = ""
        par = celula.paragraphs[0]
        pf = par.paragraph_format
        pf.line_spacing = 1.0
        pf.space_after = Pt(2)
        pf.space_before = Pt(2)
        # Celula estreita com texto justificado abre vaos enormes entre palavras.
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = par.add_run(str(texto))
        run.bold = negrito
        run.font.size = Pt(10)

    for i, (titulo, largura) in enumerate(zip(cabecalho, larguras_cm)):
        preencher(tab.rows[0].cells[i], titulo, largura, negrito=True)
        sombrear(tab.rows[0].cells[i], "E6EAF0")

    # Repete o cabecalho quando a tabela atravessa paginas.
    tr_pr = tab.rows[0]._tr.get_or_add_trPr()
    repetir = OxmlElement("w:tblHeader")
    repetir.set(qn("w:val"), "true")
    tr_pr.append(repetir)

    for valores in linhas:
        celulas = tab.add_row().cells
        for i, valor in enumerate(valores):
            preencher(celulas[i], valor, larguras_cm[i])
    return tab


def quebra():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def sumario():
    par = doc.add_paragraph()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), r'TOC \o "1-3" \h \z \u')
    filho = OxmlElement("w:r")
    texto = OxmlElement("w:t")
    texto.text = "Atualize o sumário no Word: clique com o botão direito e escolha “Atualizar campo”."
    filho.append(texto)
    fld.append(filho)
    par._p.append(fld)


# ---------------------------------------------------------------------- capa
for _ in range(2):
    p(centro=True)
p("INSTITUTO FEDERAL DE GOIÁS", centro=True, recuo=False, negrito=True, espaco=0)
p("TRABALHO DE CONCLUSÃO DE CURSO", centro=True, recuo=False, espaco=0)
for _ in range(4):
    p(centro=True)
p("CLASSIFICAÇÃO AUTOMÁTICA DE NOTÍCIAS DE VIOLÊNCIA ARMADA",
  centro=True, recuo=False, negrito=True, tamanho=16, espaco=0)
p("COM MODELOS BERT EM PORTUGUÊS E VALIDAÇÃO POR MODELO DE LINGUAGEM",
  centro=True, recuo=False, negrito=True, tamanho=16, espaco=0)
for _ in range(2):
    p(centro=True)
p("Documentação de método, implementação e avaliação",
  centro=True, recuo=False, italico=True, espaco=0)
for _ in range(8):
    p(centro=True)
p("Goiânia", centro=True, recuo=False, espaco=0)
p("2026", centro=True, recuo=False, espaco=0)

quebra()

# -------------------------------------------------------------------- resumo
h(1, "Resumo")
p("Este documento descreve o método, a implementação e o protocolo de avaliação "
  "de um sistema que classifica notícias de violência armada em indicadores, "
  "substituindo a triagem manual que hoje limita o volume de notícias analisadas "
  "e a velocidade de atualização dos indicadores. A classificação é uma tarefa "
  f"única e multirrótulo: dado o texto, quais dos {len(INDICADORES_BERT)} "
  "indicadores a notícia sustenta — podem ser vários ao mesmo tempo, ou nenhum. "
  "Não há etapa binária de triagem antes dela: a notícia fora do escopo é "
  "justamente a que não recebe indicador algum, e esses exemplos negativos são "
  "parte do corpus, não descarte. "
  f"São comparados {len(mod_treino.MODELOS)} modelos pré-treinados monolíngues "
  "de português sob uma grade de "
  f"{len(mod_treino.GRADE)} configurações de hiperparâmetros, em "
  f"{len(mod_treino.ESTRATEGIAS)} estratégias de ajuste, com relato de "
  "acurácia, precisão, recall e medida F1, em versões macro, ponderada e por "
  "classe, acompanhadas da matriz de confusão. Um modelo de linguagem de grande "
  "porte opera APÓS a marcação e condicionado a ela: ele recebe somente as "
  "instruções das classes marcadas, e a sua tarefa é aprofundá-las — montar a "
  "lista de vítimas, preencher a ficha de cada uma e as tabelas complementares. "
  "O documento detalha a construção do "
  "corpus anotado manualmente, o protocolo de divisão amostral, os critérios de "
  "seleção de hiperparâmetros e as ameaças à validade dos resultados.")
p()
p("Palavras-chave: processamento de linguagem natural; BERT; BERTimbau; "
  "classificação de texto; violência armada; indicadores de segurança pública.",
  recuo=False)

quebra()

# ------------------------------------------------------------------- sumario
h(1, "Sumário")
sumario()

quebra()

# --------------------------------------------------------------- introducao
h(1, "1 Introdução")

h(2, "1.1 Contextualização")
p("Instituições que monitoram violência armada constroem seus indicadores a "
  "partir da leitura de notícias. Um analista lê cada texto, decide se o caso "
  "pertence ao escopo, classifica a dinâmica do episódio e registra as vítimas. "
  "O procedimento é confiável quando conduzido por pessoas treinadas, mas o seu "
  "custo cresce linearmente com o volume de notícias, e é esse custo que fixa o "
  "teto do que se consegue monitorar.")
p("A consequência não é apenas de produtividade. Quando a triagem não acompanha "
  "o fluxo de publicações, os indicadores passam a descrever um recorte parcial "
  "e defasado da realidade, o que compromete o seu uso em decisões de política "
  "pública e em comunicação pública.")

h(2, "1.2 Problema")
p("O problema tratado é o da triagem e classificação automáticas de notícias "
  "para alimentar indicadores de violência armada, preservando a qualidade da "
  "classificação manual. A dificuldade está em três pontos. Primeiro, a decisão "
  "de escopo é sutil: muitas notícias mencionam armas de fogo sem relatar "
  "disparo, e o recorte do banco é estrito — registra-se disparo de arma de "
  "fogo, e vítima é apenas quem foi atingido por disparo, não quem se feriu em "
  "queda, correria, estilhaço ou mal-estar durante o fato. Segundo, os "
  "indicadores não são mutuamente evidentes a partir de palavras isoladas, pois "
  "dependem da relação entre quem atirou, contra quem e em que circunstância: a "
  "mesma expressão aparece em notícias que recebem indicadores diferentes. "
  "Terceiro, a contagem de vítimas exige consolidar "
  "fontes diferentes sobre o mesmo fato sem somá-las indevidamente.")

h(2, "1.3 Objetivos")
p("O objetivo geral é projetar, implementar e avaliar um sistema de "
  "classificação automática de notícias de violência armada em indicadores, no "
  "qual um classificador ajustado decide QUAIS indicadores a notícia sustenta e "
  "um modelo de linguagem de grande porte aprofunda apenas os indicadores "
  "marcados, extraindo deles a lista de vítimas e os campos estruturados.")
p("Os objetivos específicos são:", recuo=False)
bullet("construir um corpus de notícias anotado manualmente, com registro da "
       "fonte de cada exemplo, e disponibilizar ferramenta própria de anotação;")
bullet("comparar modelos pré-treinados monolíngues de português na atribuição "
       f"multirrótulo dos {len(INDICADORES_BERT)} indicadores;")
bullet(f"comparar as {len(mod_treino.ESTRATEGIAS)} estratégias de ajuste — uma "
       "rede com uma saída por indicador, uma rede binária por indicador, e um "
       "comitê de três modelos por indicador com decisão por votação — na mesma "
       "partição de teste;")
bullet("conduzir busca de hiperparâmetros e relatar o seu efeito sobre o "
       "desempenho;")
bullet("relatar acurácia, precisão, recall, medida F1 e matriz de confusão para "
       "cada combinação de modelo e configuração;")
bullet("avaliar o ganho da camada de modelo de linguagem sobre a saída dos "
       "classificadores.")

h(2, "1.4 Justificativa")
p("A divisão de trabalho entre o classificador e o modelo de linguagem não é um "
  "detalhe de implementação, e sim a resposta a uma restrição de custo medida em "
  "produção. Submeter toda notícia coletada a um modelo de linguagem de grande "
  "porte é caro e lento, e o limite não é teórico: o plano contratado concede "
  "8.000 tokens por minuto, enquanto uma chamada que carrega as instruções de "
  "todos os indicadores consome cerca de 6.800 — uma classificação por minuto. "
  "Com a marcação prévia, apenas as instruções das classes marcadas viajam, e a "
  "mesma chamada cai para cerca de 4.200 tokens.")
p("A folga assim obtida não é só economia. Ela é o que permite elevar o esforço "
  "de raciocínio do modelo de linguagem, e esse parâmetro mostrou-se decisivo "
  "para a estabilidade da extração: com esforço baixo, a mesma notícia submetida "
  "oito vezes produziu contagens de vítimas divergentes em quatro delas, "
  "inclusive uma lista vazia para um texto que descreve duas pessoas atingidas; "
  "com esforço médio, as cinco execuções foram concordantes. O classificador, "
  "portanto, não apenas reduz o custo: ele compra a margem de que a etapa "
  "seguinte precisa para ser confiável.")
p("O desenho também torna o sistema auditável. A marcação é registrada com a "
  "confiança de cada classe antes de o modelo de linguagem ver o texto, de modo "
  "que um erro do classificador corrigido adiante não desaparece do registro. É "
  "esse registro que permite dizer se um erro veio da marcação ou da extração, e "
  "é dele que sai o exemplo para o treino seguinte.")

quebra()

# --------------------------------------------------------- fundamentacao
h(1, "2 Fundamentação teórica")

h(2, "2.1 Representações contextuais e a arquitetura Transformer")
p("Modelos de linguagem baseados na arquitetura Transformer representam cada "
  "palavra em função do contexto em que ela aparece, por meio do mecanismo de "
  "autoatenção. A diferença em relação a representações estáticas é decisiva "
  "para o problema aqui tratado: em “a polícia apreendeu a arma” e “a polícia "
  "atirou com a arma”, a palavra “arma” ocupa papéis distintos, e só uma "
  "representação sensível ao contexto distingue os dois casos.")

h(2, "2.2 BERT e o pré-treinamento")
p("O BERT é treinado em duas tarefas não supervisionadas sobre grandes volumes "
  "de texto, entre elas a predição de palavras mascaradas, o que o obriga a "
  "modelar dependências em ambas as direções da sentença. O resultado é um "
  "conjunto de pesos que já codifica regularidades sintáticas e semânticas da "
  "língua e que pode ser especializado para uma tarefa específica com uma "
  "quantidade de exemplos anotados muito menor do que a exigida por um modelo "
  "treinado a partir do zero.")

h(2, "2.3 Modelos para português")
p("Modelos multilíngues cobrem dezenas de idiomas com um único vocabulário de "
  "subpalavras, o que reduz a capacidade dedicada a cada língua. Modelos "
  "monolíngues de português, treinados sobre corpora da língua, tendem a "
  "segmentar melhor o vocabulário e a capturar construções próprias do "
  "português brasileiro, incluindo o registro jornalístico e os termos do campo "
  "da segurança pública.")
p("Este trabalho adota o pré-treinamento monolíngue como PREMISSA, e não como "
  "hipótese a testar: a comparação restringe-se a modelos de português, e "
  "nenhum modelo multilíngue é submetido. A decisão é de recorte, e tem "
  "consequência sobre o que se pode afirmar ao final — os resultados permitem "
  "dizer qual modelo de português se comporta melhor nesta tarefa, e não se o "
  "pré-treinamento em português é superior ao multilíngue. Verificar essa "
  "superioridade exigiria incluir ao menos um modelo multilíngue, e fica "
  "registrada entre os trabalhos futuros.")

h(2, "2.4 Ajuste fino")
p("O ajuste fino acrescenta ao modelo pré-treinado uma camada de classificação "
  "inicializada aleatoriamente e atualiza todos os pesos com os exemplos "
  "anotados da tarefa. Corpora pequenos tornam o procedimento sensível: taxas de "
  "aprendizado elevadas destroem o conhecimento adquirido no pré-treinamento, e "
  "um número grande de épocas leva o modelo a memorizar o conjunto de treino em "
  "vez de generalizar. É por isso que a busca de hiperparâmetros descrita na "
  "seção 8 trabalha em uma faixa estreita e deliberadamente conservadora.")

h(2, "2.5 Métricas de classificação")
p("Seja, para uma classe, VP o número de verdadeiros positivos, FP o de falsos "
  "positivos, VN o de verdadeiros negativos e FN o de falsos negativos. "
  "Definem-se:")
p()
tabela(
    ["Métrica", "Definição", "O que responde"],
    [
        ["Acurácia", "(VP + VN) / (VP + VN + FP + FN)",
         "Que fração de todas as decisões está correta."],
        ["Precisão", "VP / (VP + FP)",
         "Entre os casos apontados como da classe, quantos realmente são."],
        ["Recall", "VP / (VP + FN)",
         "Entre os casos que são da classe, quantos o modelo encontrou."],
        ["Medida F1", "2 · (precisão · recall) / (precisão + recall)",
         "Média harmônica entre precisão e recall, penalizando desequilíbrio."],
    ],
    [2.6, 6.2, 7.2],
)
legenda("Tabela 1 — Métricas de classificação adotadas.")

p("A escolha da forma de agregação entre classes é metodologicamente relevante. "
  "A média macro calcula a métrica para cada classe e tira a média simples, "
  "atribuindo o mesmo peso a todas; a média ponderada pesa cada classe pelo seu "
  "número de exemplos. Em um corpus desbalanceado, a acurácia e as médias "
  "ponderadas podem ser altas enquanto as classes raras são sistematicamente "
  "erradas, porque acertar a classe majoritária já garante a maior parte do "
  "resultado. Por esse motivo, a medida F1 macro é adotada como métrica "
  "principal de comparação, e as demais são reportadas como complemento.")

h(2, "2.6 Matriz de confusão")
p("A matriz de confusão dispõe os rótulos verdadeiros nas linhas e as predições "
  "nas colunas. A diagonal principal contém os acertos; cada célula fora dela "
  "indica com qual classe o modelo confundiu a classe da linha. Seu valor "
  "diagnóstico está justamente fora da diagonal: ela mostra se os erros se "
  "concentram em pares de classes semanticamente próximas, o que sugere "
  "revisão dos critérios de anotação, ou se estão dispersos, o que sugere "
  "insuficiência de exemplos.")
p("Para tarefas multirrótulo, em que uma notícia pode receber mais de um "
  "indicador simultaneamente, não existe matriz única. Reporta-se então uma "
  "matriz 2×2 por indicador, cada uma tratando a presença daquele indicador "
  "como problema binário independente.")

quebra()

# ------------------------------------------------------------- arquitetura
h(1, "3 Arquitetura da solução")

h(2, "3.1 Visão geral")
p("A notícia entra por uma fila de recebidas, com o endereço eletrônico, a data "
  "informada e o texto integral. O classificador a lê e atribui, a cada "
  "indicador, uma decisão que não é binária: na estratégia de comitê, conta-se "
  "quantos dos três modelos marcaram aquele indicador, e a notícia é encaminhada "
  "conforme a concordância — nenhum voto arquiva a notícia como acervo, sem "
  "gerar pendência, voto isolado gera alerta a ser verificado, e dois ou três "
  "seguem adiante. Somente o que segue "
  "chega ao modelo de linguagem, e ele recebe apenas as instruções dos "
  "indicadores marcados. Sua tarefa é aprofundá-los: montar a lista de pessoas "
  "atingidas, preencher a ficha de cada uma e as tabelas complementares. O "
  "resultado é gravado com situação “não aprovado” e segue para revisão humana; "
  "ao ser aprovado, cada notícia da ocorrência retorna ao corpus como exemplo "
  "rotulado, fechando o ciclo de treino.")
p()
tabela(
    ["Etapa", "Natureza", "Descrição"],
    [
        ["1. Recebimento", "humana", "Analista informa endereço eletrônico, data e texto."],
        ["2. Marcação", "automática", "Classificador multirrótulo atribui indicadores."],
        ["3. Triagem", "automática", "Encaminhamento por faixa de concordância."],
        ["4. Revisão da faixa", "humana", "Verificação do que recebeu voto isolado."],
        ["5. Abertura", "humana", "Analista informa data e endereço da ocorrência."],
        ["6. Aprofundamento", "automática", "Modelo de linguagem extrai vítimas e campos estruturados."],
        ["7. Verificação", "humana", "Analista confere e corrige a lista de vítimas."],
        ["8. Regras", "automática", "Indicadores dependentes das vítimas são calculados."],
        ["9. Registro", "automática", "Gravação com situação “não aprovado”."],
        ["10. Aprovação", "humana", "Responsável libera o registro; as notícias vão ao corpus."],
    ],
    [3.4, 2.4, 10.2],
)
legenda("Tabela 2 — Etapas do fluxo, com indicação do agente responsável.")

h(2, "3.2 Por que o classificador decide e o modelo de linguagem aprofunda")
p("A divisão de papéis obedece a uma regra única: o classificador decide QUAIS "
  "indicadores existem na notícia; o modelo de linguagem só aprofunda dentro dos "
  "indicadores já marcados, e não pode acrescentar nenhum. O que ele devolver "
  "fora da marcação é descartado pelo sistema — não apenas desencorajado no "
  "texto da instrução, mas filtrado em código, porque instrução de prompt não é "
  "garantia.")
p("A regra tem três justificativas. A primeira é de custo, e está quantificada "
  "na seção 1.4. A segunda é de consistência: dois classificadores gerais sobre "
  "o mesmo texto produzem duas respostas que podem divergir, e não haveria "
  "critério para decidir qual delas prevalece. A terceira é de mensurabilidade: "
  "a marcação é reprodutível e pode ser avaliada contra um conjunto de teste "
  "fixo, enquanto a extração do modelo de linguagem não é determinística — a "
  "mesma entrada produz saídas diferentes entre execuções. Concentrar a decisão "
  "de classe no componente mensurável é o que permite afirmar algo sobre o "
  "desempenho do sistema.")
p("A marcação também AUTORIZA. Determinadas circunstâncias da vítima só são "
  "avaliadas quando a classe que as destrava foi marcada: feminicídio só é "
  "avaliado quando o indicador de mulheres baleadas está presente, e a "
  "circunstância de pessoa LGBTQIAPN+ apenas quando o indicador correspondente "
  "está. A instrução de cada uma só viaja no prompt quando pode ser usada, e a "
  "resposta é filtrada pela mesma condição, de modo que a autorização vale ainda "
  "que a instrução seja ignorada.")

h(2, "3.3 O papel do modelo de linguagem")
p("A camada de modelo de linguagem não substitui os classificadores; ela atua "
  "sobre a saída deles. Sua função é confirmar o indicador com base em "
  "instruções textuais específicas e, principalmente, consolidar a lista de "
  "vítimas relendo em conjunto todas as notícias da mesma ocorrência. Essa "
  "releitura conjunta é o que impede que pessoas citadas em fontes distintas "
  "sejam contadas mais de uma vez.")
p("A existência do estágio anterior permite avaliar o ganho da camada custosa. "
  "Como a aplicação expõe a resposta dos classificadores isoladamente, é "
  "possível verificar se o modelo de linguagem está corrigindo o classificador "
  "ou apenas repetindo a sua decisão — pergunta que não teria resposta se as "
  "duas camadas fossem medidas apenas em conjunto.")

quebra()

# ------------------------------------------------------------------ corpus
h(1, "4 Passo 1: construção do corpus")

h(2, "4.1 A premissa da anotação")
p("A primeira tarefa tem uma premissa única e deliberadamente estreita: "
  "considera-se violência armada a notícia que relata disparo de arma de fogo. "
  "Essa definição é operacional, não conceitual — ela não afirma o que é "
  "violência armada em sentido amplo, e sim fixa o critério verificável que "
  "delimita o escopo do sistema.")
p("A consequência prática é que vários casos intuitivamente associados a armas "
  "ficam fora:", recuo=False)
bullet("ameaça com arma de fogo sem que tiro tenha sido efetuado;")
bullet("apreensão de armas e munições em operação, sem confronto;")
bullet("agressão, homicídio ou lesão por outro meio, como arma branca;")
bullet("porte ilegal de arma identificado em abordagem, sem disparo.")
p("Tratar esses casos como positivos tornaria a fronteira da classe difusa, e a "
  "métrica resultante mediria a inconsistência da anotação em vez da capacidade "
  "do modelo.")

h(2, "4.2 Necessidade de exemplos negativos")
p("Um corpus formado apenas por notícias de violência armada não é utilizável "
  "para a primeira tarefa. Sem exemplos negativos, o modelo não tem o que "
  "aprender a recusar, e a matriz de confusão perde as colunas que registram "
  "justamente o erro relevante — a admissão de notícia fora do escopo. A "
  "ferramenta de anotação, por isso, exibe a contagem das duas classes e impede "
  "o início do treino enquanto uma delas não tiver exemplos suficientes.")

h(2, "4.3 Princípios da anotação")
p("A anotação segue dois princípios herdados do manual de curadoria: "
  "consideram-se apenas as circunstâncias diretamente ligadas ao MOMENTO do "
  "disparo, desprezando antecedentes do envolvido e consequências posteriores "
  "que não constituam novo disparo; e o recorte é estrito quanto à natureza do "
  "ferimento, pois registra-se disparo de arma de fogo, não ferimento por queda, "
  "estilhaço, arma branca ou mal-estar ocorridos durante o mesmo fato.")
p("A tarefa é multirrótulo: uma mesma notícia pode receber mais de um indicador, "
  "ou nenhum. O rótulo vazio é um rótulo válido e necessário, e a sua ausência "
  "no corpus é o que faz um classificador marcar algo em todo texto.")

h(2, "4.4 Rastreabilidade")
p("Cada exemplo do corpus guarda o endereço eletrônico e o texto integral da "
  "notícia que o originou, além da identificação do anotador, do instante da "
  "anotação e de um campo livre para a justificativa do rótulo em casos de "
  "fronteira. O registro da fonte não é conveniência: sem ele, nenhum resultado "
  "de treino é verificável por terceiros, e a afirmação de desempenho perde "
  "sustentação. O campo de justificativa cumpre função adicional, ao permitir "
  "revisitar decisões difíceis quando um erro sistemático aparece na matriz de "
  "confusão.")

h(2, "4.5 Ferramenta de anotação")
p("A anotação é feita em interface própria, integrada à aplicação, que cumpre o "
  "papel de ambientes especializados do gênero. A notícia é inserida com "
  "endereço eletrônico, data e texto, e o anotador marca os indicadores que o "
  "texto sustenta — nenhum, um ou vários. A definição de cada indicador fica "
  "visível ao lado da respectiva caixa durante a anotação, o que reduz a "
  "dependência de memória, e a tela afirma explicitamente que não marcar nada é "
  "uma decisão válida, não uma pendência.")
p("A fila é ordenada por pendência, não por data. Aparece primeiro a notícia "
  "que recebeu voto isolado no comitê, sinalizada em laranja, e em seguida a "
  "que ninguém examinou ainda. A notícia em que nenhum indicador foi marcado "
  "não ocupa a fila: fica retida como acervo, ao final da listagem, e é "
  "reavaliada quando há modelo novo — o falso negativo não é descartado, mas "
  "também não disputa a atenção de quem trabalha a fila. A rotulagem feita nessa fila alimenta "
  "diretamente o corpus, de modo que o trabalho de conferência e o de construção "
  "do conjunto de treino são o mesmo trabalho.")
p("A interface informa, por tarefa, se o corpus já sustenta uma medição "
  f"honesta. O limiar adotado é de {mod_corpus.MINIMO_POR_CLASSE} exemplos por "
  "classe para considerar a métrica estável, e de três exemplos como mínimo "
  "absoluto para que a classe possa ser representada nas três partições. O "
  "critério NÃO é informativo: desde outubro de 2026 o treino é recusado "
  "enquanto qualquer classe do conjunto pedido estiver abaixo do mínimo. A "
  "razão é que, abaixo de uma dezena de exemplos no conjunto de teste, a "
  "variação de um único acerto desloca o recall em dezenas de pontos "
  "percentuais, e relatar tal número como resultado seria enganoso. Treinar "
  "um subconjunto de classes já maduras continua possível, pela opção "
  "--classe, que aplica a trava apenas às classes pedidas.")

h(2, "4.6 Dimensionamento esperado")
p(f"São {len(INDICADORES_BERT)} indicadores, e a exigência amostral se conta por "
  "indicador, não por notícia: mantendo a ordem de grandeza de algumas dezenas "
  "de exemplos positivos de cada um no conjunto de treino, e somando os exemplos "
  "sem indicador algum, chega-se à casa das centenas de notícias anotadas. Como "
  "a tarefa é multirrótulo, uma mesma notícia contribui para mais de um "
  "indicador, o que atenua o total, porém não socorre o indicador raro — que "
  "continua dependendo de encontrar notícias em que ele apareça.")
p("Indicadores raros na realidade — e portanto raros no corpus — devem ser "
  "reportados com o seu suporte explícito, e a discussão dos resultados precisa "
  "distinguir limitação do modelo de insuficiência amostral. O sistema não "
  "permite treinar com menos de três exemplos da classe mais rara, por não haver "
  "como representá-la nas três partições, e recusa o treino de toda classe "
  f"com menos de {mod_corpus.MINIMO_POR_CLASSE} exemplos.")

quebra()

# --------------------------------------------------------- preprocessamento
h(1, "5 Passo 2: pré-processamento")
p("Textos jornalísticos coletados da web carregam material alheio ao fato: "
  "chamadas para outras matérias, créditos de fotografia, elementos de "
  "navegação e publicidade. Esse material consome espaço da janela de entrada do "
  "modelo sem contribuir para a decisão.")
p("O pré-processamento implementado opera no nível da sentença. Um modelo de "
  "processamento de linguagem natural segmenta o texto e preserva as sentenças "
  "que contenham pelo menos um termo do vocabulário relevante — extraído das "
  "definições dos indicadores e das listas fechadas do dicionário de dados — ou "
  "uma entidade nomeada de pessoa, local ou organização. Se o filtro for "
  "agressivo demais e restar menos de um mínimo de sentenças, o texto original é "
  "mantido, de modo que o procedimento nunca piore a entrada.")
p("ESSE FILTRO ESTÁ DESLIGADO POR PADRÃO, e a decisão é resultado de medição. "
  "Aplicado às notícias já coletadas, ele descartou cerca de um décimo do texto, "
  "e entre o descartado havia fato material: de uma reportagem sobre tiroteio "
  "saíram a sentença que posicionava a vítima em relação à viatura policial e a "
  "que informava o estado de saúde das pessoas atingidas. Nenhuma das duas "
  "continha termo do vocabulário nem entidade nomeada, e esse é todo o critério "
  "do filtro. A economia obtida, da ordem de algumas dezenas de tokens por "
  "notícia, não compensa o risco de suprimir a informação que sustenta a "
  "classificação; o recurso permanece disponível para o caso de o orçamento de "
  "tokens se tornar restritivo.")
p("Uma segunda forma de compressão, no nível da palavra, com remoção de "
  "palavras funcionais e lematização, foi implementada, testada e "
  "posteriormente removida do código. Ela remove preposições como “por” e “contra”, "
  "que são exatamente os elementos que marcam quem atirou contra quem. A "
  "supressão dessas marcas inverteu a direção atribuída ao disparo em testes "
  "conduzidos durante o desenvolvimento, produzindo classificação incorreta. O "
  "caso é registrado aqui porque ilustra um risco geral: técnicas de "
  "normalização avaliadas apenas pela economia de tokens podem destruir a "
  "informação que sustenta a tarefa.")

quebra()

# ------------------------------------------------------------------ divisao
h(1, "6 Passo 3: divisão amostral")

h(2, "6.1 Partições e proporções")
p("O corpus é dividido em três partições: treino, usada para atualizar os "
  "pesos; validação, usada para escolher hiperparâmetros e observar "
  "sobreajuste; e teste, usada uma única vez por configuração, para produzir as "
  "métricas reportadas. A proporção de referência é de "
  f"{int(mod_corpus.PROPORCAO_PADRAO[0]*100)}%, "
  f"{int(mod_corpus.PROPORCAO_PADRAO[1]*100)}% e "
  f"{int(mod_corpus.PROPORCAO_PADRAO[2]*100)}%, respectivamente.")

h(2, "6.2 Estratificação")
p("A divisão é estratificada para que a proporção de cada indicador se mantenha "
  "nas três partições. Em tarefa multirrótulo isso não é imediato: a mesma "
  "notícia pode portar três indicadores e pertenceria a três estratos "
  "simultaneamente, de modo que o procedimento usual de fatiar cada classe "
  "separadamente não se aplica.")
p("Duas soluções intuitivas foram descartadas, e o registro da razão importa "
  "porque a segunda é silenciosamente defeituosa. A primeira é tratar a "
  f"combinação de indicadores como estrato; com {len(mod_ind.do_bert())} "
  "indicadores existem, em princípio, dezenas de combinações observáveis, quase "
  "todas com um ou dois exemplos, e um estrato de um exemplo recai "
  "integralmente sobre uma partição. A segunda é embaralhar essas combinações e "
  "percorrê-las em paralelo, formando uma sequência em que o primeiro exemplo "
  "de cada combinação precede o segundo de qualquer outra, para então cortar a "
  "sequência nas proporções desejadas. O procedimento parece distribuir os "
  "estratos pequenos, mas produz o efeito oposto: como o início da sequência "
  "recebe um exemplo de cada combinação, o corte inicial — o treino — absorve "
  "todos os estratos pequenos por inteiro, e a cauda da sequência, de onde saem "
  "validação e teste, contém apenas os estratos grandes. O resultado observado "
  "em verificação empírica foi o esvaziamento completo dos indicadores raros nas "
  "duas partições de avaliação.")
p("Adota-se, em lugar disso, a estratificação iterativa para dados "
  "multirrótulo, no formato proposto por Sechidis, Tsoumakas e Vlahavas (2011). "
  "Cada partição recebe uma cota global e uma cota por indicador, ambas "
  "proporcionais ao tamanho pretendido. As notícias são distribuídas uma a uma, "
  "e a ordem de atendimento é determinada pelo indicador com menor número de "
  "exemplos ainda não distribuídos: cada notícia que o contém é encaminhada à "
  "partição com maior cota remanescente naquele indicador, descontando-se em "
  "seguida a cota de todos os indicadores que a notícia porta. Atender primeiro "
  "o indicador mais raro é o que faz os seus poucos exemplos serem repartidos "
  "entre as três partições em vez de se concentrarem na primeira. A partição que "
  "já completou a sua cota global é excluída da disputa, o que mantém os "
  "tamanhos das partições aderentes às proporções pretendidas.")
p("A notícia sem nenhum indicador — o exemplo negativo — é tratada como "
  "portadora de um indicador próprio, exclusivo dela. Sem esse artifício, as "
  "negativas seriam distribuídas apenas no fim do procedimento, pela cota que "
  "restasse, e poderiam concentrar-se em uma única partição; como são elas que "
  "ensinam o modelo a não atribuir indicador algum, a sua proporção precisa ser "
  "preservada igualmente.")
p("A consequência prática justifica o cuidado. Sem exemplo de um indicador na "
  "validação, não há como calibrar o limiar de decisão daquele indicador, e ele "
  "permanece no corte neutro, exatamente a condição em que o indicador raro "
  "deixa de ser atribuído. Sem exemplo no conjunto de teste, a medida F1 do "
  "indicador é reportada como zero com suporte zero, o que se confunde com erro "
  "do modelo. A interface de anotação exibe a contagem por indicador em cada "
  "partição justamente para que essas duas situações sejam visíveis antes do "
  "treino.")

h(2, "6.3 Fixação da divisão")
p("A partição de cada exemplo é gravada em banco de dados, e não sorteada a "
  "cada execução. A razão é de comparabilidade: se o conjunto de teste mudasse "
  "entre execuções, as métricas de dois modelos passariam a ser medidas sobre "
  "amostras diferentes, e a diferença observada entre eles confundiria efeito do "
  "modelo com efeito da amostra. O sorteio usa semente explícita, registrada "
  "junto ao resultado, e é refeito somente quando o corpus cresce.")

quebra()

# ------------------------------------------------------------------ modelos
h(1, "7 Passo 4: modelos comparados e estratégias de ajuste")

h(2, "7.1 Modelos pré-treinados")
p(f"A comparação abrange {len(mod_treino.MODELOS)} modelos pré-treinados, todos "
  "monolíngues de português, conforme a premissa registrada na seção 2.3. A "
  "variação coberta é, portanto, o efeito do tamanho do modelo e a diferença "
  "entre as duas famílias disponíveis para o idioma.")
p()
tabela(
    ["Identificação", "Modelo", "Parâmetros", "Família"],
    [[chave, m["rotulo"], m["parametros"], m["tipo"]]
     for chave, m in mod_treino.MODELOS.items()],
    [3.6, 4.0, 2.4, 6.0],
)
legenda("Tabela 3 — Modelos submetidos à comparação.")

p("A inclusão de duas variantes de tamanho da mesma família — BERTimbau base e "
  "large — permite isolar o efeito da escala, uma vez que o corpus e o "
  "procedimento de pré-treinamento são os mesmos. A Albertina PT-BR, de porte "
  "comparável à variante base mas pré-treinada sobre corpora distintos e "
  "especificamente de português do Brasil, fornece o contraste entre famílias. "
  "Se a variante base alcançar desempenho próximo ao da variante large, a "
  "escolha de implementação passa a ser decidida por custo de inferência e "
  "memória, e não por qualidade — observação relevante porque a variante large "
  "é também a que corre maior risco de não caber na memória disponível.")
p("Modelos cujo treino não se complete — por exemplo, por insuficiência de "
  "memória da unidade de processamento gráfico nas configurações de maior "
  "comprimento de sequência — têm a falha registrada com a respectiva mensagem. "
  "A informação integra o resultado: um modelo que não cabe no equipamento "
  "disponível é um achado de viabilidade, e omiti-lo da tabela daria a entender "
  "que não foi avaliado.")

h(2, "7.2 Natureza multirrótulo da tarefa")
p("A tarefa não consiste em atribuir a uma notícia o indicador que melhor a "
  "descreve, e sim em decidir, para cada indicador do catálogo, se a notícia o "
  "sustenta. Uma ocorrência em que uma adolescente é atingida durante operação "
  "policial porta simultaneamente os indicadores de criança ou adolescente "
  "baleado, de mulher baleada e de ação policial, e a atribuição de um deles não "
  "pode suprimir as demais.")
p("A consequência arquitetural é a escolha da função de ativação da camada de "
  "saída. Adota-se uma função logística independente por indicador, e não a "
  "função softmax. Sob softmax as pontuações somam um e as classes competem "
  "entre si, de modo que atribuir dois indicadores exigiria que um cedesse "
  "pontuação ao outro; sob a função logística independente, cada indicador é "
  "comparado ao seu próprio limiar e decide isoladamente. A ausência de "
  "indicador algum é, por construção, uma resposta válida, e corresponde à "
  "notícia fora do escopo de monitoramento. A função de perda correspondente é "
  "a entropia cruzada binária aplicada a cada saída.")

h(2, "7.3 Estratégias de ajuste comparadas")
p("Três estratégias produzem essa mesma forma de saída, e todas são "
  "implementadas e submetidas à mesma partição de teste, de modo que a escolha "
  "entre elas seja resultado medido e não suposição.")
p()
tabela(
    ["Estratégia", "Descrição", "Redes ajustadas"],
    [["modelo único",
      "Uma rede com uma saída logística independente por indicador; o "
      "codificador é compartilhado por todos eles.",
      "1"],
     ["um por indicador",
      "Uma rede binária independente por indicador, no esquema "
      "um-contra-resto; as saídas são reunidas em um vetor multirrótulo.",
      str(len(mod_ind.do_bert()))],
     ["comitê por indicador",
      f"{len(mod_treino.MODELOS_CLUSTER)} redes binárias por indicador, uma de "
      "cada arquitetura, com decisão por votação: o indicador só é atribuído "
      "automaticamente quando a concordância atinge o corte.",
      str(len(mod_treino.MODELOS_CLUSTER) * len(mod_ind.do_bert()))]],
    [3.4, 8.6, 2.0,
     ],
)
legenda("Tabela 4 — Estratégias de ajuste submetidas à comparação.")

p("A estratégia de modelo único tem a seu favor o compartilhamento de "
  "representação: o que a rede aprende sobre a linguagem do noticiário policial "
  "beneficia todos os indicadores ao mesmo tempo, o que é relevante quando o "
  "corpus é pequeno. A estratégia de um modelo por indicador tem a seu favor a "
  "independência: cada indicador recebe a capacidade integral de uma rede e o "
  "seu próprio ponto de parada, e um indicador raro não disputa parâmetros com "
  "os frequentes. O custo é proporcional ao número de indicadores, em tempo de "
  "treino, em espaço de armazenamento dos pesos e em memória durante a "
  "inferência, uma vez que todas as redes precisam estar carregadas para "
  "classificar uma única notícia.")
p("A terceira estratégia submete a mesma notícia a três arquiteturas distintas "
  "por indicador e decide por votação, com um limiar calibrado para CADA rede "
  "separadamente. Essa separação é deliberada: um limiar comum às três faria com "
  f"que discordassem menos por construção, e a discordância é o sinal "
  "procurado. A decisão resultante tem três faixas, e não duas — nenhum voto "
  "arquiva a notícia como acervo, sem gerar pendência, voto abaixo do corte "
  "gera alerta a ser verificado, e a partir do corte o indicador segue "
  "automaticamente. O preço é "
  "explícito nas métricas, que reportam a fração de decisões não automáticas ao "
  "lado do desempenho.")
p("A viabilidade do comitê depende de um detalhe de implementação: as redes são "
  "treinadas e descartadas uma a uma, e na inferência carrega-se apenas o grupo "
  "de um indicador por vez. Mantidas todas em memória simultaneamente, as "
  f"{len(mod_treino.MODELOS_CLUSTER) * len(mod_ind.do_bert())} redes excederiam "
  "a memória da unidade gráfica utilizada; carregadas em grupo, o pico medido "
  "ficou em torno de um quarto dela. O custo transferiu-se para o tempo: a "
  "classificação de uma notícia passa a exigir vários ciclos de carregamento.")
p("A comparação entre as estratégias multiplica o número de execuções, "
  f"razão pela qual é conduzida sobre uma única configuração da grade: com "
  f"{len(mod_treino.MODELOS)} modelos e {len(mod_treino.GRADE)} configurações, a "
  f"estratégia de modelo único ajusta "
  f"{len(mod_treino.MODELOS) * len(mod_treino.GRADE)} redes, a de um modelo por "
  f"indicador ajusta "
  f"{len(mod_treino.MODELOS) * len(mod_treino.GRADE) * len(mod_ind.do_bert())}, e "
  "a de comitê ajusta "
  f"{len(mod_treino.GRADE) * len(mod_treino.MODELOS_CLUSTER) * len(mod_ind.do_bert())} "
  "— esta última uma vez por configuração, e não por modelo, pois emprega os "
  "três simultaneamente.")

h(2, "7.4 Desbalanceamento: ponderação da perda")
p("Os indicadores têm frequências muito distintas no corpus, e o "
  "desbalanceamento afeta a classificação multirrótulo de maneira específica. "
  "Para um indicador presente em vinte de quatrocentas notícias, a perda de "
  "entropia cruzada binária é quase minimizada por um modelo que nunca o "
  "atribui: o acerto em trezentos e oitenta exemplos negativos compensa o erro "
  "em vinte positivos. O comportamento resultante — atribuir apenas os "
  "indicadores frequentes — é frequentemente interpretado como incapacidade do "
  "modelo de reconhecer mais de um indicador, quando se trata de um efeito da "
  "função de perda.")
p("A correção adotada é a ponderação do termo positivo da perda de cada "
  "indicador pela razão entre o número de exemplos negativos e o de positivos "
  "daquele indicador no conjunto de treino. A razão é limitada a um teto de "
  f"{mod_treino.TETO_PESO_POSITIVO:.0f}, pois um indicador com dois exemplos "
  "entre quatrocentos receberia peso próximo a duzentos e passaria a responder "
  "por quase toda a atualização dos parâmetros, invertendo o problema. Os pesos "
  "efetivamente aplicados são registrados junto às métricas da execução.")

h(2, "7.5 Calibração do limiar de decisão")
p("A segunda correção incide sobre o limiar de decisão. O valor convencional de "
  "cinco décimos é adequado quando as classes são equilibradas, mas não quando "
  "não são: a saída logística de um indicador raro tende a permanecer abaixo "
  "desse valor mesmo nos casos em que o indicador se aplica, de modo que o "
  "indicador deixa de ser atribuído ainda que as suas pontuações estejam "
  "corretamente ordenadas. Um limiar comum a todos os indicadores impõe, assim, "
  "um custo concentrado nos indicadores menos frequentes.")
p("Adota-se um limiar por indicador, obtido por varredura sobre o conjunto de "
  "validação, em incrementos de cinco centésimos, selecionando-se o valor que "
  "maximiza a medida F1 daquele indicador. Em caso de empate, prefere-se o "
  "limiar mais baixo, decisão justificada pela assimetria entre os dois tipos "
  "de erro nesta arquitetura: o indicador atribuído indevidamente é submetido ao "
  "modelo de linguagem da etapa seguinte, que pode descartá-lo, enquanto o "
  "indicador não atribuído não chega a ser avaliado, e a perda é definitiva.")
p("A calibração utiliza exclusivamente o conjunto de validação. Calibrar sobre "
  "o conjunto de teste produziria uma estimativa otimista, por selecionar o "
  "limiar com conhecimento dos dados sobre os quais o resultado é reportado. "
  "Para que o efeito da calibração seja verificável e não apenas afirmado, "
  "registra-se junto a cada execução a medida F1 que o mesmo modelo obteria, no "
  "mesmo conjunto de teste, com todos os limiares fixados em cinco décimos; a "
  "diferença entre os dois valores é o efeito medido do procedimento. O limiar "
  "de cada indicador é gravado com os pesos do modelo, pois a decisão de "
  "classificação não é reproduzível sem ele.")
p("Indicadores sem exemplo positivo no conjunto de validação não admitem "
  "calibração e permanecem no valor convencional, situação que é registrada "
  "explicitamente para que o valor não seja lido como resultado de seleção.")

quebra()

# ----------------------------------------------------------- hiperparametros
h(1, "8 Passo 5: busca de hiperparâmetros")

h(2, "8.1 Grade")
p("Cada modelo é treinado sob todas as configurações da grade abaixo, "
  f"totalizando {len(mod_treino.MODELOS) * len(mod_treino.GRADE)} execuções por "
  "tarefa.")
p()
tabela(
    ["#", "Taxa de aprendizado", "Lote", "Épocas", "Decaimento", "Comprimento"],
    [[i + 1, f"{h_['learning_rate']:.0e}", h_["batch_size"], h_["epocas"],
      h_["weight_decay"], h_["max_length"]]
     for i, h_ in enumerate(mod_treino.GRADE)],
    [1.0, 4.0, 1.6, 1.8, 3.0, 4.6],
)
legenda("Tabela 5 — Grade de hiperparâmetros.")

p("Os valores situam-se na faixa recomendada pela literatura de ajuste fino de "
  "modelos BERT, com taxas de aprendizado entre 2·10⁻⁵ e 5·10⁻⁵ e número "
  "reduzido de épocas. A grade é construída de modo que as configurações com "
  "mais épocas empreguem taxa menor, pois corpora pequenos sobreajustam "
  "rapidamente. Ao treino aplica-se ainda aquecimento linear da taxa de "
  "aprendizado em dez por cento dos passos e limitação da norma do gradiente, "
  "medidas usuais para estabilizar as primeiras atualizações.")

h(2, "8.2 Critério de seleção")
p("A configuração vencedora de cada tarefa é escolhida pela medida F1 macro "
  "obtida no conjunto de validação, nunca no de teste. A distinção é central "
  "para a validade do resultado: o conjunto de teste estima o desempenho em "
  "dados não vistos, e essa estimativa deixa de ser válida se ele for consultado "
  "para tomar decisões de modelagem. Escolher a configuração pelo teste "
  "transformaria o número reportado em um máximo sobre a grade, sistematicamente "
  "otimista.")
p("Quando uma tarefa não dispõe de conjunto de validação — situação possível "
  "com classes muito raras —, o critério recai sobre o teste, e essa condição é "
  "registrada explicitamente junto ao resultado, por deixar de ser uma "
  "estimativa limpa.")

h(2, "8.3 Acompanhamento do aprendizado")
p("Registram-se, por época, a perda média no conjunto de treino e a medida F1 "
  "macro no conjunto de validação. A leitura conjunta das duas séries é o que "
  "permite diagnosticar sobreajuste: perda decrescente acompanhada de medida F1 "
  "de validação estável ou em queda indica que o modelo está memorizando o "
  "treino em vez de generalizar, e sugere reduzir épocas ou taxa de aprendizado.")

quebra()

# ---------------------------------------------------------------- avaliacao
h(1, "9 Passo 6: protocolo de avaliação")

h(2, "9.1 Métricas reportadas")
p("Para cada execução são registradas a acurácia; a precisão, o recall e a "
  "medida F1 em média macro e em média ponderada pelo suporte; as mesmas três "
  "métricas por classe, acompanhadas do número de exemplos daquela classe no "
  "conjunto de teste; e a matriz de confusão completa. Para a tarefa "
  "multirrótulo reportam-se, adicionalmente, as médias micro e a acurácia de "
  "subconjunto, que exige acerto simultâneo de todos os rótulos de uma notícia.")
p("O suporte por classe é parte obrigatória do relato. Uma classe sem exemplos "
  "no conjunto de teste recebe, por convenção, medida F1 igual a zero, o que "
  "não representa erro do modelo e sim ausência de medição. Sem o suporte ao "
  "lado, o valor é lido incorretamente.")
p("Duas grandezas adicionais são registradas por serem específicas desta "
  "tarefa. A primeira é a medida F1 que o mesmo modelo obteria, no mesmo "
  "conjunto de teste, com todos os limiares fixados em cinco décimos; "
  "confrontada com a medida reportada, ela torna verificável o efeito da "
  "calibração descrita na seção 7.5. A segunda é o número médio de indicadores "
  "atribuídos por notícia, apresentado ao lado do número médio presente no "
  "gabarito, acompanhado da contagem de notícias com dois ou mais indicadores e "
  "da contagem de notícias sem indicador algum. Essas contagens respondem "
  "diretamente à pergunta sobre a capacidade do modelo de atribuir indicadores "
  "simultâneos: uma média atribuída sistematicamente inferior à do gabarito "
  "indica atribuição insuficiente, e o diagnóstico recai em primeiro lugar sobre "
  "o limiar, não sobre a representação aprendida.")

h(2, "9.2 Estrutura da matriz de confusão")
p("Em tarefa multirrótulo não existe matriz única: reporta-se uma matriz 2×2 "
  "por indicador, cada uma tratando a presença daquele indicador como problema "
  "independente. A tabela a seguir exemplifica a disposição adotada, com os "
  "quatro resultados e a sua interpretação no domínio.")
p()
tabela(
    ["", "Predito: indicador ausente", "Predito: indicador presente"],
    [
        ["Real: ausente",
         "Verdadeiro negativo — o indicador corretamente não foi atribuído.",
         "Falso positivo — indicador atribuído sem base; a instrução dele viaja no prompt sem necessidade, e o modelo de linguagem pode descartá-lo ao não achar sustentação no texto."],
        ["Real: presente",
         "Falso negativo — indicador não atribuído; a instrução dele não viaja, o aprofundamento correspondente não acontece e o indicador final é subestimado.",
         "Verdadeiro positivo — indicador atribuído e encaminhado ao aprofundamento."],
    ],
    [3.4, 6.3, 6.3],
)
legenda("Tabela 6 — Interpretação da matriz de confusão de um indicador.")

p("A assimetria entre os dois tipos de erro é relevante para a escolha "
  "operacional do ponto de corte. O falso positivo é descartável: a instrução "
  "daquela classe viaja no prompt, e tanto o modelo de linguagem quanto a revisão "
  "humana podem recusá-la. O falso negativo é perda definitiva, porque a classe "
  "que não foi marcada não chega a ser avaliada por ninguém. Essa consideração "
  "orienta duas decisões do sistema: a priorização do recall na calibração do "
  "limiar, que em caso de empate escolhe o corte mais baixo, e a retenção em "
  "acervo — e não o descarte — de toda notícia em que nenhum indicador foi "
  "marcado, de modo que um modelo posterior possa reavaliá-la.")

h(2, "9.3 Avaliação interativa")
p("Além das métricas agregadas, a aplicação disponibiliza interface de teste "
  "que recebe um texto e exibe a resposta do classificador com a distribuição "
  "de confiança sobre todas as classes, antes de qualquer intervenção do modelo "
  "de linguagem. A informação de confiança é parte do resultado: um acerto com "
  "probabilidade próxima ao limiar e outro com probabilidade alta descrevem "
  "situações distintas quanto à decisão de confiar no filtro ou encaminhar o "
  "caso à camada seguinte.")
p("A mesma interface permite enviar ao corpus um texto em que o modelo errou, "
  "fechando o ciclo entre avaliação e ampliação do conjunto anotado.")

quebra()

# ---------------------------------------------------------------------- llm
h(1, "10 Passo 7: validação por modelo de linguagem")
p("A camada final submete as notícias já filtradas a um modelo de linguagem de "
  "grande porte, com instruções textuais específicas por indicador. A saída do "
  "classificador é fornecida como sugestão, explicitamente sujeita a "
  "confirmação ou descarte, e não como decisão.")
p("A atribuição central dessa camada é a consolidação de vítimas. A cada notícia "
  "acrescentada a uma ocorrência, todas as notícias daquela ocorrência são "
  "relidas em conjunto e a lista de pessoas atingidas é reconstruída do zero. O "
  "procedimento é mais custoso do que processar apenas o texto novo, porém é o "
  "que impede a soma indevida: uma pessoa descrita em três fontes diferentes "
  "deve ser contada uma vez, e o total nunca pode ser inferior ao maior número "
  "citado em uma única fonte.")
p("Correções feitas pelo analista prevalecem sobre a saída automática nos "
  "reprocessamentos seguintes. Quando o modelo passa a discordar de um registro "
  "já corrigido — por exemplo, ao identificar em notícia posterior que uma "
  "pessoa ferida veio a falecer —, a divergência é apresentada como proposta "
  "sujeita a aceitação, preservando a correção humana até decisão explícita.")

quebra()

# ----------------------------------------------------------- reprodutibilidade
h(1, "11 Reprodutibilidade")

h(2, "11.1 Execução em unidade gráfica ou em processador")
p("O treino executa indiferentemente em unidade de processamento gráfico ou em "
  "processador comum. A seleção é automática: a opção padrão emprega a unidade "
  "gráfica quando disponível e recai sobre o processador quando não há uma. A "
  "escolha pode ser forçada por parâmetro de linha de comando, o que permite "
  "reproduzir o trabalho em equipamento sem placa dedicada. Em processador, o "
  "tempo de treino é substancialmente maior, mas o procedimento e os resultados "
  "são os mesmos.")
p("A distinção entre os dois modos reside apenas na variante da biblioteca de "
  "tensores instalada. A instalação a partir do arquivo de dependências traz a "
  "variante para processador, que funciona em qualquer equipamento; a variante "
  "com suporte à unidade gráfica é instalada em comando separado, substituindo "
  "a anterior:")
codigo([
    "# instalação base (variante para processador)",
    "pip install -r requirements.txt",
    "python -m spacy download pt_core_news_sm",
    "",
    "# opcional: variante com suporte a unidade gráfica NVIDIA",
    "pip install torch --index-url https://download.pytorch.org/whl/cu126",
    "",
    "# retorno à variante para processador",
    "pip install torch --index-url https://download.pytorch.org/whl/cpu",
])
p("O ambiente efetivamente utilizado em cada execução — versão da biblioteca, "
  "disponibilidade de unidade gráfica, modelo da placa e dispositivo "
  "selecionado — é registrado junto às métricas e exibido na aplicação, de modo "
  "que o relato de tempo de treino seja interpretável.")

h(2, "11.2 Execução dos experimentos")
p("A comparação completa é disparada por linha de comando, com as variações "
  "abaixo:")
codigo([
    "# sorteia a divisão e compara os três modelos",
    "python rodar_treino.py --dividir",
    "",
    "# compara as duas de rede única na mesma partição de teste",
    "python rodar_treino.py --estrategia ambas --grade 1",
    "",
    "# as três, incluindo o comitê, na mesma tabela",
    "python rodar_treino.py --estrategia todas --grade 1",
    "",
    "# ajusta uma rede binária por indicador",
    "python rodar_treino.py --estrategia por_indicador --grade 1",
    "",
    "# força execução em processador",
    "python rodar_treino.py --dispositivo cpu",
    "",
    "# restringe a um modelo e a uma configuração da grade",
    "python rodar_treino.py --modelo bertimbau-base --grade 1",
    "",
    "# desliga as correções de desbalanceamento, para medir o efeito delas",
    "python rodar_treino.py --sem-peso-classes --sem-calibrar-limiar --grade 1",
])

h(2, "11.3 Registro dos resultados")
p("Cada execução grava uma linha em banco de dados contendo tarefa, estratégia "
  "de ajuste, modelo, hiperparâmetros, métricas completas, matriz de confusão "
  "por indicador, limiar calibrado de cada indicador, histórico por época, "
  "tamanho das três partições, número de redes ajustadas, duração e "
  "dispositivo. O registro da estratégia e do número de redes é o que permite "
  "ler a tabela como comparação entre as duas alternativas, e não como lista de "
  "execuções heterogêneas; o registro dos limiares é condição de "
  "reprodutibilidade, pois a mesma rede sob outros limiares produz outras "
  "métricas.")
p("Os pesos do melhor modelo são gravados em disco acompanhados de um manifesto "
  "que declara a estratégia adotada, a ordem dos indicadores, o limiar de cada "
  "um e o comprimento máximo de sequência empregado. A ordem dos indicadores é "
  "registrada porque define as posições do vetor de saída: um indicador "
  "acrescentado ao catálogo após o treino não existe nos pesos, e a leitura do "
  "catálogo corrente deslocaria todas as posições. As estratégias "
  "coexistem em disco, em diretórios distintos, e o manifesto determina qual "
  "delas responde em produção. O corpus anotado é exportável em planilha, para "
  "anexação ao trabalho e verificação independente.")

h(2, "11.4 Determinismo")
p("A semente do sorteio da divisão é explícita e registrada. A inicialização da "
  "camada de classificação e a ordem de apresentação dos lotes permanecem "
  "sujeitas a variação entre execuções, de modo que repetições da mesma "
  "configuração produzem valores próximos, porém não idênticos. Diferenças "
  "pequenas entre modelos devem, por isso, ser interpretadas com cautela, e a "
  "repetição da execução vencedora com sementes distintas é recomendada antes de "
  "afirmar superioridade.")

quebra()

# --------------------------------------------------------------- resultados
h(1, "12 Resultados")
p("Esta seção é preenchida com a saída das execuções conduzidas sobre o corpus "
  "anotado. A aplicação produz as tabelas abaixo automaticamente, e os valores "
  "devem ser transcritos a partir dela, acompanhados da data de execução e do "
  "tamanho do corpus naquele momento.")
p("Enquanto o corpus estiver em construção, a seção permanece sem números: "
  "métricas calculadas sobre poucos exemplos por classe variam de forma ampla "
  "com o acerto ou erro de um único caso e não sustentam conclusão, conforme "
  "discutido na seção 4.5.")

h(2, "12.1 Caracterização do corpus")
p("Reportar o total de notícias anotadas, a distribuição por indicador, a "
  "contagem de notícias sem indicador algum e o tamanho das três partições. "
  "Indicadores com suporte reduzido devem ser identificados nesta subseção, pois "
  "condicionam a leitura das métricas.")
p()
tabela(
    ["Partição", "Total", "Com indicador", "Sem indicador"],
    [["Treino", "", "", ""], ["Validação", "", "", ""],
     ["Teste", "", "", ""], ["Total", "", "", ""]],
    [4.0, 4.0, 4.0, 4.0],
)
legenda("Tabela 7 — Composição do corpus por partição (a preencher).")

h(2, "12.2 Desempenho por modelo e configuração")
p("Reportar, para cada modelo e configuração, as métricas e a matriz de "
  "confusão. Destacar a configuração vencedora segundo o critério da seção 8.2 e "
  "discutir a distribuição dos dois tipos de erro à luz da assimetria "
  "apresentada na seção 9.2.")
p()
tabela(
    ["Modelo", "Taxa", "Lote", "Épocas", "Acurácia", "Precisão", "Recall", "F1 macro"],
    [[m["rotulo"], "", "", "", "", "", "", ""] for m in mod_treino.MODELOS.values()],
    [3.6, 1.6, 1.2, 1.6, 2.0, 1.8, 1.6, 2.0],
)
legenda("Tabela 8 — Desempenho, melhor configuração por modelo (a preencher).")

h(2, "12.3 Desempenho por indicador")
p("Reportar as médias macro e micro, as métricas por indicador e a matriz 2×2 de "
  "cada um, conforme a seção 2.6. A análise deve privilegiar os indicadores com "
  "maior taxa de falso negativo e verificar se correspondem a fronteiras "
  "reconhecidamente difíceis do manual de curadoria — hipótese que, se "
  "confirmada, indica revisão de critério de anotação em vez de limitação do "
  "modelo. O suporte de cada indicador no conjunto de teste deve ser reportado "
  "ao lado da sua métrica, sem o que a leitura de um F1 isolado é enganosa.")

h(2, "12.4 Comparação entre as estratégias de ajuste")
p("Reportar, para a mesma partição de teste, as médias macro das "
  f"{len(mod_treino.ESTRATEGIAS)} estratégias descritas na seção 7.3, "
  "acompanhadas do número de redes ajustadas e do tempo total de treino, de modo "
  "que o eventual ganho seja confrontado com o seu custo. Reportar também as "
  "métricas por indicador em cada estratégia, pois o ganho esperado da "
  "independência entre redes concentra-se nos indicadores menos frequentes e "
  "pode não aparecer na média.")
p("Para a estratégia de comitê, reportar adicionalmente a distribuição da "
  "concordância, a fração de decisões encaminhadas à revisão humana e o "
  "desempenho que se obteria exigindo unanimidade. Sem esses três números, a "
  "comparação com as demais estratégias é desigual: o comitê troca cobertura "
  "automática por precisão, e o que ele deixa de decidir não aparece na métrica "
  "de quem decide tudo.")

h(2, "12.5 Efeito das correções de desbalanceamento")
p("Reportar as médias macro em quatro condições — com e sem ponderação da "
  "perda, com e sem calibração do limiar por indicador — e, em cada uma, o "
  "número médio de indicadores atribuídos por notícia ao lado do número médio "
  "do gabarito. A comparação isola o quanto da atribuição insuficiente de "
  "indicadores se deve à função de perda e o quanto se deve ao limiar de "
  "decisão, questão distinta da capacidade de representação do modelo.")

h(2, "12.6 Contribuição da camada de modelo de linguagem")
p("Comparar a saída do classificador com a decisão final após a validação, "
  "quantificando em que proporção dos casos a camada confirma, corrige ou "
  "reverte a classificação. A comparação é viabilizada pela interface descrita "
  "na seção 9.3 e responde se a camada custosa agrega decisão própria ou apenas "
  "reproduz o estágio anterior.")

quebra()

# --------------------------------------------------------------- limitacoes
h(1, "13 Limitações e ameaças à validade")

h(2, "13.1 Validade interna")
p("A anotação foi conduzida por um único anotador, o que impede o cálculo de "
  "concordância entre anotadores e deixa sem estimativa a parcela de erro "
  "atribuível à subjetividade do rótulo. A anotação dupla de uma amostra do "
  "corpus, com cálculo de coeficiente de concordância, é a mitigação indicada.")
p("A busca de hiperparâmetros percorre grade reduzida, selecionada por "
  "conhecimento prévio da literatura, e não uma exploração ampla do espaço. O "
  "desempenho reportado é, portanto, um limite inferior do alcançável.")

h(2, "13.2 Validade externa")
p("O corpus reflete os veículos de imprensa e as regiões efetivamente cobertas "
  "durante a sua construção. Modelos treinados sobre ele podem degradar diante "
  "de vocabulário regional distinto ou de convenções de redação de outros "
  "veículos. A avaliação em conjunto proveniente de fontes não representadas no "
  "treino é o teste adequado para essa limitação.")

h(2, "13.3 Viés de cobertura")
p("Indicadores construídos a partir de notícias herdam o viés de noticiabilidade "
  "da imprensa. Episódios em determinadas localidades ou envolvendo determinados "
  "perfis de vítima são noticiados com frequência desigual, e nenhum "
  "aprimoramento de classificador corrige essa distorção de origem. O sistema "
  "mede violência armada noticiada, e os resultados devem ser enunciados nesses "
  "termos.")

h(2, "13.4 Dependência de serviço externo")
p("A camada de validação depende de serviço de terceiros, cujo modelo pode ser "
  "atualizado ou descontinuado. Como a saída do classificador é registrada "
  "separadamente da decisão final, é possível reavaliar o efeito de uma troca de "
  "modelo sem reprocessar o corpus, o que delimita o impacto dessa dependência.")

quebra()

# ---------------------------------------------------- consideracoes finais
h(1, "14 Considerações finais")
p("A documentação apresentou o método completo de um sistema de classificação "
  "de notícias de violência armada em dois estágios, com validação posterior por "
  "modelo de linguagem. As decisões de projeto foram justificadas pelos seus "
  "efeitos verificáveis: a separação entre escopo e categoria pela "
  "interpretabilidade das matrizes de confusão e pela redução de custo; a "
  "fixação da divisão amostral pela comparabilidade entre modelos; a seleção de "
  "hiperparâmetros no conjunto de validação pela validade da estimativa de "
  "desempenho; e o registro da fonte de cada exemplo pela auditabilidade do "
  "resultado.")
p("A contribuição metodológica central não está em um modelo específico, e sim "
  "no protocolo que permite afirmar algo sobre ele. O trabalho descreve um "
  "caminho verificável: um corpus cuja procedência é rastreável, uma divisão "
  "amostral estável, uma comparação ampla de arquiteturas sob busca de "
  "hiperparâmetros, métricas que não escondem classes raras e uma interface que "
  "expõe a decisão de cada camada isoladamente.")
p("Como continuidade, destacam-se a anotação dupla com medida de concordância, a "
  "ampliação do corpus nos indicadores de menor suporte, a avaliação em fontes "
  "não representadas no treino, a investigação de técnicas de balanceamento para "
  "os indicadores raros e a construção de um conjunto de referência para a etapa "
  "de extração — que hoje não tem métrica própria, por não haver gabarito dos "
  "campos que o modelo de linguagem preenche.")

quebra()

# -------------------------------------------------------------- referencias
h(1, "Referências")
for ref in [
    "DEVLIN, J. et al. BERT: pre-training of deep bidirectional transformers for "
    "language understanding. In: Proceedings of NAACL-HLT, 2019.",
    "SOUZA, F.; NOGUEIRA, R.; LOTUFO, R. BERTimbau: pretrained BERT models for "
    "Brazilian Portuguese. In: Brazilian Conference on Intelligent Systems "
    "(BRACIS), 2020.",
    "CONNEAU, A. et al. Unsupervised cross-lingual representation learning at "
    "scale. In: Proceedings of ACL, 2020.",
    "SANH, V. et al. DistilBERT, a distilled version of BERT: smaller, faster, "
    "cheaper and lighter. arXiv:1910.01108, 2019.",
    "RODRIGUES, J. et al. Advancing neural encoding of Portuguese with "
    "transformer Albertina PT-*. arXiv:2305.06721, 2023.",
    "VASWANI, A. et al. Attention is all you need. In: Advances in Neural "
    "Information Processing Systems, 2017.",
    "SOKOLOVA, M.; LAPALME, G. A systematic analysis of performance measures for "
    "classification tasks. Information Processing & Management, v. 45, n. 4, 2009.",
    "TSOUMAKAS, G.; KATAKIS, I. Multi-label classification: an overview. "
    "International Journal of Data Warehousing and Mining, v. 3, n. 3, 2007.",
    "SECHIDIS, K.; TSOUMAKAS, G.; VLAHAVAS, I. On the stratification of "
    "multi-label data. In: Machine Learning and Knowledge Discovery in Databases "
    "(ECML PKDD), 2011.",
    "LIPTON, Z. C.; ELKAN, C.; NARAYANASWAMY, B. Optimal thresholding of "
    "classifiers to maximize F1 measure. In: Machine Learning and Knowledge "
    "Discovery in Databases (ECML PKDD), 2014.",
    "BRASIL. Lei nº 8.069, de 13 de julho de 1990. Estatuto da Criança e do "
    "Adolescente.",
]:
    par = doc.add_paragraph()
    par.paragraph_format.line_spacing = 1.0
    par.paragraph_format.space_after = Pt(10)
    par.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    par.add_run(ref)

quebra()

# ----------------------------------------------------------------- apendices
h(1, "Apêndice A — Classes do classificador")
p("Reprodução das definições utilizadas na anotação e no prompt do modelo de "
  "linguagem, para os indicadores que o classificador aprende do texto. A coluna "
  "de origem registra se a circunstância correspondente depende de o indicador "
  "ter sido marcado.")
p()
tabela(
    ["Indicador", "Definição", "Circunstâncias que destrava"],
    [[i["nome"], i["definicao"], ", ".join(i.get("circunstancias") or []) or "—"]
     for i in INDICADORES_BERT],
    [3.0, 7.5, 5.5],
)
legenda(f"Tabela 9 — Os {len(INDICADORES_BERT)} indicadores que o classificador "
        "aprende, e as circunstâncias que cada um autoriza o modelo de linguagem "
        "a avaliar.")

quebra()

h(1, "Apêndice B — Indicadores")
p("Os indicadores dividem-se em dois tipos. Os de regra são calculados sobre a "
  "lista consolidada de vítimas e recalculados a cada alteração; os de texto "
  "resultam da classificação do conteúdo e são confirmados pela camada de modelo "
  "de linguagem.")
p()
tabela(
    ["Indicador", "Tipo", "Critério"],
    [[i["rotulo"], "regra" if i["tipo"] == "regra" else "texto", i["definicao"]]
     for i in INDICADORES],
    [4.2, 1.8, 10.0],
)
legenda("Tabela 10 — Indicadores e respectivos critérios.")

p("Os indicadores de regra são determinísticos e não dependem de modelo: "
  "vítima com menos de dezoito anos, vítima do gênero feminino, vítima agente "
  "de segurança, vítima com cargo ou candidatura política, e três ou mais "
  "mortos na mesma ocorrência. Por serem calculados sobre a lista final, "
  "atualizam-se sozinhos quando a situação de uma vítima muda — uma pessoa "
  "ferida que vem a falecer pode converter a ocorrência em chacina.")
p("Os indicadores de texto são os que exigem classificação e, portanto, os que "
  "entram na avaliação. O cadastro de um indicador de texto novo exige exemplos "
  "anotados para retreinamento, requisito incorporado ao fluxo da aplicação de "
  "gestão: a classe criada nasce sem exemplo e o sistema recusa o treino "
  "enquanto a classe mais rara não tiver o mínimo para ser representada nas três "
  "partições.")

quebra()

h(1, "Apêndice C — Diretrizes de anotação")
p("Resumo operacional para quem conduz a anotação.")
p()
tabela(
    ["Situação", "Decisão"],
    [
        ["Notícia relata tiro efetuado, com ou sem vítima atingida", "No escopo: anotar os indicadores que o texto sustentar"],
        ["Ameaça com arma de fogo, sem disparo", "Fora do escopo: nenhum indicador"],
        ["Apreensão de arma ou munição, sem confronto", "Fora do escopo: nenhum indicador"],
        ["Morte ou lesão por arma branca ou outro meio", "Fora do escopo: nenhum indicador"],
        ["Ferido em queda, correria ou mal-estar durante o fato", "Não é vítima; a notícia pode estar no escopo por outra pessoa atingida"],
        ["Ferido por estilhaço, vidro ou coronhada", "Não é vítima: o ferimento não é por disparo"],
        ["Bala perdida proveniente de confronto", "No escopo; a vítima recebe a circunstância bala_perdida"],
        ["Polícia presente e atirando no momento do fato", "Indicador acao_policial"],
        ["Polícia chega após o fato, apenas para atender", "NÃO caracteriza ação policial"],
        ["Policial atingido estando de folga", "NÃO é ação policial; é vitimização policial, calculada do tipo da vítima"],
        ["Pessoa com menos de 18 anos atingida", "Indicador derivado da idade; basta registrar a idade ou a palavra criança/adolescente"],
        ["Texto insuficiente para decidir um indicador", "Não marcar o indicador duvidoso; registrar a razão na observação"],
    ],
    [8.0, 8.0,],
)
legenda("Tabela 11 — Diretrizes para casos de fronteira.")

p("Três recomendações gerais. Primeiro, considera-se o que envolveu diretamente "
  "o disparo, no momento em que ocorreu, e não antecedentes do envolvido nem "
  "consequências posteriores que não constituam novo disparo. Segundo, em caso "
  "de dúvida genuína sobre um indicador, não o marcar e registrar a razão na "
  "observação: um rótulo incerto prejudica mais o treino do que a sua ausência. "
  "Terceiro, a notícia que não recebe indicador algum NÃO é descarte — ela é o "
  "exemplo negativo, e sem ela o classificador não aprende a recusar o que está "
  "fora do escopo.")

caminho = SAIDA / "TCC-Classificador-Violencia-Armada.docx"
doc.save(caminho)
print("gerado:", caminho)
print("tamanho:", round(caminho.stat().st_size / 1024), "KB")
print("paragrafos:", len(doc.paragraphs), "| tabelas:", len(doc.tables))
