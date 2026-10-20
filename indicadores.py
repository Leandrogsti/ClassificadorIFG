"""O catálogo de fenômenos e circunstâncias, e quem decide cada um.

Duas perguntas separadas, e confundi-las é o erro fácil aqui:

**Onde o resultado vai** (`classe`):
- `fenomeno` — para FENÔMENO OBSERVADO da ocorrência.
- `circunstancia` — para CIRCUNSTÂNCIA das vítimas.

**Quem decide** (`decidido_por`):
- `bert` — classe do classificador. São as sete que entram no treino e no
  corpus. O BERT marca e a LLM certifica.
- `llm` — não tem classe no BERT. A LLM decide, dentro do bloco de outra
  classe que a mencione (violência de gênero é avaliada nos blocos de
  mulheres_baleadas e lgbtqiapn).
- `regra` — calculado pelo sistema sobre a lista final de vítimas. Nem BERT nem
  LLM o marcam: vitimização policial sai de uma vítima com tipo `agente`, e
  chacina de três mortos. Ambos são dados que a LLM já preencheu na vítima, e
  derivá-los é exato, enquanto pedir ao modelo que os marque de novo só cria
  uma segunda resposta que pode contradizer a primeira.

As sete classes do BERT ficam numa lista só porque o modelo é um só: separar
em dois classificadores exigiria dois treinos, duas divisões de corpus e duas
matrizes de confusão para medir a mesma leitura do mesmo texto.

A marcação do BERT também é o que AUTORIZA a LLM a avaliar certas
circunstâncias. Feminicídio só é avaliado quando `mulheres_baleadas` foi
marcado; LGBTQIAPN+ e suicídio, quando as suas próprias classes foram. Sem
isso a LLM sairia procurando feminicídio em notícia sem mulher baleada, e o
prompt carregaria instrução que não se aplica.

Algumas classes têm ainda uma `regra` sobre a lista de vítimas. Ela não
substitui a classificação: confere o resultado depois que a LLM extraiu as
vítimas. Se a LLM registrar uma vítima de 14 anos, a ocorrência recebe
`crianca_adolescente_baleado` mesmo que o texto não tenha deixado a idade
explícita — e um ferido que morre pode transformar a ocorrência em chacina sem
que nenhuma notícia nova chegue.
"""
import json
from pathlib import Path

FONTE = Path(__file__).parent / "indicadores_fonte.json"


def carregar() -> list[dict]:
    return json.loads(FONTE.read_text(encoding="utf-8"))["indicadores"]


def salvar(indicadores: list[dict]) -> None:
    dados = json.loads(FONTE.read_text(encoding="utf-8"))
    dados["indicadores"] = indicadores
    FONTE.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def do_bert() -> list[dict]:
    """Só as classes que o BERT aprende. É esta lista que define o corpus, o
    treino e a marcação em produção."""
    return [i for i in carregar() if i.get("decidido_por") == "bert"]


def nomes() -> list[str]:
    """Os rótulos do classificador, nesta ordem. A ordem importa: ela define as
    posições do vetor multirrótulo e é gravada junto com os pesos do modelo."""
    return [i["nome"] for i in do_bert()]


def todos_os_nomes() -> list[str]:
    """Tudo que o catálogo conhece, inclusive o que o BERT não classifica."""
    return [i["nome"] for i in carregar()]


def fenomenos() -> list[dict]:
    return [i for i in carregar() if i.get("classe", "fenomeno") == "fenomeno"]


def circunstancias() -> list[dict]:
    return [i for i in carregar() if i.get("classe") == "circunstancia"]


def nomes_fenomeno() -> list[str]:
    return [i["nome"] for i in fenomenos()]


def nomes_decididos_pela_llm() -> list[str]:
    """Os que a LLM pode marcar: as classes do BERT (que ela certifica) mais os
    que só ela decide. Os de regra ficam de fora — se a LLM os devolvesse,
    teríamos uma chacina marcada no texto e desmentida pela contagem."""
    return [
        i["nome"] for i in carregar() if i.get("decidido_por") in ("bert", "llm")
    ]


def rotulos() -> dict[str, str]:
    return {i["nome"]: i.get("rotulo", i["nome"]) for i in carregar()}


def buscar(nome: str) -> dict | None:
    return next((i for i in carregar() if i["nome"] == nome), None)


def circunstancias_autorizadas(marcados: list[str] | None) -> list[str]:
    """As circunstâncias que a LLM pode marcar nas vítimas desta ocorrência.

    As condicionadas (feminicídio, LGBTQIAPN+, suicídio) só entram se a classe
    que as destrava foi marcada pelo BERT. As demais estão sempre disponíveis:
    bala perdida ou acidente não dependem de uma classe própria para serem
    reconhecidos na leitura.

    `marcados` como None significa que não há modelo treinado — aí não há como
    condicionar nada e todas ficam liberadas.
    """
    import vocabulario

    # As derivadas ficam de fora em QUALQUER caso, inclusive sem modelo
    # treinado: elas não dependem de classe marcada, dependem de o sistema
    # calculá-las sobre a lista de vítimas.
    todas = [
        valor for valor, _ in vocabulario.CIRCUNSTANCIAS
        if valor not in vocabulario.CIRCUNSTANCIAS_DERIVADAS
    ]
    if marcados is None:
        return todas

    presentes = set(marcados)
    return [
        valor for valor in todas
        if valor not in vocabulario.CIRCUNSTANCIAS_CONDICIONADAS
        or vocabulario.CIRCUNSTANCIAS_CONDICIONADAS[valor] in presentes
    ]


def _valor_do_campo(vitima: dict, campo: str):
    """O valor de `campo` na vítima, aceitando caminho com ponto.

    `vitimizacao_policial.status_agente` é o que permite uma regra olhar o dado
    da TABELA COMPLEMENTAR do agente, que vem pendurado na vítima como
    sub-dicionário. Sem isso não havia como escrever "agente em serviço", que é
    exatamente o que a chacina precisa distinguir.
    """
    atual = vitima
    for parte in (campo or "").split("."):
        if not isinstance(atual, dict):
            return None
        atual = atual.get(parte)
    return atual


def _casa(regra: dict, vitima: dict) -> bool:
    """A regra vale para ESTA vítima.

    Separado de `_avaliar_regra` porque há duas perguntas diferentes: "alguma
    vítima satisfaz?" (o indicador da ocorrência) e "esta vítima satisfaz?" (a
    exceção de uma contagem, e a circunstância de cada uma). Antes só a
    primeira existia, e por isso a contagem não sabia excluir ninguém.
    """
    # COMPOSIÇÃO POR "OU": basta uma das sub-regras valer. Existe porque um
    # mesmo fato chega ao banco por campos diferentes — a notícia que diz
    # "menino de 9 anos" preenche `idade`, e a que diz só "criança" preenche
    # `grupo_etario` com `idade` em null. Com uma regra só sobre `idade`, a
    # segunda não acionava o indicador: medido, e era falso negativo silencioso.
    if regra.get("ou"):
        return any(_casa(sub, vitima) for sub in regra["ou"])
    # COMPOSIÇÃO POR "E": todas as sub-regras valem na MESMA vítima. É o que
    # escreve "é agente E estava em serviço" — duas regras soltas valeriam com
    # um agente numa vítima e o serviço em outra.
    if regra.get("e"):
        return all(_casa(sub, vitima) for sub in regra["e"])

    campo = regra.get("campo")
    valor = regra.get("valor")
    operacao = regra.get("operacao")
    atual = _valor_do_campo(vitima, campo)

    if operacao == "campo_menor_que":
        if atual is None or str(atual).strip() == "":
            return False
        try:
            return float(atual) < float(valor)
        except (TypeError, ValueError):
            return False
    if operacao == "campo_igual":
        return (atual or "") == valor
    if operacao == "campo_em":
        return (atual or "") in set(regra.get("valores") or [])
    if operacao == "lista_contem":
        # Campos de marcação múltipla (tipo_vitima, circunstancias) chegam como
        # lista; comparar com == daria falso sempre que houvesse duas marcações.
        return valor in (atual or [])
    if operacao == "campo_preenchido":
        return bool(str(atual or "").strip())
    return False


def _elegiveis_da_contagem(regra: dict, vitimas: list[dict]) -> list[int]:
    """Os índices das vítimas que CONTAM para uma regra de contagem.

    `exceto` é quem não conta, avaliado vítima por vítima. É o que separa o
    algoz da vítima numa chacina: num confronto em que um agente em serviço
    morre junto com três pessoas, o agente não entra no grupo.
    """
    campo, valor = regra.get("campo"), regra.get("valor")
    excecao = regra.get("exceto")
    return [
        i for i, v in enumerate(vitimas)
        if (_valor_do_campo(v, campo) or "") == valor
        and not (excecao and _casa(excecao, v))
    ]


def _avaliar_regra(regra: dict, vitimas: list[dict]) -> bool:
    """A regra vale para a OCORRÊNCIA: alguma vítima a satisfaz, ou a contagem
    alcança o mínimo."""
    if regra.get("operacao") == "contagem_minima":
        return len(_elegiveis_da_contagem(regra, vitimas)) >= int(
            regra.get("minimo", 1)
        )
    if regra.get("ou"):
        return any(_avaliar_regra(sub, vitimas) for sub in regra["ou"])
    return any(_casa(regra, v) for v in vitimas)


def vitimas_de_chacina(vitimas: list[dict]) -> list[int]:
    """Os índices das vítimas que recebem a circunstância `chacina`.

    A circunstância é CALCULADA, não lida no texto, e sai da mesma regra que o
    fenômeno — é isso que impede uma chacina marcada em duas vítimas ou um
    fenômeno que nenhuma vítima sustenta.

    A regra, em palavras: **conta quem morreu, menos o agente que estava em
    serviço.** Três civis mortos é chacina. Dois civis mortos mais um agente
    morto é chacina dos três quando o agente estava de folga, aposentado ou sem
    status informado; quando ele estava em serviço, ele fica fora da contagem e
    sobram dois — e aí não há chacina.

    O agente em serviço fica fora porque numa ação policial ele está do outro
    lado do mesmo fato: se ele é baleado e mata três pessoas, a chacina é das
    três. Agente de folga é uma pessoa como as outras e conta normalmente.

    Devolve lista vazia quando a contagem não alcança o mínimo: sem chacina,
    nenhuma vítima recebe a marcação.
    """
    regra = (buscar("chacina") or {}).get("regra") or {}
    if regra.get("operacao") != "contagem_minima":
        return []
    elegiveis = _elegiveis_da_contagem(regra, vitimas)
    return elegiveis if len(elegiveis) >= int(regra.get("minimo", 1)) else []


def calcular(vitimas: list[dict], fenomenos_texto: list[str]) -> list[str]:
    """Os fenômenos observados da ocorrência, cada um pela sua fonte.

    Os de `regra` saem só da lista de vítimas: o que a LLM tenha dito sobre
    eles é ignorado, porque a contagem é exata e a leitura não. Os de `bert`
    entram pelo texto OU pela regra — o texto pode ser vago sobre a idade e a
    lista de vítimas não, e a lista muda depois, quando um ferido morre. Os de
    `llm` saem só do texto: são juízos de leitura, sem contagem que os
    derive.
    """
    do_texto = set(fenomenos_texto or [])
    resultado = []
    for indicador in fenomenos():
        decisor = indicador.get("decidido_por", "bert")
        regra = indicador.get("regra")
        por_regra = bool(regra) and _avaliar_regra(regra, vitimas)

        if decisor == "regra":
            entra = por_regra
        elif decisor == "llm":
            entra = indicador["nome"] in do_texto
        else:
            entra = indicador["nome"] in do_texto or por_regra

        if entra:
            resultado.append(indicador["nome"])
    return resultado


def origem(vitimas: list[dict], fenomenos_texto: list[str]) -> dict[str, str]:
    """Para cada fenômeno marcado, de onde ele veio: `texto`, `regra` ou
    `ambos`. É o que permite ao analista saber se discorda do modelo ou do
    cadastro das vítimas."""
    do_texto = set(fenomenos_texto or [])
    procedencia = {}
    for indicador in fenomenos():
        decisor = indicador.get("decidido_por", "bert")
        regra = indicador.get("regra")
        por_texto = decisor != "regra" and indicador["nome"] in do_texto
        por_regra = bool(regra) and _avaliar_regra(regra, vitimas)
        if decisor == "llm":
            por_regra = False
        if por_texto and por_regra:
            procedencia[indicador["nome"]] = "ambos"
        elif por_texto:
            procedencia[indicador["nome"]] = "texto"
        elif por_regra:
            procedencia[indicador["nome"]] = "regra"
    return procedencia
