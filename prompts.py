"""Os prompts da LLM, fora do código e editáveis pela tela.

Nenhum texto que vai para a LLM fica escondido na lógica do sistema. O que esta
aplicação manda numa chamada é a soma de cinco coisas, e as cinco são editáveis:

1. As seções do dicionário (`PAPEL`, `RECORTE`, `REGRAS-MESTRAS`), que vêm de
   `categorias_fonte.json` e são geradas em `dicionario_categorias.txt`.
2. A **skill** de cada indicador, em `indicadores_fonte.json`, editável em
   /gestao — e que só viaja quando o BERT marca aquele indicador.
3. O **prompt mestre**, aqui, que manda a LLM preencher a ocorrência e a lista
   de vítimas. Vai em TODA chamada.
4. O **formato de saída**, aqui: o JSON que a resposta tem de ter.
5. Os três **cabeçalhos** da tarefa de classes, aqui: um para quando o BERT
   marcou classes, um para quando ele não marcou nenhuma, e um para quando não
   há modelo treinado.

Os textos de 3 a 5 moravam dentro de `classificador.py`. Ficam neste módulo
porque mudar uma instrução da LLM não deveria exigir mexer em código — e porque
uma instrução escondida numa constante de Python é, na prática, uma instrução
que ninguém revisa.

SUBSTITUIÇÃO SEGURA. Os textos têm marcadores como `{generos}`, que o sistema
troca pelas listas fechadas do vocabulário. A troca é feita marcador por
marcador, e NUNCA com `str.format`: o formato de saída é um JSON cheio de
chaves literais, e `str.format` quebraria nelas. Marcador desconhecido é
deixado como está, para que um erro de digitação apareça no prompt em vez de
derrubar a classificação.
"""
import json
from pathlib import Path

FONTE = Path(__file__).parent / "prompts_fonte.json"

# Os marcadores que cada bloco aceita, e o que cada um significa. A tela mostra
# esta lista ao lado do campo: sem ela, quem edita não tem como saber que
# `{generos}` existe nem que apagá-lo deixa a LLM sem a lista fechada.
MARCADORES = {
    "prompt_mestre": {
        "generos": "lista fechada de gênero",
        "grupos": "lista fechada de grupo etário",
        "tipos": "lista fechada de tipo de vítima",
        "tabelas_vitima": "as tabelas complementares por vítima que se"
                          " aplicam: a do agente sempre, a do político só"
                          " quando `politico_baleado` foi marcado",
        "locais": "lista fechada de local",
        "circunstancias": "circunstâncias AUTORIZADAS para esta ocorrência,"
                          " que dependem do que o BERT marcou",
    },
    "tabela_agente": {
        "corporacoes": "lista fechada de corporação do agente",
        "cargos_agente": "lista fechada de patente/cargo do agente",
        "status_agente": "lista fechada de situação de serviço do agente",
    },
    "tabela_politico": {},
    "formato": {},
    "classes_marcadas": {},
    "sem_classificacao": {
        "classes": "os nomes das classes que o classificador avaliou",
    },
    "sem_modelo": {},
}

ORDEM = [
    "prompt_mestre", "tabela_agente", "tabela_politico", "formato",
    "classes_marcadas", "sem_classificacao", "sem_modelo",
]


def carregar() -> dict:
    """Os blocos vigentes. Bloco ausente do arquivo cai no padrão de fábrica,
    para que um arquivo editado pela metade não derrube a classificação."""
    dados = {}
    if FONTE.exists():
        try:
            dados = json.loads(FONTE.read_text(encoding="utf-8")).get("blocos") or {}
        except json.JSONDecodeError:
            dados = {}

    blocos = {}
    for nome in ORDEM:
        padrao = PADROES[nome]
        gravado = dados.get(nome) or {}
        blocos[nome] = {
            **padrao,
            "texto": gravado.get("texto") or padrao["texto"],
            "editado": bool(gravado.get("texto"))
            and gravado["texto"] != padrao["texto"],
        }
    return blocos


def salvar(textos: dict[str, str]) -> None:
    """Grava só o que difere do padrão. O bloco igual ao de fábrica não é
    gravado, de modo que uma melhoria futura no padrão chegue a quem não o
    personalizou."""
    blocos = {}
    for nome, texto in textos.items():
        if nome not in PADROES:
            continue
        limpo = (texto or "").strip()
        if limpo and limpo != PADROES[nome]["texto"].strip():
            blocos[nome] = {"texto": limpo}

    FONTE.write_text(
        json.dumps(
            {
                "_leia_me": (
                    "Prompts da LLM. Editável aqui ou pela tela /prompts."
                    " Bloco ausente usa o padrão de fábrica de prompts.py."
                    " Os marcadores {assim} são trocados pelo sistema — ver"
                    " MARCADORES em prompts.py."
                ),
                "blocos": blocos,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def restaurar(nome: str) -> None:
    """Devolve um bloco ao padrão de fábrica, apagando a personalização dele."""
    blocos = {
        chave: {"texto": bloco["texto"]}
        for chave, bloco in carregar().items()
        if bloco["editado"] and chave != nome
    }
    salvar({chave: bloco["texto"] for chave, bloco in blocos.items()})


def texto(nome: str, **valores) -> str:
    """O texto de um bloco, com os marcadores trocados.

    Troca marcador por marcador, nunca com `str.format`: `formato` é um JSON
    cheio de `{` literais e `str.format` morreria neles.
    """
    bruto = carregar()[nome]["texto"]
    for marcador, valor in valores.items():
        bruto = bruto.replace("{" + marcador + "}", str(valor))
    return bruto


def marcadores_faltando(nome: str, texto_novo: str) -> list[str]:
    """Marcadores que o bloco aceita e que o texto editado não usa.

    Não impede o salvamento: apagar `{circunstancias}` é uma escolha legítima
    (a LLM passa a não receber a lista fechada), mas é uma escolha que precisa
    ser consciente, e a tela avisa.
    """
    return [
        marcador for marcador in MARCADORES.get(nome, {})
        if "{" + marcador + "}" not in texto_novo
    ]


# --------------------------------------------------------------------------
# Padrões de fábrica
# --------------------------------------------------------------------------
# Estes são os textos que o sistema usa quando ninguém os editou. Vinham de
# constantes em classificador.py; o conteúdo é o mesmo, palavra por palavra.

PADROES = {
    "prompt_mestre": {
        "rotulo": "Prompt mestre — ocorrência e vítimas",
        "descricao": (
            "Vai em TODA chamada, antes dos blocos de indicador. É ele que manda"
            " a LLM montar a lista de vítimas e preencher os campos de cada uma:"
            " nome, status, gênero, idade, tipo, local, data de morte e"
            " divergências entre fontes — cor/raça não, que é do analista, e a"
            " FAIXA ETÁRIA só quando a notícia não dá o número, porque o"
            " sistema a deriva da idade. Também é aqui que estão"
            " as regras de contagem que impedem somar as vítimas de duas"
            " notícias sobre o mesmo fato."
        ),
        "texto": """
# TAREFA 2: LISTA DE VÍTIMAS

Você recebe TODAS as notícias desta mesma ocorrência. Elas DEVERIAM descrever o
MESMO fato sob fontes diferentes. Monte a lista de PESSOAS DISTINTAS atingidas,
refazendo-a do zero. NUNCA some as vítimas de notícias diferentes.

ANTES DE COMEÇAR, confira se são mesmo o mesmo fato. Data, local e vítimas
precisam bater. Se uma das notícias descrever um fato DIFERENTE — outro dia,
outro bairro, outras pessoas —, NÃO misture as vítimas dela: deixe-a de fora da
lista e escreva em `divergencias` qual notícia é, e por que ela parece ser de
outro caso. Anexar a notícia errada a uma ocorrência é engano de digitação
comum, e só você vê o texto das duas.

Regras de contagem:
1. Mesma pessoa citada em várias notícias conta UMA vez. O nome identifica a
   pessoa; sem nome, compare idade, gênero e tipo de vítima.
2. Sem nomes, cruze circunstância e status. O total NUNCA pode ser menor que
   o maior número citado em uma única notícia: "um homem baleado" em um jornal
   e "dois homens baleados" em outro resultam em 2 vítimas, não 3.
3. CADA CAMPO É PREENCHIDO PELA FONTE QUE O TEM. As notícias não trazem as
   mesmas informações: uma dá só a patente, outra a idade, outra o nome. A
   ficha da vítima é a SOMA de todas elas. Antes de dar um campo por vazio,
   procure-o em TODAS as notícias.
   "Ainda não há informações sobre a identidade", "não foi identificado" e
   "não divulgou o nome" são AUSÊNCIA naquela fonte, não um fato sobre a
   vítima: elas NÃO cancelam o nome que outra notícia traz. O mesmo vale para
   idade, patente, unidade e qualquer outro campo. Use null só quando NENHUMA
   das notícias informar.
4. Quando as fontes divergirem de verdade — dizem coisas diferentes sobre o
   mesmo campo, como duas idades ou dois status —, escolha o valor da fonte
   mais específica e registre a divergência com o trecho de cada fonte no
   campo "divergencia" da vítima. Divergência é conflito entre dois valores,
   nunca um valor contra um silêncio.
5. ESCOPO: este banco registra SÓ disparo de arma de fogo, e vítima é só
   quem foi ATINGIDO POR DISPARO — "status" `morta` ou `ferida`. NÃO entram na
   lista: ferido em queda, correria ou atropelamento durante o fato; quem passou
   mal, desmaiou ou teve crise nervosa; atingido por estilhaço, vidro, pedra,
   coronhada, faca ou qualquer arma que não seja de fogo; e quem estava na cena
   e escapou sem ser atingido. Se NINGUÉM foi atingido por disparo, devolva
   `vitimas` vazia e `vitimas_baleadas` igual a `"N"` — mesmo que haja feridos
   por outra causa.
6. Não invente vítimas: só entra quem o texto descreve como atingida.
7. Em "nome", escreva o nome próprio como QUALQUER uma das notícias o traz,
   inclusive sobrenome ou como-é-conhecido vindo junto da patente: "o sargento
   Pacheco" dá `"Pacheco"`. Use null só quando NENHUMA notícia nomear a pessoa
   — e NUNCA descreva a pessoa nesse campo. "Adolescente de 16 anos", "Homem de
   24 anos" e "Vítima não identificada" são INVÁLIDOS: idade e gênero já têm
   campos próprios, e um nome inventado faz a mesma pessoa ser contada duas
   vezes quando outra notícia a descreve de outro jeito.
8. Em "fonte", liste TODAS as notícias que descreveram esta pessoa, não só a
   primeira: "Notícia 1, Notícia 3". É o que mostra ao analista de onde veio
   cada pedaço da ficha.
9. Em "data_morte", preencha a data em que a vítima morreu quando a notícia a
   informar, no formato AAAA-MM-DD. Use null quando a vítima está ferida ou
   quando o texto não traz a data — NUNCA repita a data da ocorrência como se
   fosse a data da morte, porque um ferido pode morrer dias depois e é essa
   diferença que o registro precisa guardar.

Preencha "genero" com UM destes valores:
{generos}

Em "idade", o número que a notícia informar. "grupo_etario" só precisa ser
preenchido QUANDO NÃO HOUVER NÚMERO — a notícia diz "uma criança", "um
adolescente", "um idoso" e nada mais. Havendo número, o sistema deriva a faixa
dele sozinho e sobrescreve o que vier aqui; não gaste atenção nisso.
{grupos}

Preencha "tipo_vitima" com UMA OU MAIS destas marcações:
{tipos}
É marcação múltipla: um policial que também é vereador é `agente` e `politico`.
Toda vítima que não for agente nem político é `civil`. Marque `agente` quando o
texto identificar a pessoa como policial, guarda, agente penitenciário, militar
ou outro agente público de segurança — em serviço, de folga ou aposentado. Na
ausência de indicação, `civil`. Marque `politico` só quem tem cargo eletivo,
candidatura ou pré-candidatura: assessor, dirigente partidário, liderança
comunitária sem cargo e parente de político sem função própria são `civil`.

{tabelas_vitima}

Em "descricao", escreva SÓ o que não tem campo próprio. Patente, corporação,
unidade, situação de serviço, partido, cargo, idade, gênero e local têm cada um
o seu campo: repeti-los na descrição faz o dado ficar invisível para o sistema,
que lê os campos e não a prosa. A descrição é para o que sobra — a parte do
corpo atingida, o que a pessoa estava fazendo, o detalhe que nenhum campo
guarda.

Preencha "local" com UM destes valores:
{locais}
Só escolha um local quando o texto o disser. Rua, esquina, "nas proximidades" e
endereço solto não são nenhum destes: a resposta certa aí é `nao_identificado`,
e o analista completa depois na tela de Vítimas.

Preencha "circunstancias" com UMA OU MAIS da lista AUTORIZADA desta ocorrência:
{circunstancias}
TODA CIRCUNSTÂNCIA É DE UMA VÍTIMA, e vítima é só quem foi ATINGIDO. Quase
toda notícia aqui relata disparo, e em muitas ninguém é atingido: nessas não há
vítima, logo não há circunstância nenhuma a marcar. Não use a circunstância
para descrever o fato — ela descreve a pessoa.

Três delas valem em qualquer ocorrência e por isso o critério vem aqui:
- `bala_perdida` — a vítima estava FORA DA AÇÃO. Não basta não ser o alvo: ela
  não participava do fato de modo algum — passava na rua, estava em casa,
  dentro de um ônibus, trabalhando. Quem estava no confronto e foi atingido por
  disparo que não era para ele NÃO é bala perdida, porque estava na ação.
- `trajeto_escolar` — a vítima foi baleada indo ou voltando da escola, ou no
  entorno dela em horário de entrada e saída. Vale para estudante de qualquer
  idade, não só criança.
- `acidente` — a vítima foi atingida por disparo involuntário: manuseio
  indevido, limpeza da arma, queda da arma, criança que achou a arma. Se houve
  intenção de atingir alguém, não é acidente, por mais que o resultado tenha
  sido diferente do pretendido.
`chacina` NÃO está nesta lista e você não a marca: o sistema a calcula sobre a
lista de vítimas que você montar — conta os mortos, menos o agente de segurança
que estava em serviço. Para que a conta saia certa, só o que você preenche
importa: o `status` de cada vítima e, no agente, o `status_agente`.

Use `nao_se_aplica` sozinho quando nenhuma das demais descrever o caso — ele
nunca convive com outra marcação. NÃO use circunstância fora desta lista, mesmo
que o texto sugira: as que faltam dependem de uma classe que o BERT não marcou,
ou são calculadas pelo sistema a partir da lista de vítimas.
""",
    },
    "tabela_agente": {
        "rotulo": "Tabela complementar — vítima agente de segurança",
        "descricao": (
            "Entra no prompt mestre, no lugar de `{tabelas_vitima}`. Desde"
            " 2026-10-09 viaja SÓ quando o BERT marca `vitimizacao_policial`,"
            " que virou classe justamente para condicioná-la: são ~940 tokens"
            " com as listas fechadas, a maior peça condicionável do prompt."
            " `acao_policial` não serviria de condição, porque policial"
            " baleado de folga é vitimização policial e não é ação policial."
        ),
        "texto": """Quando a vítima tiver `agente`, é OBRIGATÓRIO acrescentar DENTRO dela a chave
`vitimizacao_policial`, com os quatro campos. Não deixe de mandá-la: o sistema
não tem outro lugar de onde tirar esses dados.
```json
{"corporacao": "policia_militar", "cargo": "sargento", "status_agente": "fora_de_servico", "unidade": "41º BPM (Irajá)"}
```
`corporacao` — UM destes valores:
{corporacoes}
`cargo` — UM destes valores (é a patente ou o cargo, não a função do dia):
{cargos_agente}
`status_agente` — UM destes valores:
{status_agente}
"De folga", "à paisana" e "no horário de descanso" são `fora_de_servico`.
"Reformado", "da reserva" e "exonerado" são `aposentado_exonerado`.
`unidade` — o batalhão, a delegacia ou o grupamento citado, em texto livre e
como a notícia o traz: "41º BPM", "15ª DP", "BOPE", "GAT". Use null só quando a
notícia não citar nenhum.

LEIA O TEXTO INTEIRO antes de desistir de um campo: "sargento do 41º BPM em
folga" preenche três deles de uma vez. Use `nao_identificado` apenas no campo
que a notícia realmente não informar.""",
    },
    "tabela_politico": {
        "rotulo": "Tabela complementar — vítima política",
        "descricao": (
            "Entra no prompt mestre, no lugar de `{tabelas_vitima}`, e SÓ"
            " quando o classificador marcou `politico_baleado`: sem a classe"
            " não há vítima política, e são 296 tokens que deixam de viajar."
        ),
        "texto": """Quando a vítima tiver `politico`, acrescente DENTRO dela a chave
`politico_baleado`:
```json
{"partido": "sigla como a notícia a traz, ou \\"Não identificado\\"", "cargo": "vereador|prefeito|deputado_estadual|deputado_federal|senador|governador|presidente|ministro|secretario|vice_prefeito|vice_governador|vice_presidente", "status_politico": "no_mandato|candidato|pre_candidato"}
```""",
    },
    "formato": {
        "rotulo": "Formato de saída (JSON)",
        "descricao": (
            "O JSON que a resposta tem de ter. Mexer aqui é o mais arriscado dos"
            " cinco: o sistema lê estas chaves por nome, e renomear uma faz o"
            " campo correspondente deixar de ser preenchido. Acrescentar"
            " instrução em texto é seguro; trocar nome de chave, não."
        ),
        "texto": """
# FORMATO DE SAÍDA

Retorne SOMENTE este JSON, sem texto antes ou depois:
```json
{
  "fenomenos": ["classes de FENÔMENO que você confirmou"],
  "vitimas_baleadas": "S",
  "vitimas": [
    {
      "nome": "Nome da pessoa ou null",
      "status": "morta|ferida",
      "genero": "um valor da lista fechada",
      "grupo_etario": "um valor da lista fechada",
      "idade": 30,
      "tipo_vitima": ["civil"],
      "circunstancias": ["nao_se_aplica"],
      "local": "um valor da lista fechada",
      "descricao": "observação individual sobre esta vítima, ou null",
      "data_morte": "AAAA-MM-DD ou null",
      "fonte": "Notícia 1",
      "divergencia": "o que cada fonte diz, quando houver conflito, ou null",
      "vitimizacao_policial": {"corporacao": "...", "cargo": "...", "status_agente": "...", "unidade": "..."},
      "politico_baleado": {"partido": "...", "cargo": "...", "status_politico": "..."}
    }
  ],
  "divergencias": ["conflitos entre fontes que o analista precisa decidir"],
  "nao_sustentados": ["classes marcadas que o texto não sustenta"]
}
```
- `fenomenos`: só os de classe FENÔMENO, e só os que o classificador já marcou.
  Suicídio e LGBTQIAPN+ NÃO entram aqui — são circunstâncias e vão dentro da
  vítima a que se aplicam.
- `nao_sustentados`: as classes marcadas para as quais você NÃO achou base no
  texto. Elas continuam marcadas; esta lista só avisa que os campos delas
  ficaram em branco de propósito, e não por esquecimento.
- `idade`: número inteiro ou null quando a notícia não informar.
- `vitimas`: lista vazia e `vitimas_baleadas` igual a `"N"` se ninguém foi
  atingido.
- `vitimizacao_policial`: obrigatória em toda vítima com `agente` no
  `tipo_vitima`, e ausente nas outras. `politico_baleado`: idem para
  `politico`. Elas ficam DENTRO da vítima, não na raiz.
- As tabelas complementares descritas nos blocos abaixo acrescentam chaves na
  RAIZ deste JSON. Só inclua as que forem pedidas.
""",
    },
    "classes_marcadas": {
        "rotulo": "Cabeçalho — o BERT marcou classes",
        "descricao": (
            "Abre a lista de blocos quando o classificador marcou ao menos uma"
            " classe. É o texto que proíbe a LLM de procurar classes novas e que"
            " explica o que fazer quando o texto não sustenta uma classe"
            " marcada."
        ),
        "texto": (
            "# TAREFA 1: APROFUNDAR AS CLASSES JÁ CLASSIFICADAS\n"
            "O classificador JÁ decidiu quais classes esta notícia tem, e a"
            " decisão é dele. As classes abaixo estão marcadas; as que não"
            " aparecem aqui não se aplicam.\n"
            "NÃO procure classes novas e NÃO acrescente nenhuma a `fenomenos`: o"
            " que vier fora desta lista é ignorado pelo sistema. Se o texto"
            " deixar outra classe evidente, escreva isso em `divergencias` — o"
            " analista revisa e corrige o classificador, que é como ele"
            " aprende.\n"
            "Sua tarefa em cada classe abaixo é EXTRAIR do texto o que a"
            " instrução dela pedir. Quando o texto não sustentar a classe, não"
            " invente nada: deixe as chaves dela de fora e registre o nome dela"
            " em `nao_sustentados`. A classe permanece marcada; só os campos"
            " dela ficam em branco, para o analista preencher à mão."
        ),
    },
    "sem_classificacao": {
        "rotulo": "Pré-prompt padrão — nenhuma classe marcada",
        "descricao": (
            "O comportamento padrão da LLM quando o classificador leu a notícia"
            " e não encontrou nenhum indicador. Nenhum bloco de indicador viaja"
            " nessa chamada: é só este texto, o prompt mestre e o formato. A LLM"
            " continua preenchendo as informações gerais e as vítimas."
        ),
        "texto": (
            "# TAREFA 1: CLASSES\n"
            "O classificador leu esta notícia e NÃO encontrou nenhuma das"
            " classes monitoradas. Isso costuma significar que ela está fora do"
            " escopo (sem disparo de arma de fogo) ou que nenhum dos perfis se"
            " aplica.\n"
            "Essa decisão é dele, não sua: devolva `fenomenos` VAZIO. NÃO marque"
            " nenhuma classe, mesmo que o texto pareça sustentá-la — se você"
            " discordar, escreva o motivo em `divergencias` e o analista decide."
            " Preencha o resto normalmente: as vítimas e os dados"
            " gerais da notícia continuam sendo sua tarefa.\n"
            "Para referência, as classes que o classificador avaliou e descartou"
            " são: {classes}."
        ),
    },
    "sem_modelo": {
        "rotulo": "Cabeçalho — não há modelo treinado",
        "descricao": (
            "A exceção do fluxo. Sem modelo treinado não existe classificação"
            " prévia, e recusar a leitura da LLM deixaria o sistema sem"
            " classificar nada — então aqui, e só aqui, ela classifica. Nesta"
            " chamada viajam os blocos de TODAS as classes, e é por isso que ela"
            " é a mais cara."
        ),
        "texto": (
            "# TAREFA 1: CLASSES\n"
            "Não há marcação prévia do BERT para esta notícia. Avalie as classes"
            " abaixo e marque as que o texto sustentar — podem ser várias, ou"
            " nenhuma."
        ),
    },
}
