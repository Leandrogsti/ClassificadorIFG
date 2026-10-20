"""Pré-processamento, marcação pelo BERT e classificação final pela LLM.

O encadeamento é este: o BERT lê a notícia e marca as suas classes; a marcação
vira um atributo da ocorrência; e só então a LLM é chamada, recebendo APENAS os
prompts das classes marcadas. É ela quem certifica cada classe e preenche o que
a marcação sozinha não diz — a ficha de cada vítima, a circunstância dela e as
tabelas complementares de ação policial e crime organizado.

O QUE O SISTEMA REGISTRA: fenômeno observado da ocorrência e circunstância da
vítima, e nada além disso. Não há motivação: ela foi retirada do sistema em
2026-10-06, e com ela as quinze categorias que o prompt carregava.

A marcação também AUTORIZA: feminicídio só é avaliado quando o BERT marcou
`mulheres_baleadas`, LGBTQIAPN+ e suicídio só quando as suas próprias classes
foram marcadas. A instrução de cada uma só viaja quando pode ser usada.

Mandar as dez skills e o formato de todas as tabelas complementares em toda
chamada seria pagar por instrução que não se aplica. Com a marcação do BERT, o
prompt leva um núcleo fixo e dois ou três blocos — e o núcleo, por ser sempre
igual e vir primeiro, é o que a API consegue manter em cache entre chamadas.

O ORÇAMENTO É APERTADO e isso explica várias decisões deste arquivo. O plano da
Groq em uso dá 8.000 tokens POR MINUTO. Medido sem o texto da notícia (a tabela
completa está no README):

    sem modelo treinado      catálogo inteiro viaja    ~6.750 tokens
    duas classes marcadas    só os blocos delas        ~4.620 tokens
    nenhuma classe marcada   só o núcleo fixo          ~3.540 tokens

As duas últimas linhas NÃO são o mesmo caso: "sem modelo treinado" é não haver
quem marque, e então tudo viaja; "nenhuma classe marcada" é o BERT ter lido e
descartado, e aí viaja o mínimo. Cada bloco que não viaja é folga para o modelo
raciocinar, e raciocinar é o que estabiliza a lista de vítimas.

A cada notícia adicionada, a LLM relê TODAS as notícias da ocorrência juntas e
refaz a lista de vítimas do zero. Isso custa mais tokens do que processar só a
notícia nova, mas é o que impede a soma indevida: se um jornal diz "um homem
baleado" e outro diz "dois homens baleados", o total é 2, nunca 3.
"""
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import (
    APIConnectionError, APIError, APIStatusError, APITimeoutError, Groq
)

import auditoria
import indicadores
import inferencia
import preprocessar_noticia
import prompts
import vocabulario

load_dotenv()

RAIZ = Path(__file__).parent
DICIONARIO = RAIZ / "dicionario_categorias.txt"


class ErroClassificacao(RuntimeError):
    pass


def _recado_da_api(erro: APIError, marcados: list[str] | None,
                   conteudo: str = "") -> str:
    """Traduz a falha da API no que o analista pode fazer a respeito.

    O estouro de tokens ganha recado próprio, e ele separa as DUAS causas que a
    Groq devolve com o mesmo 413. O teto é por MINUTO, então uma chamada que
    cabe sozinha é recusada igual se outra acabou de gastar a cota — e aí o
    conselho certo é esperar, não encolher nada. Dizer só "erro 413" manda
    procurar no lugar errado; dizer "o prompt é grande demais" quando não é
    manda reescrever um prompt que está certo.
    """
    detalhe = getattr(erro, "message", None) or str(erro)
    if isinstance(erro, APIStatusError) and erro.status_code in (413, 429):
        tamanho = (len(montar_prompt_sistema(marcados)) + len(conteudo)) // 4
        return (
            f"A API recusou a chamada por limite de tokens. Esta requisição tem"
            f" cerca de {tamanho} tokens, e o teto do plano é POR MINUTO — duas"
            f" causas possíveis: ou ela não cabe sozinha no teto, e aí é preciso"
            f" reduzir o número de notícias da ocorrência; ou a cota do minuto"
            f" já foi gasta por uma classificação anterior, e aí basta repetir"
            f" daqui a pouco. Compare o número acima com o limite que a API"
            f" informa: {detalhe}"
        )
    if "json_validate_failed" in detalhe:
        return (
            "O modelo não conseguiu fechar o JSON da resposta. Não é limite de"
            " tokens nem falha de rede: é a geração em si que saiu malformada,"
            " o que acontece de vez em quando com ocorrências de muitas"
            " notícias. Repetir a classificação costuma resolver; se insistir,"
            " reduza o número de notícias desta ocorrência. O texto recebido"
            f" está gravado e nada se perdeu. Resposta da API: {detalhe}"
        )
    return f"A API recusou a chamada: {detalhe}"


def _cliente() -> tuple[Groq, str]:
    chave = os.getenv("GROQ_API_KEY")
    if not chave or chave.startswith("cole_sua"):
        raise ErroClassificacao(
            "GROQ_API_KEY não configurada. Edite o arquivo .env na raiz do projeto."
        )
    modelo = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    if "prompt-guard" in modelo.lower():
        raise ErroClassificacao(
            "Prompt Guard detecta ataques, não classifica notícias. "
            "Configure um modelo generativo no .env."
        )
    return Groq(api_key=chave), modelo


# --------------------------------------------------------------------------
# Montagem do prompt
# --------------------------------------------------------------------------

def _secoes_do_dicionario() -> dict[str, str]:
    """Quebra o dicionário gerado nas suas seções `# TÍTULO`.

    O arquivo é gerado por gerar_dicionario.py a partir de categorias_fonte.json
    e os títulos são fixos. Separar as seções é o que permite enviar as regras
    de desempate sempre e o catálogo completo das 15 categorias só quando não há
    marcação do BERT para estreitar a escolha.
    """
    secoes: dict[str, str] = {}
    titulo = None
    corpo: list[str] = []
    for linha in DICIONARIO.read_text(encoding="utf-8").splitlines():
        if linha.startswith("# "):
            if titulo:
                secoes[titulo] = "\n".join(corpo).strip()
            titulo = linha[2:].strip()
            corpo = []
        else:
            corpo.append(linha)
    if titulo:
        secoes[titulo] = "\n".join(corpo).strip()
    return secoes


def _secao(secoes: dict[str, str], prefixo: str) -> str:
    """Busca por prefixo porque um dos títulos carrega o número de categorias
    ("AS 15 CATEGORIAS..."), que muda quando alguém edita a fonte."""
    for titulo, corpo in secoes.items():
        if titulo.startswith(prefixo):
            return f"# {titulo}\n{corpo}"
    return ""


def _lista_fechada(pares) -> str:
    return "\n".join(f"- `{valor}` — {rotulo}" for valor, rotulo in pares)


def _bloco_do_indicador(indicador: dict) -> str:
    """O prompt de uma classe: a sua skill, as circunstâncias que ela destrava
    e o JSON extra que exige. É esta unidade que entra ou não entra na
    chamada."""
    partes = [
        f"## {indicador['nome']} — {indicador['rotulo']}",
        indicador["definicao"],
        indicador.get("skill", ""),
    ]

    destravadas = indicador.get("circunstancias") or []
    condicionadas = [
        c for c in destravadas if c in vocabulario.CIRCUNSTANCIAS_CONDICIONADAS
    ]
    if condicionadas:
        nomes = ", ".join(
            f"`{c}` ({vocabulario.ROTULOS_CIRCUNSTANCIA[c]})" for c in condicionadas
        )
        partes.append(
            f"Esta classe é a ÚNICA que autoriza a circunstância {nomes}. Sem ela"
            " marcada, essa circunstância não pode aparecer em vítima nenhuma."
        )

    # Uma classe pode exigir mais de uma tabela: ação policial pede a da
    # ocorrência e a do agente ferido, porque vitimização policial deixou de
    # ter bloco próprio — ela é derivada do tipo da vítima.
    for saida in indicador.get("saidas") or []:
        onde = (
            "dentro de CADA vítima a que se aplica"
            if saida.get("por_vitima")
            else "na raiz do JSON"
        )
        partes.append(
            f"TABELA COMPLEMENTAR — acrescente a chave `{saida['chave']}` {onde}:\n"
            f"```json\n{saida['formato']}\n```\n{saida.get('notas', '')}".strip()
        )

    # Título e definição sozinhos não dizem à LLM o que FAZER: eles descrevem
    # um indicador que o BERT marca do texto ou que uma regra deriva da lista de
    # vítimas. Nesses casos o bloco não viaja — criança/adolescente sai da
    # idade, vitimização policial e chacina saem da contagem, e político já é
    # pedido no prompt mestre pelo `tipo_vitima`. São ~245 tokens por chamada
    # que não mudavam nenhuma resposta.
    #
    # Os derivados de regra continuam NOMEADOS na nota de `_nota_dos_derivados`,
    # que é o que impede a LLM de tentar marcá-los por conta.
    # `partes[0]` é o título e `partes[1]` a definição; tudo a partir de 2 é
    # instrução — skill, circunstância destravada, tabela complementar.
    if not [p for p in partes[2:] if p]:
        return ""

    return "\n\n".join(p for p in partes if p).strip()


def _nota_dos_derivados() -> str:
    """Os fenômenos que a LLM NÃO marca precisam ser nomeados, senão o modelo
    tenta marcá-los por conta e a contagem passa a ter duas respostas."""
    de_regra = [
        i for i in indicadores.carregar() if i.get("decidido_por") == "regra"
    ]
    if not de_regra:
        return ""
    linhas = "\n".join(
        f"- `{i['nome']}` ({i['rotulo']}) — {i['definicao']}" for i in de_regra
    )
    return (
        "# FENÔMENOS QUE VOCÊ NÃO MARCA\n"
        "Estes são calculados pelo sistema a partir da lista de vítimas que você"
        " montar. NUNCA os inclua em `fenomenos`; preencha a vítima corretamente"
        " e eles aparecem sozinhos.\n" + linhas
    )


# Os prompts mestre, o formato de saída e os três cabeçalhos da tarefa de
# classes NÃO moram mais aqui: estão em prompts.py, gravados em
# prompts_fonte.json e editáveis na tela /prompts. Instrução de LLM presa
# numa constante de Python é instrução que ninguém revisa.


def _tabelas_da_vitima(marcados: list[str] | None) -> str:
    """As tabelas complementares por vítima que se aplicam a esta ocorrência.

    A do POLÍTICO só viaja quando o BERT marcou `politico_baleado`: sem a
    classe não há vítima política, e são 296 tokens de instrução sem uso. Sem
    modelo treinado (`marcados is None`) ela viaja, porque aí não há marcação
    nenhuma para condicionar.

    A do AGENTE segue a mesma regra, condicionada a `vitimizacao_policial` —
    que virou classe do BERT em 2026-10-09 justamente por isto. Antes ela
    viajava sempre, porque `acao_policial` não servia de condição: policial
    baleado de folga é vitimização policial e não é ação policial. São ~940
    tokens com as listas fechadas, a maior peça condicionável do prompt.
    """
    marcadas = set(marcados or [])
    partes = []
    if marcados is None or "vitimizacao_policial" in marcadas:
        partes.append(prompts.texto(
            "tabela_agente",
            corporacoes=_lista_fechada(vocabulario.CORPORACOES),
            cargos_agente=_lista_fechada(vocabulario.CARGOS_AGENTE),
            status_agente=_lista_fechada(vocabulario.STATUS_AGENTE),
        ))
    if marcados is None or "politico_baleado" in marcadas:
        partes.append(prompts.texto("tabela_politico"))
    return "\n\n".join(x for x in partes if x)


def montar_nucleo(marcados: list[str] | None = None) -> str:
    """A parte do prompt que quase não muda, e por isso vem primeiro: é o
    prefixo que a API consegue manter em cache. Leva o papel, o recorte de
    análise e as listas fechadas das vítimas.

    Não leva mais nada sobre motivação: o sistema registra FENÔMENO OBSERVADO e
    CIRCUNSTÂNCIA, e nada além disso. O catálogo das quinze categorias, os dois
    níveis e as regras de desempate saíram daqui porque não havia mais onde
    guardar a resposta deles.
    """
    secoes = _secoes_do_dicionario()
    autorizadas = indicadores.circunstancias_autorizadas(marcados)
    return "\n\n".join(
        parte
        for parte in [
            _secao(secoes, "PAPEL"),
            _secao(secoes, "RECORTE DE ANÁLISE"),
            _nota_dos_derivados(),
            prompts.texto(
                "prompt_mestre",
                generos=_lista_fechada(vocabulario.GENEROS),
                grupos=_lista_fechada(vocabulario.GRUPOS_ETARIOS),
                tipos=_lista_fechada(vocabulario.TIPOS_VITIMA),
                tabelas_vitima=_tabelas_da_vitima(marcados),
                locais=_lista_fechada(vocabulario.LOCAIS_VITIMA),
                circunstancias="\n".join(
                    f"- `{valor}` — {vocabulario.ROTULOS_CIRCUNSTANCIA[valor]}"
                    for valor in autorizadas
                ),
            ),
            prompts.texto("formato"),
        ]
        if parte
    )


def montar_prompt_indicadores(marcados: list[str] | None) -> str:
    """Os blocos das classes que o BERT marcou, e só eles.

    Três situações, cada uma com o seu cabeçalho editável em /prompts:

    - `marcados` com classes: vão os blocos delas e o cabeçalho que proíbe a LLM
      de procurar outras.
    - `marcados` vazio: o classificador leu e não achou nada. Nenhum bloco de
      indicador viaja, só o pré-prompt padrão.
    - `marcados` como None: não há modelo treinado. Aí não há como estreitar
      nada e vão TODOS os blocos. Funciona, mas é a chamada cara — é exatamente
      o que a marcação do BERT existe para evitar.
    """
    catalogo = indicadores.carregar()

    if marcados is None:
        blocos = [_bloco_do_indicador(i) for i in catalogo]
        cabecalho = prompts.texto("sem_modelo")
        return "\n\n".join([cabecalho, *blocos])

    escolhidos = [i for i in catalogo if i["nome"] in set(marcados)]

    if not escolhidos:
        return prompts.texto(
            "sem_classificacao",
            classes=", ".join(f"`{i['nome']}`" for i in catalogo),
        )

    return "\n\n".join(
        [prompts.texto("classes_marcadas"),
         *(_bloco_do_indicador(i) for i in escolhidos)]
    )


def montar_prompt_sistema(marcados: list[str] | None = None) -> str:
    """O prompt inteiro, para inspeção e para os testes. Em produção as duas
    partes vão como mensagens separadas, para que o núcleo seja cacheável."""
    return montar_nucleo(marcados) + "\n\n" + montar_prompt_indicadores(marcados)


# --------------------------------------------------------------------------
# Normalização da resposta
# --------------------------------------------------------------------------

def _texto(valor):
    valor = str(valor).strip() if valor is not None else ""
    return valor or None


def _idade(bruta):
    try:
        n = int(bruta) if bruta is not None and str(bruta).strip() != "" else None
    except (TypeError, ValueError):
        return None
    return n if n is not None and 0 <= n <= 200 else None


# Circunstâncias que só fazem sentido numa vítima de gênero feminino, com o
# motivo pelo qual cada uma só cabe ali. A lista autorizada da ocorrência não
# resolve isto: ela libera a circunstância para a ocorrência inteira, e numa
# ocorrência com uma mulher e um homem baleados a marcação pode cair no
# segundo. O recorte por vítima tem de ser feito por vítima.
SO_EM_MULHER = {
    "feminicidio_tentativa": "feminicídio é a morte de mulher pela condição de"
                             " gênero",
    "agente_seguranca_agressor": "marca a mulher ATINGIDA por disparo de agente"
                                 " de segurança, não o agente",
}


def _circunstancias_validas(brutas, autorizadas: list[str],
                            genero: str = "") -> list[str]:
    """Filtra pela lista AUTORIZADA, não pela lista completa. A LLM pode
    devolver feminicídio numa ocorrência em que o BERT não marcou mulher
    baleada; aqui essa marcação cai, porque a condição não foi satisfeita.

    E as de `SO_EM_MULHER` caem em qualquer vítima que não seja mulher. Foi
    assim que `agente_seguranca_agressor` acabou no policial que tinha sido
    BALEADO: quem leva um tiro é vítima, nunca o agressor, e nenhuma instrução
    de prompt garante isso — esta linha garante.
    """
    permitidas = [
        (valor, rotulo)
        for valor, rotulo in vocabulario.CIRCUNSTANCIAS
        if valor in set(autorizadas)
    ]
    marcadas = vocabulario.normalizar_lista(brutas, permitidas)
    if genero not in vocabulario.GENEROS_FEMININOS:
        marcadas = [c for c in marcadas if c not in SO_EM_MULHER]
    # `nao_se_aplica` é a ausência de circunstância: não convive com outra.
    concretas = [c for c in marcadas if c != vocabulario.CIRCUNSTANCIA_PADRAO]
    return concretas or [vocabulario.CIRCUNSTANCIA_PADRAO]


def _complemento_agente(bruto: dict | None) -> dict:
    bruto = bruto or {}
    return {
        "corporacao": vocabulario.normalizar(
            bruto.get("corporacao"), vocabulario.CORPORACOES, "nao_identificado"
        ),
        "cargo": vocabulario.normalizar(
            bruto.get("cargo"), vocabulario.CARGOS_AGENTE, "nao_identificado"
        ),
        "status_agente": vocabulario.normalizar(
            bruto.get("status_agente"), vocabulario.STATUS_AGENTE, "nao_identificado"
        ),
        "unidade": _texto(bruto.get("unidade")),
    }


def _complemento_politico(bruto: dict | None) -> dict:
    bruto = bruto or {}
    partido = _texto(bruto.get("partido")) or "Não identificado"
    conhecidos = {p.upper(): p for p in vocabulario.partidos()}
    return {
        # Sigla fora da lista é preservada como veio: o TSE muda a lista, e
        # perder o que a notícia disse seria pior do que guardar um valor novo.
        "partido": conhecidos.get(partido.upper(), partido),
        "cargo": vocabulario.normalizar(
            bruto.get("cargo"), vocabulario.CARGOS_POLITICO, ""
        ) or None,
        "status_politico": vocabulario.normalizar(
            bruto.get("status_politico"), vocabulario.STATUS_POLITICO, ""
        ) or None,
    }


def _normalizar_vitima(bruta: dict, autorizadas: list[str]) -> dict:
    idade = _idade(bruta.get("idade"))
    grupo = vocabulario.normalizar(
        bruta.get("grupo_etario"), vocabulario.GRUPOS_ETARIOS, "nao_identificado"
    )
    # Com idade em número, ela manda: deixar os dois campos se contradizerem
    # faria a mesma vítima ser contada num grupo etário na tabela e em outro no
    # gráfico.
    if idade is not None:
        grupo = vocabulario.grupo_etario(idade)

    tipos = vocabulario.normalizar_lista(
        bruta.get("tipo_vitima"), vocabulario.TIPOS_VITIMA, "civil"
    )

    genero = vocabulario.normalizar(
        bruta.get("genero"), vocabulario.GENEROS, "nao_identificado"
    )

    vitima = {
        "nome": _texto(bruta.get("nome")),
        "status": vocabulario.normalizar(
            bruta.get("status"), vocabulario.STATUS_VITIMA, "ferida"
        ),
        "genero": genero,
        "cor_raca": vocabulario.normalizar(
            bruta.get("cor_raca"), vocabulario.CORES_RACA, "nao_identificado"
        ),
        "grupo_etario": grupo,
        "idade": idade,
        "tipo_vitima": tipos,
        "circunstancias": _circunstancias_validas(
            bruta.get("circunstancias"), autorizadas, genero
        ),
        "local": vocabulario.normalizar(
            bruta.get("local"), vocabulario.LOCAIS_VITIMA, "nao_identificado"
        ),
        "descricao": _texto(bruta.get("descricao")),
        "data_morte": _texto(bruta.get("data_morte")),
        "fonte": _texto(bruta.get("fonte")),
        "divergencia": _texto(bruta.get("divergencia")),
    }

    # As complementares existem só para quem tem o tipo correspondente. A LLM
    # às vezes manda a tabela sem marcar o tipo; nesse caso o tipo é que está
    # errado, porque a tabela só é pedida quando a classe foi marcada.
    if "agente" in tipos or bruta.get("vitimizacao_policial"):
        vitima["tipo_vitima"] = vocabulario.normalizar_lista(
            [*tipos, "agente"], vocabulario.TIPOS_VITIMA, "civil"
        )
        vitima["vitimizacao_policial"] = _complemento_agente(
            bruta.get("vitimizacao_policial")
        )
    if "politico" in tipos or bruta.get("politico_baleado"):
        vitima["tipo_vitima"] = vocabulario.normalizar_lista(
            [*vitima["tipo_vitima"], "politico"], vocabulario.TIPOS_VITIMA, "civil"
        )
        vitima["politico_baleado"] = _complemento_politico(
            bruta.get("politico_baleado")
        )

    return vitima


def _normalizar_acao_policial(bruto: dict | None) -> dict:
    bruto = bruto or {}
    return {
        "contexto": vocabulario.normalizar(
            bruto.get("contexto"), vocabulario.CONTEXTOS_ACAO_POLICIAL,
            "acao_de_rotina",
        ),
        "corporacoes_presentes": vocabulario.normalizar_lista(
            bruto.get("corporacoes_presentes"), vocabulario.CORPORACOES,
            "nao_identificado",
        ),
        "unidades_presentes": _texto(bruto.get("unidades_presentes")),
    }


def _normalizar_crime_organizado(bruto: dict | None) -> dict:
    bruto = bruto or {}
    grupos = bruto.get("grupos_citados") or []
    if isinstance(grupos, str):
        grupos = [grupos]
    return {
        "contextos": vocabulario.normalizar_lista(
            bruto.get("contextos"), vocabulario.CONTEXTOS_CRIME_ORGANIZADO
        ),
        # Lista aberta: nome de facção não cabe em lista fechada.
        "grupos_citados": [g for g in (_texto(g) for g in grupos) if g],
    }


# --------------------------------------------------------------------------
# Conciliação de vítimas entre notícias
# --------------------------------------------------------------------------

def chave_vitima(vitima: dict) -> str:
    """Rótulo legível da identidade da vítima, usado para exibição e conferência.

    NÃO serve para casar vítimas entre reprocessamentos — use `mesma_pessoa`.
    Comparar chaves por igualdade duplicaria a pessoa sempre que uma informação
    nova chegasse: uma vítima sem nome que o analista batiza passaria a ter
    chave diferente da versão que a LLM continua devolvendo sem nome.
    """
    nome = nome_identificador(vitima)
    if nome:
        return f"nome:{nome}"
    return (
        f"perfil:{vitima.get('idade')}|{vitima.get('genero')}"
        f"|{','.join(vitima.get('tipo_vitima') or [])}"
    )


DESCONHECIDOS = {None, "", "nao_identificado"}

# Palavras que aparecem quando se descreve a pessoa em vez de nomeá-la.
DESCRITIVAS = {
    "adolescente", "anos", "bebe", "bebê", "crianca", "criança", "desconhecida",
    "desconhecido", "garota", "garoto", "homem", "homens", "identificada",
    "identificado", "idosa", "idoso", "jovem", "menina", "menino", "menor",
    "moca", "moça", "mulher", "mulheres", "nao", "não", "rapaz", "sem",
    "senhor", "senhora", "travesti", "vitima", "vítima",
}


def nome_identificador(vitima: dict) -> str:
    """Nome que serve para identificar a pessoa, ou "" quando não há.

    A LLM às vezes descreve a pessoa no lugar do nome ("Adolescente de 16
    anos"). Aceitar isso como nome próprio faria a mesma pessoa aparecer duas
    vezes assim que outra notícia a descrevesse de outro jeito — e, sendo as
    duas contadas como mortas, criaria chacinas que não existiram.
    """
    nome = (vitima.get("nome") or "").strip().lower()
    if not nome:
        return ""
    palavras = re.findall(r"\w+", nome)
    if any(palavra in DESCRITIVAS for palavra in palavras):
        return ""
    return nome


def _compativel(a, b) -> bool:
    """Valor ausente ou "não identificado" casa com qualquer coisa: uma fonte
    que não informou a idade não contradiz a que informou."""
    if a in DESCONHECIDOS or b in DESCONHECIDOS:
        return True
    return a == b


def mesma_pessoa(a: dict, b: dict) -> bool:
    """Regra de contagem do fluxo: o nome identifica a pessoa; sem nome em
    algum dos lados, compara idade, gênero e tipo de vítima."""
    nome_a = nome_identificador(a)
    nome_b = nome_identificador(b)
    if nome_a and nome_b:
        return nome_a == nome_b or nome_a in nome_b or nome_b in nome_a

    tipos_a = set(a.get("tipo_vitima") or [])
    tipos_b = set(b.get("tipo_vitima") or [])
    return (
        _compativel(a.get("idade"), b.get("idade"))
        and _compativel(a.get("genero"), b.get("genero"))
        # Tipos sem interseção são pessoas diferentes; um policial e um civil
        # não são a mesma vítima descrita de dois jeitos.
        and bool(tipos_a & tipos_b or not tipos_a or not tipos_b)
    )


# --------------------------------------------------------------------------
# Chamada
# --------------------------------------------------------------------------

# Quantas vezes insistir, e quanto esperar entre as tentativas. Os números são
# baixos de propósito: isto roda DENTRO da requisição do analista, que está
# olhando a tela. Uma fila de verdade, em segundo plano, aguentaria esperar o
# minuto inteiro do teto de tokens; aqui o limite é a paciência de quem espera.
TENTATIVAS = 3
ESPERA_MAXIMA = 12


def _transitorio(erro: APIError) -> float | None:
    """Quanto esperar antes de repetir, ou None se não vale a pena repetir.

    Três coisas falham de um jeito que some sozinho:

    - `json_validate_failed`: o modelo não fechou o JSON. É sorteio, e a
      tentativa seguinte costuma sair inteira. A biblioteca NÃO repete este
      sozinha, porque 400 normalmente significa pedido malformado — mas aqui o
      pedido está certo e quem errou foi a geração.
    - conexão e tempo esgotado: rede.
    - 5xx: a API.

    O teto de tokens por minuto fica de fora quando a espera é longa: segurar a
    requisição por um minuto para o analista é pior do que dizer a ele que
    repita daqui a pouco.
    """
    detalhe = getattr(erro, "message", None) or str(erro)
    if "json_validate_failed" in detalhe:
        return 1.0
    if isinstance(erro, (APIConnectionError, APITimeoutError)):
        return 2.0
    if isinstance(erro, APIStatusError):
        if erro.status_code >= 500:
            return 2.0
        if erro.status_code in (413, 429):
            espera = _segundos_sugeridos(erro)
            if espera is not None and espera <= ESPERA_MAXIMA:
                return espera
    return None


def _segundos_sugeridos(erro: APIStatusError) -> float | None:
    """O `retry-after` da resposta, quando a API o informa."""
    try:
        valor = erro.response.headers.get("retry-after")
    except Exception:
        return None
    if not valor:
        return None
    try:
        return float(valor)
    except ValueError:
        return None


def _pedir_a_llm(cliente, modelo: str, mensagens: list[dict]):
    """A chamada, com repetição no que é transitório.

    Devolve a resposta ou levanta o último erro, para que quem chama o traduza.
    """
    ultimo = None
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            return cliente.chat.completions.create(
                model=modelo,
                messages=mensagens,
                temperature=0,
                response_format={"type": "json_object"},
                reasoning_effort="low",
                max_tokens=4000,
            )
        except APIError as erro:
            ultimo = erro
            espera = _transitorio(erro)
            if espera is None or tentativa == TENTATIVAS:
                raise
            time.sleep(espera)
    raise ultimo


def classificar_ocorrencia(noticias: list[dict], con=None,
                           ocorrencia_id: int | None = None) -> dict:
    """Relê todas as notícias juntas e devolve fenômenos, vítimas e as
    tabelas complementares.

    `noticias` é uma lista de dicts com ao menos `texto` e, opcionalmente,
    `link` e `texto_processado`.

    Com `con`, cada marcação do BERT é gravada no banco de classificações antes
    de a LLM ser chamada. Sem ele a classificação funciona igual, mas o erro do
    modelo some assim que a LLM o corrige.
    """
    if not noticias:
        raise ErroClassificacao("A ocorrência não tem nenhuma notícia para classificar.")

    cliente, modelo = _cliente()

    # Etapa 1: o BERT lê cada notícia e marca as suas classes. A união das
    # marcações decide quais prompts entram na chamada seguinte e quais
    # circunstâncias a LLM fica autorizada a usar.
    blocos = []
    marcados: set[str] = set()
    tem_bert = True
    for i, noticia in enumerate(noticias, start=1):
        # A poda roda SEMPRE, mesmo sobre o `texto_processado` já gravado. Ela
        # é idempotente — podar texto já podado não muda nada —, e isso cura
        # sozinho as notícias gravadas quando a poda ainda estava desligada: o
        # valor antigo era o texto inteiro, e sem esta passagem ele continuaria
        # chegando inteiro à LLM para sempre.
        processado = preprocessar_noticia.resumir(
            noticia.get("texto_processado") or noticia["texto"]
        )
        marcacao = inferencia.marcar(processado)

        # O texto é gravado SEMPRE, antes de qualquer coisa e mesmo sem modelo
        # treinado. É a única forma de nada se perder: texto sem indicador
        # nenhum, e texto que chega antes do primeiro treino, não deixavam
        # rastro em lugar nenhum embora já fossem enviados à LLM.
        entrada_id = None
        if con is not None:
            entrada_id = auditoria.registrar_entrada(
                con, noticia["texto"], marcacao,
                link=noticia.get("link"),
                noticia_id=noticia.get("id"),
                ocorrencia_id=ocorrencia_id,
            )

        if marcacao is None:
            tem_bert = False
        else:
            marcados.update(marcacao["previsto"])
            if con is not None:
                auditoria.registrar(
                    con, marcacao, noticia["texto"],
                    link=noticia.get("link"),
                    noticia_id=noticia.get("id"),
                    ocorrencia_id=ocorrencia_id,
                    entrada_id=entrada_id,
                )
        cabecalho = f"## Notícia {i}"
        if noticia.get("link"):
            cabecalho += f" — {noticia['link']}"
        blocos.append(f"{cabecalho}\n{processado}")

    # Confirma AQUI, antes da LLM, o que o usuário enviou e o que o BERT leu.
    # Quem chama abre um `with con:` em volta de tudo, e esse bloco faz rollback
    # quando a LLM falha — era o que apagava o rastro justamente nas falhas em
    # que ele é mais útil: o texto entrava, a chamada estourava e não sobrava
    # linha nenhuma em `entrada`. Até este ponto a transação só tem o texto
    # recebido e a marcação; as vítimas só são escritas depois da resposta, então
    # nada pela metade é confirmado.
    if con is not None:
        con.commit()

    marcados_bert = sorted(marcados) if tem_bert else None
    autorizadas = indicadores.circunstancias_autorizadas(marcados_bert)

    conteudo = "\n\n".join(blocos)
    if marcados_bert:
        conteudo += (
            "\n\n## Classes marcadas pelo BERT (certifique ou descarte)\n"
            + ", ".join(marcados_bert)
        )

    # Etapa 2: a LLM. O núcleo vai numa mensagem própria e primeiro, porque
    # quase não muda e é o que a API pode manter em cache; os blocos de classe,
    # que mudam a cada notícia, vêm depois dele.
    try:
        resposta = _pedir_a_llm(cliente, modelo, [
            {"role": "system", "content": montar_nucleo(marcados_bert)},
            {"role": "system", "content": montar_prompt_indicadores(marcados_bert)},
            {"role": "user", "content": conteudo},
        ])
    except APIError as erro:
        # Sem esta conversão a exceção da biblioteca sobe inteira: quem chama só
        # trata `ErroClassificacao`, então a requisição morria com traceback no
        # terminal e a tela não chegava a dizer o que aconteceu.
        raise ErroClassificacao(
            _recado_da_api(erro, marcados_bert, conteudo)
        ) from erro

    escolha = resposta.choices[0]
    if escolha.finish_reason == "length":
        raise ErroClassificacao(
            "A resposta da LLM foi cortada por limite de tokens. "
            "Reduza o número de notícias da ocorrência ou aumente max_tokens."
        )

    try:
        bruto = json.loads(escolha.message.content)
    except json.JSONDecodeError as erro:
        raise ErroClassificacao(f"A LLM não devolveu um JSON válido: {erro}") from erro

    vitimas = [
        _normalizar_vitima(v, autorizadas) for v in bruto.get("vitimas") or []
    ]
    for vitima in vitimas:
        vitima["chave"] = chave_vitima(vitima)

    # QUEM DECIDE AS CLASSES É O BERT. A LLM não acrescenta nenhuma: o que ela
    # devolver fora da marcação é descartado aqui, e não só desencorajado no
    # prompt — instrução de prompt não é garantia, esta linha é.
    #
    # Duas exceções, as duas necessárias:
    #  - os indicadores que SÓ a LLM decide (violencia_genero) não têm classe no
    #    BERT para marcá-los, então a leitura dela é a única fonte que existe;
    #  - sem modelo treinado (`marcados_bert is None`) não há classificação
    #    prévia nenhuma, e recusar a leitura da LLM deixaria o sistema sem
    #    classificar nada.
    #
    # Os de regra ficam de fora em qualquer caso, mesmo que a LLM os devolva: a
    # contagem das vítimas é exata e a leitura não, e aceitar as duas criaria
    # uma chacina com dois mortos.
    de_fenomeno = indicadores.nomes_fenomeno()
    do_llm = set(bruto.get("fenomenos") or [])
    so_da_llm = {
        nome for nome in indicadores.nomes_decididos_pela_llm()
        if nome not in set(indicadores.nomes())
    }

    if marcados_bert is None:
        conhecidos = set(indicadores.nomes_decididos_pela_llm())
        fenomenos = [n for n in de_fenomeno if n in do_llm and n in conhecidos]
    else:
        marcados = set(marcados_bert)
        fenomenos = [
            nome for nome in de_fenomeno
            if nome in marcados or (nome in so_da_llm and nome in do_llm)
        ]

    # As classes marcadas para as quais a LLM não achou base no texto. Elas
    # CONTINUAM em `fenomenos` — quem revisa o classificador é o analista. O que
    # muda é que os campos delas não são preenchidos automaticamente.
    nao_sustentados = [
        nome for nome in (bruto.get("nao_sustentados") or [])
        if nome in set(fenomenos)
    ]
    # Uma classe marcada cuja tabela complementar a LLM não devolveu também
    # conta como não sustentada, mesmo que ela não tenha dito nada: o efeito
    # para o analista é o mesmo, um formulário em branco à espera dele.
    ignorados_pela_llm = [
        nome for nome in do_llm if nome not in set(fenomenos)
    ]

    resultado = {
        "fenomenos": fenomenos,
        # O que o BERT marcou, guardado separado para que se possa ver depois o
        # que a LLM aprofundou e o que ela deixou em branco.
        "fenomenos_bert": marcados_bert,
        "nao_sustentados": nao_sustentados,
        # O que a LLM tentou marcar por conta e foi recusado. Não vira
        # classificação, mas fica registrado: é sinal de falso negativo do BERT,
        # e portanto de notícia que vale anotar no corpus.
        "sugeridos_pela_llm": ignorados_pela_llm,
        "vitimas": vitimas,
        "vitimas_baleadas": "S" if vitimas else "N",
        "divergencias": bruto.get("divergencias") or [],
        "acao_policial": None,
        "crime_organizado": None,
    }

    # As complementares só são preenchidas quando a classe está marcada E a LLM
    # devolveu a tabela. Sem a tabela, o campo fica None em vez de receber os
    # valores padrão: formulário em branco é um pedido de revisão visível, e
    # `nao_identificado` em tudo se confunde com dado conferido.
    for nome, normalizar in (
        ("acao_policial", _normalizar_acao_policial),
        ("crime_organizado", _normalizar_crime_organizado),
    ):
        if nome not in set(fenomenos) or nome in set(nao_sustentados):
            continue
        tabela = bruto.get(nome)
        if tabela:
            resultado[nome] = normalizar(tabela)
        else:
            nao_sustentados.append(nome)

    resultado["nao_sustentados"] = nao_sustentados

    return resultado
