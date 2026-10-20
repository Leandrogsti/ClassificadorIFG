"""
Limpa o texto da notícia antes de ele ser classificado.

O QUE ESTE ARQUIVO FAZ, EM DUAS ETAPAS

1. `normalizar(texto)` — arruma o texto sem jogar nada material fora: junta
   espaço repetido, tira emoji, apaga linha duplicada, corta rabicho de crédito
   de foto e desfaz ofuscação do tipo `m0rto a tir0s`.
2. `resumir(texto)` — normaliza e depois REMOVE as linhas de ruído de portal:
   assinatura, data de publicação, "Compartilhe no Whatsapp", "Baixe o app",
   legenda de foto, logo de navegação.

Quem chama `resumir` é tanto a fila de recebidas (antes do BERT) quanto o
classificador (antes da LLM). Isso é de propósito: se os dois lessem textos
diferentes, a triagem poderia promover uma notícia por uma classe cujo trecho
o outro não enxerga.

A REGRA QUE GOVERNA A PODA, E O MOTIVO DELA

Uma linha só é removida quando **casa com um padrão de ruído E não carrega
nenhum marcador de fato**. A segunda metade é um VETO, e é ela que torna a
poda segura.

A versão anterior fazia o contrário: MANTINHA a frase que tivesse termo do
vocabulário ou entidade nomeada, e descartava o resto. Medido nas notícias do
banco, isso apagava fato material — saíram "Mulher estava atrás de viatura",
que é o que posiciona a vítima em relação à polícia, e "os dois estavam
conscientes e sem gravidade aparente", que é o estado das vítimas. Nenhuma das
duas tinha termo do vocabulário nem entidade nomeada, e esse era todo o
critério. Por isso o filtro vivia desligado, e por isso ele foi invertido:

- deixar lixo passar custa ~40 tokens;
- deixar fato sair custa uma classificação errada.

Com o critério invertido, o erro cai sempre para o lado barato.

NÃO COMPRIMIR PALAVRA POR PALAVRA. Existiu aqui uma compressão que removia
pontuação, acentos e stopwords e lematizava o resto. Ela derruba preposições
como "por" e "contra", que são exatamente o que marca quem atirou em quem:
"recebidas a tiros POR um grupo armado" virava uma sopa de palavras onde não dá
mais para saber a direção do tiro, e isso causou uma classificação errada em
produção. A lição fica para quem pensar em reintroduzi-la.

Uso: import preprocessar_noticia; curto = preprocessar_noticia.resumir(texto)
"""
import json
import re
import unicodedata
from pathlib import Path

FONTE = Path(__file__).parent / "categorias_fonte.json"
FONTE_INDICADORES = Path(__file__).parent / "indicadores_fonte.json"

# Interruptor geral. Diferente do antigo `FILTRAR_FRASES`, que nascia
# desligado por não ser confiável: este pode ficar ligado porque a remoção
# exige padrão de ruído reconhecido, e não ausência de sinal.
PODAR = True

# Piso de segurança. Se a poda derrubar mais que isto do texto, alguma coisa
# está errada com os padrões e é melhor mandar o texto inteiro para o modelo do
# que mandar um texto mutilado.
FRACAO_MAXIMA_REMOVIDA = 0.45
MINIMO_DE_LINHAS = 2

VOCABULARIO_BASE = [
    "tiro", "tiros", "tiroteio", "disparo", "disparos", "baleado", "baleada",
    "arma", "pistola", "revólver", "fuzil", "bala", "vítima", "vítimas",
    "polícia", "policial", "policiais", "suspeito", "morto", "morta", "morte",
    "ferido", "ferida", "assalto", "roubo", "sequestro", "confronto",
    "prisão", "preso", "executado", "executada", "execução", "assassinado",
    "assassinada", "baleados", "baleadas", "atirou", "atiraram",
    # QUEM é a vítima e QUEM atirou é metade do que o sistema extrai. Sem estas
    # palavras, "Mulher estava atrás de viatura" não tinha um único termo.
    "mulher", "mulheres", "homem", "homens", "criança", "adolescente", "idoso",
    "menina", "menino", "jovem", "moradora", "morador", "mototaxista",
    "viatura", "batalhão", "delegacia", "corporação", "agente", "agentes",
    "guarda", "militar", "sargento", "soldado", "cabo", "tenente", "capitão",
    "coronel", "delegado", "inspetor", "folga", "serviço", "operação",
    "facção", "milícia", "emboscada", "feminicídio", "companheiro",
    "ex-companheiro", "marido", "esposa", "namorada", "namorado",
    "atingido", "atingida", "atingidos", "socorrido", "socorrida", "hospital",
    "óbito", "consciente", "conscientes", "gravidade", "ferimentos",
]

# Padrões de ruído de portal de notícia. Cada um saiu de uma linha real das
# notícias do banco — não são hipóteses.
PADROES_RUIDO = [
    # Assinatura e data: "Por g1 Rio", "Publicado em 08/08/26 12:22Por Joubert"
    r"^por\s+[a-z0-9]",
    r"^publicado\s+em\b",
    r"^atualizado\b",
    r"\d{1,2}h\d{2}\b.*\batualizado\b",
    r"^em\s+atualiza(ç|c)(ã|a)o\b",
    # Crédito e legenda de imagem
    r"^(foto|imagem|v(í|i)deo|arte|infogr(á|a)fico|legenda)\s*:",
    r"^(reprodu(ç|c)(ã|a)o|divulga(ç|c)(ã|a)o|arquivo pessoal)\b",
    # Convite a compartilhar, seguir, baixar
    r"^compartilh",
    r"^(siga|curta|inscreva|cadastre|receba|baixe|clique|acesse|assine)\b",
    r"^(assista|confira|veja|leia|saiba)\b",
    r"\bbaixe (o|a|os|as|nosso)\b",
    r"\b(aplicativo|app)\b.*\b(dispon(í|i)vel|gratuitamente|gr(á|a)tis|celular)\b",
    r"\b(epis(ó|o)dio|v(í|i)deos curtos|newsletter|podcast)\b",
    r"\bgoogle\s+(news|discovery)\b",
    r"^fique por dentro\b",
    r"\bnotícias? (que importam|em tempo real)\b",
    # Navegação, rodapé, caixa de relacionados
    r"^(tags|assuntos|t(ó|o)picos|mais lidas|relacionad|agora no)\b",
    r"-logo$",
    r"todos os direitos reservados",
    r"^©",
    r"^(compartilhe|publicidade|propaganda|an(ú|u)ncio)$",
]

# O rabicho de crédito que vem colado numa linha que vale a pena guardar:
# "Agentes de segurança no local ... . Foto: Reprodução" perde só o fim.
RABICHO_CREDITO = re.compile(
    r"\s*(—|–|-|\|)?\s*(foto|imagem|v(í|i)deo|cr(é|e)dito)s?\s*:\s*"
    r"[^.!?]{0,60}$",
    re.IGNORECASE,
)

# Ofuscação de rede social: zero no lugar de "o", 1 no lugar de "i", e assim
# por diante, para a palavra furar moderação automática. Só é desfeita quando o
# resultado é uma palavra que ESTE domínio conhece — ver `_desofuscar`.
TROCAS_OFUSCADAS = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                    "7": "t", "@": "a", "$": "s"}

# Emoji e pictogramas. Não carregam sinal de classificação e custam token.
_CATEGORIAS_SIMBOLO = {"So", "Cf"}

_termos_relevantes: set[str] | None = None
_ruido = [re.compile(p, re.IGNORECASE) for p in PADROES_RUIDO]


def _textos_do_dominio() -> list[str]:
    """De onde sai o vocabulário de relevância.

    Vinha só de `categorias_fonte.json`, que descreve as categorias de
    MOTIVAÇÃO — conceito que saiu do sistema. O vocabulário do domínio atual
    está nos RÓTULOS dos indicadores e nas listas fechadas da vítima, e era por
    isso que "mulher" e "viatura" não contavam como termo relevante.

    SÓ O RÓTULO DO INDICADOR, nunca a definição nem a skill. Medido: colhendo
    lema da prosa das skills entravam "perder", "seguir", "escrever" e "decidir",
    e com eles o veto passava a proteger qualquer linha — inclusive "você pode
    seguir o palco do g1 Rio para não perder nenhum episódio". Veto que protege
    tudo não protege nada.
    """
    textos: list[str] = []

    if FONTE_INDICADORES.exists():
        dados = json.loads(FONTE_INDICADORES.read_text(encoding="utf-8"))
        for indicador in dados.get("indicadores", []):
            textos.append(indicador.get("rotulo", ""))

    # As listas fechadas: são exatamente os valores que a LLM tem de achar no
    # texto, então a palavra que os nomeia é relevante por definição.
    try:
        import vocabulario

        for lista in (vocabulario.GENEROS, vocabulario.TIPOS_VITIMA,
                      vocabulario.LOCAIS_VITIMA, vocabulario.CIRCUNSTANCIAS,
                      vocabulario.CORPORACOES, vocabulario.CARGOS_AGENTE,
                      vocabulario.GRUPOS_ETARIOS, vocabulario.STATUS_AGENTE):
            textos.extend(rotulo for _, rotulo in lista)
    except Exception:
        # A limpeza não pode cair por causa do vocabulário: sem ele o veto fica
        # mais estreito, e isso é degradação, não falha.
        pass

    # Só o NOME da categoria antiga, pela mesma razão: a definição dela é prosa
    # e traria verbo genérico de volta para o veto.
    if FONTE.exists():
        dados = json.loads(FONTE.read_text(encoding="utf-8"))
        for cat in dados.get("categorias", []):
            textos.append((cat.get("nome", "") or "").replace("_", " "))

    return [t for t in textos if t]


# Palavras de ligação dos rótulos. "Criança ou adolescente baleado" contribui
# com três termos úteis e com "ou", que não é marcador de nada.
_LIGACAO = {
    "ou", "e", "de", "do", "da", "dos", "das", "em", "no", "na", "a", "o",
    "as", "os", "um", "uma", "com", "sem", "por", "para", "que", "se",
    # As listas fechadas trazem "Não identificado" e "Não se aplica", e daí
    # saíam "não", "se" e "aplica" como se fossem marcadores de fato. "não"
    # está em quase toda frase de português: com ele no conjunto, o veto
    # protegia o texto inteiro e a poda não removia nada.
    "não", "nao", "aplica", "identificado", "identificada", "nenhum",
    "nenhuma", "outro", "outra", "outros", "outras", "mais", "ainda", "ser",
    "está", "estar", "foi", "era", "tipo", "status", "geral", "comum",
}


def termos_do_dominio() -> set[str]:
    """As palavras que fazem uma linha valer a pena, calculadas UMA vez.

    SÃO AS PALAVRAS COMO ELAS APARECEM, sem lematizar, e isto é a correção de
    um defeito medido: o rótulo "Bala perdida" lematizava `perdida` em
    `perder`, e com esse lema no conjunto o veto protegia qualquer linha que
    contivesse "perder" — inclusive "para não perder nenhum episódio". Rótulo
    curto é a palavra que a notícia usa; lematizá-lo não acrescenta nada e
    custa precisão.

    A flexão que importa está escrita à mão em `VOCABULARIO_BASE`
    ("baleado", "baleada", "baleados", "baleadas"), onde dá para ver e revisar.
    Sem spaCy aqui, a função também deixou de custar 0,12 s por processo.
    """
    global _termos_relevantes
    if _termos_relevantes is not None:
        return _termos_relevantes

    termos = {t.lower() for t in VOCABULARIO_BASE}
    for texto in _textos_do_dominio():
        for palavra in _palavras(texto):
            if len(palavra) > 2 and palavra not in _LIGACAO:
                termos.add(palavra)

    _termos_relevantes = termos
    return termos


def _palavras(linha: str) -> set[str]:
    return set(re.findall(r"[0-9a-zà-öø-ÿ][0-9a-zà-öø-ÿ'\-]*", linha.lower()))


def _desofuscar(linha: str) -> str:
    """`m0rto a tir0s` volta a ser `morto a tiros`.

    Só troca dígito por letra quando a palavra MISTURA letra e dígito e o
    resultado é um termo que este domínio conhece. É essa dupla condição que
    protege o que não pode ser tocado: `45 anos`, `41º BPM` e `29` não têm
    letra junto do dígito, então nunca entram aqui — e uma idade adulterada
    seria pior que a ofuscação.
    """
    termos = termos_do_dominio()

    def trocar(casamento: re.Match) -> str:
        palavra = casamento.group(0)
        letras = sum(c.isalpha() for c in palavra)
        digitos = sum(c in TROCAS_OFUSCADAS for c in palavra)
        if letras < 2 or digitos < 1:
            return palavra
        candidata = "".join(TROCAS_OFUSCADAS.get(c, c) for c in palavra)
        return candidata if candidata.lower() in termos else palavra

    return re.sub(r"[0-9a-zA-Zà-öø-ÿÀ-ÖØ-Þ@$]+", trocar, linha)


def _sem_simbolos(texto: str) -> str:
    return "".join(
        c for c in texto
        if unicodedata.category(c) not in _CATEGORIAS_SIMBOLO
    )


def tem_marcador_de_fato(linha: str) -> bool:
    """O VETO da poda: esta linha diz algo que a classificação usa?

    Três coisas contam, e qualquer uma basta:

    - uma palavra do domínio (vítima, agente, arma, situação, local);
    - uma idade ou contagem de pessoas ("45 anos", "três pessoas");
    - um horário, que é o que ancora o fato no tempo.
    """
    palavras = _palavras(linha)
    if palavras & termos_do_dominio():
        return True
    if re.search(r"\b\d{1,3}\s*anos?\b", linha, re.IGNORECASE):
        return True
    if re.search(r"\b\d{1,2}\s*h\s*\d{0,2}\b", linha, re.IGNORECASE):
        return True
    if re.search(
        r"\b(um|uma|dois|duas|tr(ê|e)s|quatro|cinco|seis|sete|oito|nove|dez|"
        r"\d{1,3})\s+(pessoas?|v(í|i)timas?|homens?|mulheres?|crian(ç|c)as?|"
        r"adolescentes?|jovens?|policiais?|agentes?|mortos?|mortas?|feridos?|"
        r"feridas?|baleados?|baleadas?)\b",
        linha, re.IGNORECASE,
    ):
        return True
    return False


def e_ruido(linha: str) -> bool:
    """Linha de portal que não diz nada sobre o fato.

    A ordem importa: o marcador de fato tem a última palavra. Uma legenda de
    foto que descreve a cena ("Agentes de segurança no local onde um sargento
    foi assassinado") casa com padrão de crédito e MESMO ASSIM fica.
    """
    enxuta = linha.strip()
    if not enxuta:
        return True
    if tem_marcador_de_fato(enxuta):
        return False
    return any(p.search(enxuta) for p in _ruido)


def normalizar(texto: str) -> str:
    """Limpa sem decidir o que é relevante. Não remove linha de conteúdo.

    Faz quatro coisas, todas reversíveis em termos de sentido: tira emoji,
    corta rabicho de crédito de foto, desfaz ofuscação de dígito e apaga linha
    repetida — a legenda de foto dos portais repete a manchete palavra por
    palavra, duas vezes seguidas, e isso é token pago duas vezes.
    """
    if not texto:
        return ""

    linhas, vistas = [], set()
    for bruta in _sem_simbolos(texto).splitlines():
        linha = re.sub(r"[ \t ]+", " ", bruta).strip()
        if not linha:
            continue
        linha = RABICHO_CREDITO.sub("", linha).strip(" -—–|").strip()
        if not linha:
            continue
        linha = _desofuscar(linha)
        chave = re.sub(r"[^0-9a-zà-öø-ÿ]+", "", linha.lower())
        if chave and chave in vistas:
            continue
        vistas.add(chave)
        linhas.append(linha)

    return "\n".join(linhas)


def resumir(texto: str, min_frases: int = MINIMO_DE_LINHAS,
            filtrar: bool | None = None) -> str:
    """O texto como o BERT e a LLM o recebem: normalizado e sem ruído.

    `filtrar=False` devolve só o normalizado, para comparar o efeito da poda
    sem mexer na constante. `min_frases` é o piso de linhas abaixo do qual a
    poda é descartada inteira.
    """
    limpo = normalizar(texto)
    if not (PODAR if filtrar is None else filtrar) or not limpo:
        return limpo

    linhas = limpo.splitlines()
    mantidas = [l for l in linhas if not e_ruido(l)]

    # Os dois piso de segurança. Poda que derruba quase tudo não é poda, é
    # defeito nos padrões, e aí o texto inteiro é a resposta menos pior.
    if len(mantidas) < min_frases:
        return limpo
    if len("\n".join(mantidas)) < len(limpo) * (1 - FRACAO_MAXIMA_REMOVIDA):
        return limpo

    return "\n".join(mantidas)


def relatorio(texto: str) -> dict:
    """O que a poda faria com este texto, linha por linha. Serve ao ajuste dos
    padrões e à tela de testes: poda que ninguém consegue auditar é poda que
    ninguém confia."""
    limpo = normalizar(texto)
    linhas = limpo.splitlines()
    removidas = [l for l in linhas if e_ruido(l)]
    final = resumir(texto)
    return {
        "original": len(texto or ""),
        "normalizado": len(limpo),
        "final": len(final),
        "linhas": len(linhas),
        "removidas": removidas,
        "desistiu": final == limpo and bool(removidas),
    }
