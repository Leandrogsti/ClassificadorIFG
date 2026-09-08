"""
Reduz o texto da noticia antes de enviar ao modelo, usando spaCy para descartar
frases sem relacao com a ocorrencia (propaganda, "leia tambem", navegacao,
créditos de foto etc.), que só consomem tokens sem ajudar na classificação.

Mantém apenas frases com pelo menos um termo relevante (extraído das próprias
categorias em categorias_fonte.json, mais um vocabulário base de violência
armada) ou com entidades nomeadas (pessoa/local/organização), que costumam
carregar os fatos da notícia.

Uso como script: coloque a notícia crua em noticia.txt e rode
    python preprocessar_noticia.py
O resultado limpo é salvo em noticia_processada.txt (arquivo de entrada e
arquivo de saída separados, como pedido).

Existe também uma compressão a nível de palavra (comprimir/processar): remove
pontuação, acentos e stopwords, e lematiza o resto. NÃO É o padrão usado pelo
teste_groq.py — testamos em produção e ela derruba preposições como "por" e
"contra", que são exatamente o que marca quem atirou em quem (ex.: "recebidas
a tiros POR um grupo armado" vira uma sopa de palavras onde não dá mais pra
saber a direção do tiro), e isso já causou uma classificação errada. Use
resumir() para o fluxo normal; comprimir()/processar() ficam disponíveis só
para quem aceitar esse risco em troca de mais economia de tokens.

Uso como módulo: import preprocessar_noticia; texto_curto = preprocessar_noticia.resumir(texto)
"""
import json
import unicodedata
from pathlib import Path

import spacy

FONTE = Path("categorias_fonte.json")
ENTRADA = Path("noticia.txt")
SAIDA = Path("noticia_processada.txt")

VOCABULARIO_BASE = [
    "tiro", "tiros", "tiroteio", "disparo", "disparos", "baleado", "baleada",
    "arma", "pistola", "revólver", "fuzil", "bala", "vítima", "vítimas",
    "polícia", "policial", "suspeito", "morto", "morta", "ferido", "ferida",
    "assalto", "roubo", "sequestro", "confronto", "prisão", "preso",
]

NEGACOES = {"não", "nunca", "nem", "sem", "jamais"}

# Palavras que marcam quem fez o quê a quem (agente/alvo) ou que o modelo
# pt_core_news_sm classifica erradamente como stopword (ex.: "grupo"); nunca
# remover, mesmo sendo stopword, senão a direção do fato se perde.
MANTER_MESMO_SE_STOPWORD = {"por", "contra", "grupo", "grupos"}

_nlp = None


def _carregar_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("pt_core_news_sm")
    return _nlp


def _vocabulario_categorias() -> set[str]:
    if not FONTE.exists():
        return set()
    dados = json.loads(FONTE.read_text(encoding="utf-8"))
    nlp = _carregar_nlp()
    termos: set[str] = set()
    for cat in dados.get("categorias", []):
        texto = cat["definicao"] + " " + cat.get("nao_e", "")
        doc = nlp(texto)
        for token in doc:
            if token.pos_ in {"NOUN", "VERB", "ADJ"} and not token.is_stop and token.is_alpha:
                termos.add(token.lemma_.lower())
    return termos


def resumir(texto: str, min_frases: int = 2) -> str:
    """Remove frases sem sinal relevante para a classificação. Se filtrar
    demais (fica com menos de min_frases), devolve o texto original."""
    nlp = _carregar_nlp()
    termos_relevantes = set(VOCABULARIO_BASE) | _vocabulario_categorias()

    doc = nlp(texto)
    frases_mantidas = []
    for sent in doc.sents:
        tem_entidade = any(ent.label_ in {"PER", "LOC", "ORG"} for ent in sent.ents)
        lemas = {t.lemma_.lower() for t in sent if t.is_alpha}
        tem_termo_relevante = bool(lemas & termos_relevantes)
        if tem_termo_relevante or tem_entidade:
            frases_mantidas.append(sent.text.strip())

    if len(frases_mantidas) < min_frases:
        return texto.strip()

    return " ".join(frases_mantidas)


def _remover_acentos(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def comprimir(texto: str) -> str:
    """Compressão a nível de palavra: remove pontuação, acentos e stopwords,
    e lematiza o restante. Entidades nomeadas e números ficam com o texto
    original (sem lematizar/sem remover acento) para preservar nomes, locais
    e quantidades. Negações nunca são removidas, mesmo sendo stopwords."""
    nlp = _carregar_nlp()
    doc = nlp(texto)
    indices_entidade = {token.i for ent in doc.ents for token in ent}

    palavras = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        if token.text.lower() in NEGACOES:
            palavras.append(_remover_acentos(token.text.lower()))
            continue
        if token.i in indices_entidade or token.like_num:
            palavras.append(_remover_acentos(token.text))
            continue
        if not token.is_alpha:
            continue
        if token.is_stop and token.text.lower() not in MANTER_MESMO_SE_STOPWORD:
            continue
        lema = _remover_acentos(token.lemma_.lower())
        if lema:
            palavras.append(lema)

    return " ".join(palavras)


def processar(texto: str) -> str:
    """Pipeline completo: descarta frases irrelevantes (resumir) e depois
    comprime o que sobrou palavra por palavra (comprimir)."""
    return comprimir(resumir(texto))


if __name__ == "__main__":
    noticia = ENTRADA.read_text(encoding="utf-8")
    reduzida = resumir(noticia)
    SAIDA.write_text(reduzida, encoding="utf-8")
    print(f"Entrada:  {ENTRADA} ({len(noticia)} caracteres)")
    print(f"Saída:    {SAIDA} ({len(reduzida)} caracteres)")
