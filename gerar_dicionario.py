"""
Gera o dicionario_categorias.txt (prompt enxuto) a partir de categorias_fonte.json.

Uso: escreva/edite categorias em categorias_fonte.json com texto livre (definicao)
e uma frase curta de desambiguacao (nao_e). Este script usa spaCy para extrair
automaticamente as palavras-chave (gatilhos) de cada definicao, evitando que a
descricao completa em prosa precise ir para o prompt (que e o que mais consome
tokens na API).

ATENCAO: as CATEGORIAS DE MOTIVACAO que este script gerava sairam do sistema em
2026-10-06 — o classificador registra fenomeno observado e circunstancia, e nada
mais. O que ainda sai daqui e e usado em producao sao as secoes PAPEL e RECORTE
DE ANALISE do prompt, lidas por `classificador._secoes_do_dicionario`. O resto
do arquivo gerado permanece como registro do conceito.

Para editar as secoes que ainda viajam no prompt (PAPEL e RECORTE DE ANALISE),
mexa em categorias_fonte.json, na chave "secoes", e rode:

    python gerar_dicionario.py

Elas ficam no JSON, e nao no TEMPLATE abaixo, para haver UM lugar de edicao: o
README sempre mandou editar o JSON, e enquanto esses dois textos moravam aqui
quem seguia o README nao via efeito nenhum.
"""
import json
from pathlib import Path

import spacy

FONTE = Path("categorias_fonte.json")
SAIDA = Path("dicionario_categorias.txt")

POS_RELEVANTES = {"NOUN", "PROPN", "VERB", "ADJ"}
MAX_PALAVRAS_CHAVE = 8


def extrair_palavras_chave(nlp, texto: str) -> list[str]:
    doc = nlp(texto)
    termos: list[str] = []

    # Expressoes de 2-3 palavras (ex.: "arma de fogo", "agentes de seguranca")
    for chunk in doc.noun_chunks:
        palavras = [t for t in chunk if not t.is_stop and not t.is_punct and t.is_alpha]
        if 2 <= len(palavras) <= 3:
            frase = " ".join(t.lemma_.lower() for t in palavras)
            if frase not in termos:
                termos.append(frase)

    # Palavras isoladas com carga semantica (substantivos, verbos, adjetivos)
    for token in doc:
        if (
            token.pos_ in POS_RELEVANTES
            and not token.is_stop
            and token.is_alpha
            and len(token.lemma_) > 2
        ):
            lema = token.lemma_.lower()
            if lema not in termos and not any(lema in t for t in termos):
                termos.append(lema)

    return termos[:MAX_PALAVRAS_CHAVE]


def montar_categorias(nlp, categorias: list[dict]) -> str:
    nomes = [c["nome"] for c in categorias]
    linhas = []
    for i, cat in enumerate(categorias, start=1):
        gatilhos = extrair_palavras_chave(nlp, cat["definicao"])
        linhas.append(
            f"{i}. **{cat['nome']}** — gatilhos: {', '.join(gatilhos)}. "
            f"Não é: {cat['nao_e']}"
        )
    return "\n".join(linhas), nomes


def montar_motivacoes_json(nomes: list[str]) -> str:
    pares = ",".join(f'"{n}":0' for n in nomes)
    return "{" + pares + "}"


TEMPLATE = """# PAPEL
{papel}

# RECORTE DE ANÁLISE
{recorte_de_analise}

# DOIS NÍVEIS
- PRINCIPAL: o fato que ORIGINOU os tiros (sempre exatamente 1). Pergunta-chave: "o que originou os tiros?"
- COMPLEMENTAR(ES): fatos de 2ª ordem somados depois, mas ainda dentro do próprio evento de disparo (refém tomado na fuga, suicídio do autor, troca de tiros com a polícia que chegou durante o crime ainda em andamento). Zero ou mais.
Morte é RESULTADO, não motivo → Homicídio só é principal se matar era o objetivo e não há outro motivo melhor (roubo/briga/disputa/policial/tortura viram principal, com homicidio_tentativa complementar).

# AS {n} CATEGORIAS (gatilhos · não-é / desempate)

{categorias}

# REGRAS-MESTRAS DE DESEMPATE
1. Fato que ORIGINOU os tiros = principal; o que se somou depois = complementar.
2. Motivo patrimonial vence local: tiro em bar durante assalto = roubo_tentativa, não ataque_a_civis.
3. Local só define ataque_a_civis se os disparos foram direcionados à coletividade/espaço, não por estar ali.
4. Bala perdida sempre segue o confronto de origem (nunca disparo_acidental nem ataque_a_civis).
5. Polícia trocando tiros ou sendo alvo NO MOMENTO do fato = acao_operacao_policial. Polícia que só chega depois para atender/investigar um crime já encerrado NÃO conta nesta categoria. Agente de folga, suicídio de agente ou acidente com a própria arma NÃO entram nesta categoria.
6. Refém só vira sequestro_carcere_privado com retenção extra ou deslocamento forçado; senão é roubo_tentativa comum.
7. Tortura exige sinal objetivo; muitos tiros letais sozinhos não bastam. Item roubado só encontrado perto do corpo ou usado na fuga não descarta tortura nem homicidio_tentativa.
8. disparo_acidental = involuntário; tiros_a_esmo = intencional sem alvo definido.
9. Mesmo com informação parcial, escolha sempre a categoria mais provável — nunca deixe motivo_principal vazio.

# FORMATO DE SAÍDA
Retorne só este JSON (todas as chaves, sem comentários/texto fora):
```json
{{
  "motivo_principal": "roubo_tentativa",
  "motivos_complementares": ["acao_operacao_policial", "homicidio_tentativa"],
  "motivacoes": {motivacoes_exemplo}
}}
```
Notas do formato:
- `motivo_principal`: 1 dos {n} slugs (sempre exatamente 1, nunca null).
- `motivos_complementares`: zero ou mais, nunca repete o principal.
- `motivacoes`: flag 1 sse o slug está no principal ou nos complementares; todos os demais 0.

# EXEMPLOS RÁPIDOS DE CALIBRAÇÃO
- 6 pedestres abordados em sequência na mesma avenida, tiro p/ alto → **arrastao**.
- Homens atiram em rajada contra bar cheio, alvo era 1 pessoa → **ataque_a_civis** + compl. `homicidio_tentativa`.
- Briga de vizinhos por som alto termina em morte → **briga** + compl. `homicidio_tentativa`.
- Corpo com mãos amarradas e tiros nas pernas/cabeça → **tortura** + compl. `homicidio_tentativa`.
- Operação do Bope em comunidade em guerra de facções, 2 mortos → **acao_operacao_policial** + compl. `disputa`.
- Vítima interceptada por indivíduos que atiram e fogem, sem motivo relatado → **homicidio_tentativa**.
- PM de folga reage a assalto ao sair de restaurante → **roubo_tentativa** + compl. `homicidio_tentativa`.
- Carro-forte atacado durante abastecimento de caixas em agência → **roubo_de_cargas**.
- Homem mata ex-companheira e se suicida em seguida → **homicidio_tentativa** + compl. `suicidio`.
- Homem é morto a tiros após briga; polícia só chega depois para socorrer, apreender arma/corpo e abrir investigação (sem troca de tiros) → **briga** + compl. `homicidio_tentativa`; NÃO é acao_operacao_policial, mesmo citando UPP/operação contra tráfico em outro contexto.

# INSTRUÇÃO FINAL
Confirme os indicadores, monte a lista de vítimas, identifique o fato que originou os tiros (principal), depois os fatos que se somaram (complementares), e devolva só o JSON.
"""


def main() -> None:
    dados = json.loads(FONTE.read_text(encoding="utf-8"))
    categorias = dados["categorias"]

    nlp = spacy.load("pt_core_news_sm", disable=["ner"])

    categorias_txt, nomes = montar_categorias(nlp, categorias)

    # PAPEL e RECORTE vêm do JSON, e não daqui: são as duas seções que ainda
    # viajam no prompt de produção, e o README manda editar o JSON. Mantê-las
    # no template deste script dava dois lugares de edição — quem seguisse o
    # README mexia no JSON e nada mudava.
    secoes = dados.get("secoes") or {}
    faltando = [c for c in ("papel", "recorte_de_analise") if not secoes.get(c)]
    if faltando:
        raise SystemExit(
            f"{FONTE} está sem a(s) seção(ões) {', '.join(faltando)} em"
            " 'secoes'. São elas que vão para o prompt de produção."
        )

    texto_final = TEMPLATE.format(
        n=len(nomes),
        categorias=categorias_txt,
        motivacoes_exemplo=montar_motivacoes_json(nomes),
        papel=secoes["papel"],
        recorte_de_analise=secoes["recorte_de_analise"],
    )
    SAIDA.write_text(texto_final, encoding="utf-8")
    print(f"OK: {SAIDA} gerado com {len(nomes)} categorias.")


if __name__ == "__main__":
    main()
