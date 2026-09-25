"""Etapas 3 a 5: pré-processamento, filtro do BERTimbau e validação pela LLM.

A cada notícia adicionada, a LLM relê TODAS as notícias da ocorrência juntas e
refaz a lista de vítimas do zero. Isso custa mais tokens do que processar só a
notícia nova, mas é o que impede a soma indevida: se um jornal diz "um homem
baleado" e outro diz "dois homens baleados", o total é 2, nunca 3.

O prompt estático (dicionário de motivações + skills dos indicadores) vai como
primeira mensagem de sistema para aproveitar o cache da API entre chamadas.
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

import bertimbau
import indicadores
import preprocessar_noticia

load_dotenv()

RAIZ = Path(__file__).parent
DICIONARIO = RAIZ / "dicionario_categorias.txt"

from vocabulario import (
    CIRCUNSTANCIA_PADRAO, CIRCUNSTANCIAS, GENEROS, ROTULOS_CIRCUNSTANCIA,
    SITUACOES, TIPOS_VITIMA,
)


class ErroClassificacao(RuntimeError):
    pass


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


def _skills_indicadores() -> str:
    linhas = []
    for indicador in indicadores.carregar():
        if indicador["tipo"] != "texto":
            continue
        linhas.append(
            f"- **{indicador['nome']}** ({indicador['rotulo']}): "
            f"{indicador['definicao']} {indicador.get('skill', '')}".strip()
        )
    return "\n".join(linhas)


INSTRUCOES_VITIMAS = """
# SEGUNDA TAREFA: LISTA DE VÍTIMAS

Você recebe TODAS as notícias desta mesma ocorrência. Elas descrevem o MESMO
fato sob fontes diferentes. Monte a lista de PESSOAS DISTINTAS atingidas,
refazendo-a do zero. NUNCA some as vítimas de notícias diferentes.

Regras de contagem:
1. Mesma pessoa citada em várias notícias conta UMA vez. O nome identifica a
   pessoa; sem nome, compare idade, gênero e tipo de vítima.
2. Sem nomes, cruze circunstância e situação. O total NUNCA pode ser menor que
   o maior número citado em uma única notícia: "um homem baleado" em um jornal
   e "dois homens baleados" em outro resultam em 2 vítimas, não 3.
3. Quando as fontes divergirem sobre a mesma pessoa (idade, situação, nome),
   registre a divergência com o trecho de cada fonte no campo "divergencia"
   da vítima, e escolha o valor da fonte mais específica.
4. Só entra na lista quem foi ATINGIDO por disparo: "situacao" é `morta` ou
   `ferida`. Quem escapou sem ser atingido não é vítima e não entra.
5. Não invente vítimas: só entra quem o texto descreve como atingida.
6. Em "nome", escreva o nome próprio como a notícia o traz. Se a notícia não
   informar o nome, use null — NUNCA descreva a pessoa nesse campo. "Adolescente
   de 16 anos", "Homem de 24 anos" e "Vítima não identificada" são INVÁLIDOS:
   a idade e o gênero já têm campos próprios, e um nome inventado faz a mesma
   pessoa ser contada duas vezes quando outra notícia a descreve de outro jeito.

Preencha "circunstancia" com UM destes valores, e nenhum outro:
{circunstancias}
Use `nao_se_aplica` quando nenhum dos demais descrever o caso.

# TERCEIRA TAREFA: INDICADORES DE TEXTO

Confirme quais destes indicadores se aplicam, seguindo a skill de cada um:

{skills}

Só marque o que o texto sustenta. Uma ocorrência pode receber mais de um.
"""

FORMATO = """
# FORMATO DE SAÍDA (substitui o formato descrito acima)

Retorne SOMENTE este JSON, sem texto antes ou depois:
```json
{
  "motivo_principal": "slug",
  "motivos_complementares": ["slug"],
  "indicadores_texto": ["acao_policial"],
  "vitimas": [
    {
      "nome": "Nome da pessoa ou null",
      "idade": 30,
      "genero": "feminino|masculino|nao_informado",
      "tipo_vitima": "civil|agente_seguranca|politico|suspeito|nao_informado",
      "situacao": "morta|ferida",
      "data_morte": "AAAA-MM-DD ou null",
      "cargo_politico": "cargo/candidatura ou null",
      "circunstancia": "um dos valores da lista fechada acima",
      "fonte": "Notícia 1",
      "divergencia": "o que cada fonte diz, quando houver conflito, ou null"
    }
  ],
  "divergencias": ["conflitos entre fontes que o analista precisa decidir"]
}
```
- `idade`: número inteiro ou null quando a notícia não informar.
- `vitimas`: lista vazia se nenhuma pessoa foi atingida.
"""


def montar_prompt_sistema() -> str:
    dicionario = DICIONARIO.read_text(encoding="utf-8")
    lista_circunstancias = "\n".join(
        f"- `{slug}` — {rotulo}" for slug, rotulo in CIRCUNSTANCIAS
    )
    return (
        dicionario
        + "\n"
        + INSTRUCOES_VITIMAS.format(
            skills=_skills_indicadores(), circunstancias=lista_circunstancias
        )
        + "\n"
        + FORMATO
    )


def _normalizar_vitima(bruta: dict) -> dict:
    def texto(valor):
        valor = (str(valor).strip() if valor is not None else "")
        return valor or None

    idade = bruta.get("idade")
    try:
        idade = int(idade) if idade is not None and str(idade).strip() != "" else None
    except (TypeError, ValueError):
        idade = None

    genero = (bruta.get("genero") or "nao_informado").strip().lower()
    tipo = (bruta.get("tipo_vitima") or "nao_informado").strip().lower()
    situacao = (bruta.get("situacao") or "ferida").strip().lower()
    circunstancia = (bruta.get("circunstancia") or "").strip().lower()

    return {
        "nome": texto(bruta.get("nome")),
        "idade": idade,
        "genero": genero if genero in GENEROS else "nao_informado",
        "tipo_vitima": tipo if tipo in TIPOS_VITIMA else "nao_informado",
        "situacao": situacao if situacao in SITUACOES else "ferida",
        "data_morte": texto(bruta.get("data_morte")),
        "cargo_politico": texto(bruta.get("cargo_politico")),
        "circunstancia": (
            circunstancia
            if circunstancia in ROTULOS_CIRCUNSTANCIA
            else CIRCUNSTANCIA_PADRAO
        ),
        "fonte": texto(bruta.get("fonte")),
        "divergencia": texto(bruta.get("divergencia")),
    }


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
        f"perfil:{vitima.get('idade')}|{vitima.get('genero')}|{vitima.get('tipo_vitima')}"
    )


DESCONHECIDOS = {None, "", "nao_informado"}

# Palavras que aparecem quando se descreve a pessoa em vez de nomeá-la.
DESCRITIVAS = {
    "adolescente", "anos", "bebe", "bebê", "crianca", "criança", "desconhecida",
    "desconhecido", "garota", "garoto", "homem", "homens", "identificada",
    "identificado", "idosa", "idoso", "jovem", "menina", "menino", "menor",
    "moca", "moça", "mulher", "mulheres", "nao", "não", "rapaz", "sem",
    "senhor", "senhora", "vitima", "vítima",
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
    """Valor ausente ou "não informado" casa com qualquer coisa: uma fonte que
    não informou a idade não contradiz a que informou."""
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
    return (
        _compativel(a.get("idade"), b.get("idade"))
        and _compativel(a.get("genero"), b.get("genero"))
        and _compativel(a.get("tipo_vitima"), b.get("tipo_vitima"))
    )


def classificar_ocorrencia(noticias: list[dict]) -> dict:
    """Relê todas as notícias juntas e devolve motivação, vítimas e indicadores.

    `noticias` é uma lista de dicts com ao menos `texto` e, opcionalmente,
    `link` e `texto_processado`.
    """
    if not noticias:
        raise ErroClassificacao("A ocorrência não tem nenhuma notícia para classificar.")

    cliente, modelo = _cliente()

    blocos = []
    sugestoes_bert: set[str] = set()
    for i, noticia in enumerate(noticias, start=1):
        processado = noticia.get("texto_processado") or preprocessar_noticia.resumir(
            noticia["texto"]
        )
        sugestao = bertimbau.classificar(processado)
        if sugestao:
            sugestoes_bert.update(sugestao)
        cabecalho = f"## Notícia {i}"
        if noticia.get("link"):
            cabecalho += f" — {noticia['link']}"
        blocos.append(f"{cabecalho}\n{processado}")

    conteudo = "\n\n".join(blocos)
    if sugestoes_bert:
        conteudo += (
            "\n\n## Sugestão do classificador BERTimbau (confirme ou descarte)\n"
            + ", ".join(sorted(sugestoes_bert))
        )

    resposta = cliente.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": montar_prompt_sistema()},
            {"role": "user", "content": conteudo},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        reasoning_effort="low",
        max_tokens=4000,
    )

    escolha = resposta.choices[0]
    if escolha.finish_reason == "length":
        raise ErroClassificacao(
            "A resposta da LLM foi cortada por limite de tokens. "
            "Reduza o número de notícias da ocorrência ou aumente max_tokens."
        )

    try:
        bruto = json.loads(escolha.message.content)
    except json.JSONDecodeError as erro:
        raise ErroClassificacao(
            f"A LLM não devolveu um JSON válido: {erro}"
        ) from erro

    vitimas = [_normalizar_vitima(v) for v in bruto.get("vitimas") or []]
    for vitima in vitimas:
        vitima["chave"] = chave_vitima(vitima)

    nomes_texto = {
        i["nome"] for i in indicadores.carregar() if i["tipo"] == "texto"
    }
    indicadores_texto = [
        nome for nome in bruto.get("indicadores_texto") or [] if nome in nomes_texto
    ]

    return {
        "motivo_principal": bruto.get("motivo_principal"),
        "motivos_complementares": bruto.get("motivos_complementares") or [],
        "indicadores_texto": indicadores_texto,
        "vitimas": vitimas,
        "divergencias": bruto.get("divergencias") or [],
    }
