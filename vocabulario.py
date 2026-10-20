"""Listas fechadas do Dicionário de Dados.

Fica num módulo próprio porque cada lista vale em quatro lugares ao mesmo
tempo: o esquema do banco, a validação da resposta da LLM, os formulários e a
exportação em planilha. Mantê-las aqui é o que impede que o rótulo mostrado na
tela e o valor gravado divirjam.

Cada lista é uma sequência de pares `(valor, rótulo)`: o valor é o que vai para
o banco, o rótulo é o que a pessoa lê. O valor nunca muda depois de gravado;
o rótulo pode ser reescrito à vontade.

Campos que o dicionário marca como "comporta missing" têm um valor explícito
para o desconhecido — `nao_identificado`, `nao_se_aplica` ou vazio — em vez de
NULL. A diferença importa na hora de contar: `nao_identificado` é uma resposta
("procuramos e não achamos"), NULL é a ausência de resposta.
"""

import json
from pathlib import Path

RAIZ = Path(__file__).parent


def _valores(pares) -> list[str]:
    return [valor for valor, _ in pares]


def _rotulos(pares) -> dict[str, str]:
    return dict(pares)


FONTE_EDITAVEL = RAIZ / "vocabulario_fonte.json"

# As listas que a tela /locais deixa editar, com o nome que aparece lá. São as
# que mudam por decisão de quem usa — uma corporação nova, um cargo que faltava
# — e não por mudança de conceito do sistema.
EDITAVEIS = {
    "CORPORACOES": "Corporação do agente",
    "CARGOS_AGENTE": "Patente ou cargo do agente",
    "STATUS_AGENTE": "Situação de serviço do agente",
    "CARGOS_POLITICO": "Cargo do político",
    "STATUS_POLITICO": "Situação do político",
    "CONTEXTOS_ACAO_POLICIAL": "Contexto da ação policial",
    "CONTEXTOS_CRIME_ORGANIZADO": "Contexto do crime organizado",
}


def _personalizadas() -> dict[str, list]:
    """O que foi editado na tela. Arquivo ausente ou quebrado cai no padrão:
    uma lista fechada que some derrubaria a validação de toda a aplicação."""
    if not FONTE_EDITAVEL.exists():
        return {}
    try:
        return json.loads(FONTE_EDITAVEL.read_text(encoding="utf-8")).get("listas") or {}
    except json.JSONDecodeError:
        return {}


def _lista(nome: str, padrao: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """O valor vigente de uma lista editável: o do arquivo, se houver, senão o
    de fábrica. Os pares voltam como tupla para que o resto do código não
    perceba diferença entre uma lista editada e uma que nunca foi tocada."""
    gravada = _personalizadas().get(nome)
    if not gravada:
        return padrao
    return [(item[0], item[1]) for item in gravada if len(item) >= 2]


def salvar_lista(nome: str, pares: list[tuple[str, str]]) -> None:
    """Grava uma lista editada. Só as de `EDITAVEIS`: as demais são conceito do
    sistema, e mudá-las em arquivo deixaria o código e o dado em desacordo."""
    if nome not in EDITAVEIS:
        raise ValueError(f"Lista não editável: {nome}")
    dados = {"_leia_me": "", "listas": {}}
    if FONTE_EDITAVEL.exists():
        try:
            dados = json.loads(FONTE_EDITAVEL.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    dados.setdefault("listas", {})[nome] = [list(par) for par in pares]
    FONTE_EDITAVEL.write_text(
        json.dumps(dados, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------
# Ocorrência
# --------------------------------------------------------------------------

FAIXAS_HORARIAS = [
    ("madrugada", "Madrugada"),
    ("manha", "Manhã"),
    ("tarde", "Tarde"),
    ("noite", "Noite"),
    ("nao_identificado", "Não identificado"),
]
ROTULOS_FAIXA_HORARIA = _rotulos(FAIXAS_HORARIAS)

# A faixa é derivada da hora quando ela existe, para que as duas colunas nunca
# se contradigam. Sem hora, fica "não identificado" — e não um chute.
LIMITES_FAIXA_HORARIA = [
    (0, 5, "madrugada"),
    (6, 11, "manha"),
    (12, 17, "tarde"),
    (18, 23, "noite"),
]


def faixa_horaria(hora: str | None) -> str:
    if not hora:
        return "nao_identificado"
    try:
        h = int(str(hora).split(":")[0])
    except (ValueError, IndexError):
        return "nao_identificado"
    for inicio, fim, faixa in LIMITES_FAIXA_HORARIA:
        if inicio <= h <= fim:
            return faixa
    return "nao_identificado"


ESTADOS = [
    ("AC", "Acre"), ("AL", "Alagoas"), ("AP", "Amapá"), ("AM", "Amazonas"),
    ("BA", "Bahia"), ("CE", "Ceará"), ("DF", "Distrito Federal"),
    ("ES", "Espírito Santo"), ("GO", "Goiás"), ("MA", "Maranhão"),
    ("MT", "Mato Grosso"), ("MS", "Mato Grosso do Sul"), ("MG", "Minas Gerais"),
    ("PA", "Pará"), ("PB", "Paraíba"), ("PR", "Paraná"), ("PE", "Pernambuco"),
    ("PI", "Piauí"), ("RJ", "Rio de Janeiro"), ("RN", "Rio Grande do Norte"),
    ("RS", "Rio Grande do Sul"), ("RO", "Rondônia"), ("RR", "Roraima"),
    ("SC", "Santa Catarina"), ("SP", "São Paulo"), ("SE", "Sergipe"),
    ("TO", "Tocantins"),
]
ROTULOS_ESTADO = _rotulos(ESTADOS)

# --------------------------------------------------------------------------
# Vítima
# --------------------------------------------------------------------------

STATUS_VITIMA = [
    ("morta", "Morta"),
    ("ferida", "Ferida"),
]
ROTULOS_STATUS_VITIMA = _rotulos(STATUS_VITIMA)

GENEROS = [
    ("mulher_cis", "Mulher Cis"),
    ("homem_cis", "Homem Cis"),
    ("mulher_trans_travesti", "Mulher Trans e Travesti"),
    ("homem_trans", "Homem Trans"),
    ("nao_binario", "Não binário"),
    ("nao_identificado", "Não identificado"),
]
ROTULOS_GENERO = _rotulos(GENEROS)

# Os gêneros que contam para o fenômeno "Mulheres baleadas". Mulher trans e
# travesti entram: o fenômeno é sobre a condição de mulher, não sobre o sexo
# designado ao nascer.
GENEROS_FEMININOS = ["mulher_cis", "mulher_trans_travesti"]

CORES_RACA = [
    ("negra", "Negra"),
    ("indigena", "Indígena"),
    ("amarela", "Amarela"),
    ("branca", "Branca"),
    ("nao_identificado", "Não identificado"),
]
ROTULOS_COR_RACA = _rotulos(CORES_RACA)

GRUPOS_ETARIOS = [
    ("crianca", "Criança"),
    ("adolescente", "Adolescente"),
    ("adulto", "Adulto"),
    ("idoso", "Idoso"),
    ("nao_identificado", "Não identificado"),
]
ROTULOS_GRUPO_ETARIO = _rotulos(GRUPOS_ETARIOS)

# Faixas do ECA (Lei 8.069/1990) para criança e adolescente, e 60 anos para
# idoso (Estatuto da Pessoa Idosa, Lei 10.741/2003).
LIMITES_GRUPO_ETARIO = [(0, 11, "crianca"), (12, 17, "adolescente"),
                        (18, 59, "adulto"), (60, 200, "idoso")]


def grupo_etario(idade) -> str:
    """Deriva o grupo da idade. A notícia que diz só "criança", sem número,
    deixa a idade em branco e o grupo é preenchido direto pela LLM."""
    if idade is None or str(idade).strip() == "":
        return "nao_identificado"
    try:
        n = int(idade)
    except (TypeError, ValueError):
        return "nao_identificado"
    for inicio, fim, grupo in LIMITES_GRUPO_ETARIO:
        if inicio <= n <= fim:
            return grupo
    return "nao_identificado"


# Marcação MÚLTIPLA: um policial que também é vereador é agente e político.
TIPOS_VITIMA = [
    ("civil", "Civil"),
    ("agente", "Agente"),
    ("politico", "Político"),
]
ROTULOS_TIPO_VITIMA = _rotulos(TIPOS_VITIMA)

# Marcação MÚLTIPLA. `nao_se_aplica` é o valor de quando nenhuma das outras
# descreve o caso, e por isso nunca convive com outra na mesma vítima.
CIRCUNSTANCIAS = [
    ("feminicidio_tentativa", "Feminicídio/tentativa"),
    ("suicidio", "Suicídio"),
    ("acidente", "Acidente"),
    ("trajeto_escolar", "Trajeto escolar"),
    ("bala_perdida", "Bala perdida"),
    ("chacina", "Chacina"),
    ("lgbtqiapn", "LGBTQIAPN+"),
    ("tribunal_do_crime", "Tribunal do crime"),
    ("agente_seguranca_agressor", "Agente de segurança agressor"),
    ("nao_se_aplica", "Não se aplica"),
]
ROTULOS_CIRCUNSTANCIA = _rotulos(CIRCUNSTANCIAS)
CIRCUNSTANCIA_PADRAO = "nao_se_aplica"

# Circunstâncias que a LLM só avalia quando o BERT marcou a classe
# correspondente. É o que impede a LLM de sair procurando feminicídio em
# notícia sem mulher baleada, e é também o que mantém o prompt pequeno: a
# instrução de cada uma só é enviada quando pode ser usada.
CIRCUNSTANCIAS_CONDICIONADAS = {
    "feminicidio_tentativa": "mulheres_baleadas",
    # Agente agressor é marcação da MULHER ATINGIDA por disparo de agente de
    # segurança, e por isso depende de `mulheres_baleadas`. Antes dependia de
    # `acao_policial`, e o resultado foi a circunstância cair no policial que
    # tinha sido BALEADO — quem é vítima não é agressor.
    "agente_seguranca_agressor": "mulheres_baleadas",
    "lgbtqiapn": "lgbtqiapn",
    "suicidio": "suicidio",
}

# Circunstâncias que o SISTEMA deriva e a LLM não marca.
#
# `chacina` voltou para cá em 2026-10-09. Ela esteve aqui, saiu quando a
# distinção entre a vítima e o algoz parecia só legível no texto, e voltou
# quando essa distinção virou uma CONTA que a aplicação sabe fazer: conta quem
# morreu, menos o agente que estava em serviço. Três civis mortos é chacina;
# dois civis mais um agente morto é chacina se o agente estava de folga, e não
# é se ele estava em serviço, porque aí ele está do outro lado do mesmo fato.
#
# Deriva, e não é pedida, pelo motivo de sempre: pedir ao modelo o que a conta
# já sabe cria uma segunda resposta capaz de contradizer a primeira — uma
# chacina marcada em duas vítimas, ou um fenômeno que nenhuma vítima sustenta.
# A regra mora em `indicadores_fonte.json` e é aplicada por
# `indicadores.vitimas_de_chacina`.
CIRCUNSTANCIAS_DERIVADAS: set[str] = {"chacina"}

LOCAIS_VITIMA = [
    ("residencia", "Residência"),
    ("area_lazer_aglomeracao", "Área de lazer/Aglomeração"),
    ("espacos_privados_servicos", "Espaços privados de serviços"),
    ("transporte_publico", "Transporte público"),
    ("automovel", "Automóvel (carro, moto, caminhão)"),
    ("unidades_ensino", "Unidades de ensino"),
    # Sem `outros` de propósito: era a saída fácil, e a LLM a tomava em todo
    # caso que não encaixasse de primeira. Local que o texto não diz é
    # `nao_identificado`, que é a verdade e o analista completa depois.
    ("nao_identificado", "Não identificado"),
]
ROTULOS_LOCAL_VITIMA = _rotulos(LOCAIS_VITIMA)


# --------------------------------------------------------------------------
# Tabela complementar: vitimização policial
# --------------------------------------------------------------------------

_PADRAO_CORPORACOES = [
    ("policia_militar", "Polícia Militar"),
    ("policia_civil", "Polícia Civil"),
    ("bombeiro", "Bombeiro"),
    ("guarda_municipal", "Guarda Municipal"),
    ("policia_federal", "Polícia Federal"),
    ("policia_rodoviaria_federal", "Polícia Rodoviária Federal"),
    ("policia_ferroviaria_federal", "Polícia Ferroviária Federal"),
    ("seguranca_presente", "Segurança Presente"),
    ("exercito", "Exército"),
    ("marinha", "Marinha"),
    ("aeronautica", "Aeronáutica"),
    ("policia_penal", "Polícia Penal"),
    ("forca_nacional", "Força Nacional"),
    ("outros", "Outros"),
    ("nao_identificado", "Não identificado"),
]
CORPORACOES = _lista("CORPORACOES", _PADRAO_CORPORACOES)
ROTULOS_CORPORACAO = _rotulos(CORPORACOES)

_PADRAO_CARGOS_AGENTE = [
    ("nao_identificado", "Não identificado"),
    ("soldado", "Soldado"), ("cabo", "Cabo"), ("sargento", "Sargento"),
    ("sub_tenente", "Sub-Tenente"), ("tenente", "Tenente"),
    ("capitao", "Capitão"), ("major", "Major"),
    ("tenente_coronel", "Tenente-Coronel"), ("coronel", "Coronel"),
    ("escrivao", "Escrivão"), ("agente_de_policia", "Agente de Polícia"),
    ("investigador", "Investigador"), ("perito", "Perito"),
    ("papiloscopista", "Papiloscopista"), ("delegado", "Delegado"),
    ("lider", "Líder"), ("subinspetor", "Subinspetor"),
    ("inspetor", "Inspetor"), ("inspetor_regional", "Inspetor Regional"),
    ("agente_operacional", "Agente Operacional"),
    ("agente_especial", "Agente Especial"),
    ("general_de_brigada", "General de Brigada"),
    ("general_de_divisao", "General de Divisão"),
    ("general_de_exercito", "General de Exército"),
    ("marechal", "Marechal"), ("suboficial", "Suboficial"),
    ("capitao_tenente", "Capitão-tenente"),
    ("capitao_de_corveta", "Capitão de Corveta"),
    ("capitao_de_fragata", "Capitão de Fragata"),
    ("capitao_de_mar_e_guerra", "Capitão de Mar e Guerra"),
    ("contra_almirante", "Contra-Almirante"),
    ("vice_almirante", "Vice-Almirante"),
    ("almirante_de_esquadra", "Almirante de Esquadra"),
    ("almirante", "Almirante"), ("brigadeiro", "Brigadeiro"),
    ("major_brigadeiro", "Major-Brigadeiro"),
    ("tenente_brigadeiro", "Tenente-Brigadeiro"),
    ("marechal_do_ar", "Marechal do Ar"),
    ("agente_penitenciario", "Agente penitenciário"),
    ("comissario", "Comissário"), ("cadete", "Cadete"),
    ("agente_socioeducativo", "Agente socioeducativo"),
    ("oficial_de_cartorio", "Oficial de cartório"),
    ("sub_oficial", "Sub-oficial"),
    ("outros", "Outros"), ("nao_se_aplica", "Não se aplica"),
]
CARGOS_AGENTE = _lista("CARGOS_AGENTE", _PADRAO_CARGOS_AGENTE)
ROTULOS_CARGO_AGENTE = _rotulos(CARGOS_AGENTE)

_PADRAO_STATUS_AGENTE = [
    ("em_servico", "Em serviço"),
    ("fora_de_servico", "Fora de serviço"),
    ("aposentado_exonerado", "Aposentado / exonerado"),
    ("nao_identificado", "Não identificado"),
]
STATUS_AGENTE = _lista("STATUS_AGENTE", _PADRAO_STATUS_AGENTE)
ROTULOS_STATUS_AGENTE = _rotulos(STATUS_AGENTE)


# --------------------------------------------------------------------------
# Tabela complementar: político baleado
# --------------------------------------------------------------------------

_PADRAO_CARGOS_POLITICO = [
    ("deputado_estadual", "Deputado estadual"),
    ("deputado_federal", "Deputado federal"),
    ("governador", "Governador"),
    ("ministro", "Ministro"),
    ("prefeito", "Prefeito"),
    ("presidente", "Presidente"),
    ("secretario", "Secretário"),
    ("senador", "Senador"),
    ("vereador", "Vereador"),
    ("vice_governador", "Vice-governador"),
    ("vice_prefeito", "Vice-prefeito"),
    ("vice_presidente", "Vice-presidente"),
]
CARGOS_POLITICO = _lista("CARGOS_POLITICO", _PADRAO_CARGOS_POLITICO)
ROTULOS_CARGO_POLITICO = _rotulos(CARGOS_POLITICO)

_PADRAO_STATUS_POLITICO = [
    ("no_mandato", "No mandato"),
    ("candidato", "Candidato"),
    ("pre_candidato", "Pré-candidato"),
]
STATUS_POLITICO = _lista("STATUS_POLITICO", _PADRAO_STATUS_POLITICO)
ROTULOS_STATUS_POLITICO = _rotulos(STATUS_POLITICO)


def partidos() -> list[str]:
    return json.loads(
        (RAIZ / "partidos.json").read_text(encoding="utf-8")
    )["partidos"]


# --------------------------------------------------------------------------
# Tabela complementar: ação policial
# --------------------------------------------------------------------------

_PADRAO_CONTEXTOS_ACAO_POLICIAL = [
    ("acao_de_rotina", "Ação de rotina"),
    ("operacao_policial", "Operação policial"),
]
CONTEXTOS_ACAO_POLICIAL = _lista("CONTEXTOS_ACAO_POLICIAL", _PADRAO_CONTEXTOS_ACAO_POLICIAL)
ROTULOS_CONTEXTO_ACAO = _rotulos(CONTEXTOS_ACAO_POLICIAL)


# --------------------------------------------------------------------------
# Tabela complementar: crime organizado
# --------------------------------------------------------------------------

_PADRAO_CONTEXTOS_CRIME_ORGANIZADO = [
    ("disputa_grupos_armados", "Disputa de grupos armados"),
    ("ataque_alvos_civis", "Ataque a alvos civis"),
    ("presenca_grupos_armados", "Presença de grupos armados"),
    ("tribunal_do_crime", "Tribunal do crime"),
    ("chacina", "Chacina"),
]
CONTEXTOS_CRIME_ORGANIZADO = _lista("CONTEXTOS_CRIME_ORGANIZADO", _PADRAO_CONTEXTOS_CRIME_ORGANIZADO)
ROTULOS_CONTEXTO_CRIME = _rotulos(CONTEXTOS_CRIME_ORGANIZADO)


# --------------------------------------------------------------------------
# Validação
# --------------------------------------------------------------------------

def normalizar(valor, pares, padrao: str) -> str:
    """Encaixa um valor numa lista fechada. Vale tanto para o que a LLM
    devolve quanto para o que chega de um formulário adulterado."""
    texto = (str(valor).strip().lower() if valor is not None else "")
    return texto if texto in _valores(pares) else padrao


def normalizar_lista(valores, pares, padrao: str | None = None) -> list[str]:
    """Mesma coisa para campo de marcação múltipla. Devolve na ordem da lista
    fechada, não na ordem em que chegaram: assim duas gravações com as mesmas
    marcações produzem exatamente o mesmo JSON, e o histórico não acusa uma
    alteração que não houve."""
    if isinstance(valores, str):
        valores = [valores]
    escolhidos = {str(v).strip().lower() for v in (valores or [])}
    marcados = [valor for valor in _valores(pares) if valor in escolhidos]
    if not marcados and padrao is not None:
        return [padrao]
    return marcados
