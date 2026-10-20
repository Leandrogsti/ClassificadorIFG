"""Treino e avaliação do classificador de indicadores, antes da camada de LLM.

Uma tarefa só, multirrótulo, descrita em corpus.py: quais dos oito indicadores
a notícia sustenta. Não existe filtro binário antes dela — a notícia fora do
escopo é a que o modelo deixa sem nenhum indicador.

MULTIRRÓTULO, NÃO MULTICLASSE. É a decisão que organiza o arquivo todo. A
pergunta não é "qual é o indicador desta notícia", e sim "quais destes oito ela
sustenta". Cada indicador tem a sua própria saída, passa pela sua própria
sigmoide e decide sozinho se entra, sem disputar com os outros: uma notícia pode
sair com nenhum, com um ou com os sete. Não há `softmax`, não há `argmax` e não
há nada que pare a leitura depois do primeiro indicador marcado.

Duas ESTRATÉGIAS de treino produzem esse mesmo tipo de saída, e as duas estão
implementadas para que a escolha entre elas seja um resultado medido do
trabalho, e não um palpite:

- `unico` — UMA rede com sete saídas sigmoide. O encoder é compartilhado, então
  o que o modelo aprende sobre a linguagem das notícias serve a todos os
  indicadores, e há um treino só. É o padrão.
- `por_indicador` — SETE redes binárias independentes (um-contra-resto). Cada
  indicador ganha um modelo inteiro, com o seu próprio ponto de parada e o seu
  próprio limiar, e um indicador raro não disputa a capacidade da rede com os
  frequentes. Custa sete treinos, sete checkpoints em disco e sete redes
  carregadas na memória durante a inferência.

Duas correções valem para as DUAS estratégias e, em corpus pequeno e
desbalanceado, pesam mais que a escolha entre elas:

1. PESO DAS CLASSES na perda (`pos_weight`). Com 20 exemplos de um indicador
   contra 400 sem ele, a perda não ponderada é quase minimizada por um modelo
   que nunca o marca — e o resultado é exatamente o sintoma de "só sai um
   indicador". O peso de cada classe é a razão entre os negativos e os
   positivos dela no treino, com teto para que uma classe com 2 exemplos não
   domine o gradiente.
2. LIMIAR POR INDICADOR, calibrado na VALIDAÇÃO. Cortar todo mundo em 0,5 é um
   palpite: a saída de um indicador raro costuma ficar abaixo disso mesmo
   quando ele é o rótulo certo, e aí ele nunca aparece, por bem ordenadas que
   estejam as pontuações. Cada indicador recebe o limiar que maximiza o seu F1
   na validação, e esse limiar é gravado junto com os pesos.

A calibração olha a validação e NUNCA o teste — calibrar no teste infla o número
que vai para a banca. Por isso as métricas trazem também o F1 que o MESMO modelo
teria com todos os cortes em 0,5: a diferença entre os dois é o efeito medido da
calibração, não uma afirmação de que ela ajuda.

Roda em GPU quando há uma disponível e em CPU quando não há, sem mudar nada no
código: `dispositivo="auto"` resolve isso. Em CPU o treino é lento, mas o
resultado é o mesmo, o que permite reproduzir o trabalho em qualquer máquina.
"""
import json
import time
from pathlib import Path

import numpy as np

import corpus
from db import agora

RAIZ = Path(__file__).parent
PASTA_MODELOS = RAIZ / "modelos"

# O arquivo que a inferência lê para saber o que foi treinado: a estratégia, a
# ordem dos rótulos, o limiar de cada um e onde estão os pesos. Sem ele a
# inferência teria de adivinhar o formato da pasta.
MANIFESTO = "modelo.json"

ESTRATEGIAS = {
    "unico": "Uma rede multirrótulo, com uma saída sigmoide por indicador",
    "por_indicador": "Uma rede binária por indicador (um-contra-resto)",
    # O corte é preenchido abaixo, a partir de `votos_para_prosseguir`:
    # escrito à mão, este texto já divergiu da regra uma vez.
    "cluster": "Três modelos por indicador, votando",
}
ESTRATEGIA_PADRAO = "unico"

# Os três modelos que formam cada cluster. A ordem não importa para o voto, mas
# fica fixa para que o manifesto seja estável entre treinos.
MODELOS_CLUSTER = ["albertina-100m", "bertimbau-base", "bertimbau-large"]

# A FRAÇÃO dos modelos que precisa concordar para a classe seguir sozinha para a
# LLM. 2 de 3, por decisão da usuária em 2026-10-07:
#
#     0 votos  -> vermelho, revisão obrigatória por pessoa
#     1 voto   -> laranja, alerta; pessoa verifica
#     2 ou 3   -> segue para a LLM
#
# Fração, e não contagem fixa, porque `unico` e `por_indicador` têm UM modelo:
# com o número 2 cravado, nenhuma classe passaria nessas estratégias e a LLM
# nunca seria chamada. Em 1 de 1 a fração dá 1,0 e a classe segue, que é o
# comportamento certo para elas.
FRACAO_PARA_PROSSEGUIR = 2 / 3


def votos_para_prosseguir(n_modelos: int) -> int:
    """Quantos votos, em n modelos, bastam para a classe seguir à LLM."""
    import math

    return max(1, math.ceil(FRACAO_PARA_PROSSEGUIR * max(1, n_modelos)))


# Mantido com o nome antigo porque o manifesto de um cluster já treinado o
# gravou; vale 2 para o trio de três.
VOTOS_PARA_CLASSIFICAR = votos_para_prosseguir(3)

ESTRATEGIAS["cluster"] += (
    f"; {VOTOS_PARA_CLASSIFICAR} de 3 votos classificam"
    f"{', o resto vai a uma pessoa' if VOTOS_PARA_CLASSIFICAR < 3 else ''}"
)

# Os três modelos da comparação, todos MONOLÍNGUES de português. A restrição é
# uma decisão do trabalho: o corpus é de notícias brasileiras, e um vocabulário
# treinado em português representa "baleado", "chacina" ou "milícia" em menos
# subpalavras que um vocabulário repartido entre cem idiomas.
#
# Com isso a comparação cobre UM eixo — o tamanho do modelo (100M, 110M e 335M)
# — e deixa de medir dois que os multilíngues cobriam: monolíngue contra
# multilíngue e destilado contra completo. Não é omissão, é recorte; o que o
# trabalho afirma passa a ser qual modelo de português vai melhor, e não se o
# português ajuda.
MODELOS = {
    "bertimbau-base": {
        "id": "neuralmind/bert-base-portuguese-cased",
        "rotulo": "BERTimbau base",
        "parametros": "110M",
        "tipo": "monolíngue (português)",
    },
    "bertimbau-large": {
        "id": "neuralmind/bert-large-portuguese-cased",
        "rotulo": "BERTimbau large",
        "parametros": "335M",
        "tipo": "monolíngue (português)",
    },
    "albertina-100m": {
        "id": "PORTULAN/albertina-100m-portuguese-ptbr-encoder",
        "rotulo": "Albertina PT-BR 100M",
        "parametros": "100M",
        "tipo": "monolíngue (português do Brasil)",
    },
}

# Grade de busca. Valores escolhidos em torno do que a literatura de fine-tuning
# de BERT recomenda (lr entre 2e-5 e 5e-5, poucas épocas): corpora pequenos
# sobreajustam rápido, então mais épocas pedem lr menor.
GRADE = [
    {"learning_rate": 2e-5, "batch_size": 16, "epocas": 4, "weight_decay": 0.01, "max_length": 256},
    {"learning_rate": 3e-5, "batch_size": 16, "epocas": 4, "weight_decay": 0.01, "max_length": 256},
    {"learning_rate": 5e-5, "batch_size": 16, "epocas": 3, "weight_decay": 0.01, "max_length": 256},
    {"learning_rate": 2e-5, "batch_size": 8, "epocas": 5, "weight_decay": 0.1, "max_length": 320},
]

HIPERPARAMETROS_PADRAO = {**GRADE[0], "peso_classes": True, "calibrar_limiar": True}

# Teto do peso de uma classe na perda. Sem ele, um indicador com 2 positivos em
# 400 exemplos entraria com peso 199 e o gradiente dele passaria a responder por
# quase toda a atualização — o modelo marcaria essa classe em tudo.
TETO_PESO_POSITIVO = 10.0

# Limiar quando não há como calibrar (nenhum positivo na validação, ou
# calibração desligada). Não é um bom corte, é o corte neutro.
LIMIAR_NEUTRO = 0.5

# Os candidatos da calibração: 0,05 a 0,95 de 0,05 em 0,05. Mais fino que isso
# só ajusta ao ruído de uma validação de algumas dezenas de exemplos.
LIMIARES_CANDIDATOS = [round(0.05 * i, 2) for i in range(1, 20)]


def dispositivo_disponivel(preferido: str = "auto") -> str:
    import torch

    if preferido == "cpu":
        return "cpu"
    if preferido == "cuda" or preferido == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if preferido == "cuda":
            raise RuntimeError(
                "CUDA foi pedida mas não está disponível. Instale o torch com"
                " CUDA ou use dispositivo='cpu'."
            )
    return "cpu"


def descrever_ambiente() -> dict:
    import torch

    return {
        "torch": torch.__version__,
        "cuda_disponivel": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dispositivo": dispositivo_disponivel(),
    }


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------

def _metricas(y_verdadeiro, y_previsto, rotulos: list[str]) -> dict:
    """Precisão, recall e F1 — macro, micro e por indicador.

    O macro é o número a reportar: ele dá o mesmo peso a cada indicador, então
    um indicador raro mal classificado aparece. O micro pondera pela frequência
    e mostra o desempenho agregado. A acurácia de subconjunto exige acertar a
    combinação inteira de uma notícia, o que é severo em multirrótulo — ela
    entra como referência, não como o número principal.

    Cada indicador tem a sua própria matriz de confusão 2x2 (marcou/não marcou),
    porque em multirrótulo não existe uma matriz única: a mesma notícia pode
    estar certa num indicador e errada em outro.
    """
    from sklearn.metrics import (
        accuracy_score, multilabel_confusion_matrix,
        precision_recall_fscore_support,
    )

    macro = precision_recall_fscore_support(
        y_verdadeiro, y_previsto, average="macro", zero_division=0
    )
    micro = precision_recall_fscore_support(
        y_verdadeiro, y_previsto, average="micro", zero_division=0
    )
    por_classe = precision_recall_fscore_support(
        y_verdadeiro, y_previsto, average=None, zero_division=0,
        labels=list(range(len(rotulos))),
    )
    matrizes = multilabel_confusion_matrix(
        y_verdadeiro, y_previsto, labels=list(range(len(rotulos)))
    )
    return {
        "acuracia_subconjunto": float(accuracy_score(y_verdadeiro, y_previsto)),
        "precisao_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precisao_micro": float(micro[0]),
        "recall_micro": float(micro[1]),
        "f1_micro": float(micro[2]),
        "por_classe": {
            rotulos[i]: {
                "precisao": float(por_classe[0][i]),
                "recall": float(por_classe[1][i]),
                "f1": float(por_classe[2][i]),
                "suporte": int(por_classe[3][i]),
            }
            for i in range(len(rotulos))
        },
        "matrizes_por_classe": {
            rotulos[i]: matrizes[i].tolist() for i in range(len(rotulos))
        },
        # Quantos indicadores o modelo marcou por notícia, em média, e quantos
        # o gabarito trazia. É a resposta direta a "ele está marcando só um?":
        # se o gabarito tem 1,8 e o modelo 0,9, ele está marcando de menos.
        "marcacoes_por_noticia": _marcacoes_por_noticia(y_verdadeiro, y_previsto),
    }


def _marcacoes_por_noticia(y_verdadeiro, y_previsto) -> dict:
    verdadeiro = np.asarray(y_verdadeiro).reshape(len(y_verdadeiro), -1)
    previsto = np.asarray(y_previsto).reshape(len(y_previsto), -1)
    if not len(verdadeiro):
        return {}
    por_noticia_v = verdadeiro.sum(axis=1)
    por_noticia_p = previsto.sum(axis=1)
    return {
        "media_verdadeira": float(por_noticia_v.mean()),
        "media_prevista": float(por_noticia_p.mean()),
        "maximo_verdadeiro": int(por_noticia_v.max()),
        "maximo_previsto": int(por_noticia_p.max()),
        # Notícias em que o gabarito pede mais de um indicador e o modelo
        # devolveu mais de um: a medida de "classifica vários ao mesmo tempo".
        "multiplos_verdadeiros": int((por_noticia_v > 1).sum()),
        "multiplos_previstos": int((por_noticia_p > 1).sum()),
        "nenhum_previsto": int((por_noticia_p == 0).sum()),
    }


# --------------------------------------------------------------------------
# Rótulos, pesos e limiares
# --------------------------------------------------------------------------

def _codificar_rotulo(rotulo: list[str], rotulos: list[str]) -> list[float]:
    """Lista de indicadores -> vetor de 0 e 1, uma posição por indicador. O
    vetor todo zero é o exemplo negativo e é tão válido quanto os outros."""
    vetor = [0.0] * len(rotulos)
    for nome in rotulo:
        if nome in rotulos:
            vetor[rotulos.index(nome)] = 1.0
    return vetor


def _conjunto(exemplos: list[dict], rotulos: list[str]) -> dict:
    """Codifica os exemplos de uma partição para a lista de rótulos dada.

    Recebe os exemplos já lidos do banco em vez de consultá-lo: na estratégia
    `por_indicador` a mesma partição é codificada uma vez por indicador, e
    reconsultar o banco sete vezes não mudaria nada além do tempo.
    """
    return {
        "textos": [e["texto"] for e in exemplos],
        "rotulos": [_codificar_rotulo(e["rotulo"], rotulos) for e in exemplos],
        "ids": [e["id"] for e in exemplos],
    }


def _pesos_positivos(rotulos_codificados: list[list[float]]) -> list[float]:
    """O peso de cada classe na perda: negativos / positivos, com teto.

    É o que impede o colapso em "não marca nada". Uma classe sem nenhum
    positivo no treino recebe peso 1 — não há o que pesar, e qualquer outro
    valor só amplificaria o nada.
    """
    matriz = np.asarray(rotulos_codificados, dtype=float)
    if matriz.size == 0:
        return []
    positivos = matriz.sum(axis=0)
    negativos = len(matriz) - positivos
    return [
        1.0 if pos <= 0
        else float(min(TETO_PESO_POSITIVO, max(1.0, neg / pos)))
        for pos, neg in zip(positivos, negativos)
    ]


def _calibrar_limiares(y, probabilidades, rotulos: list[str]) -> dict[str, float]:
    """O limiar de CADA indicador, escolhido pelo F1 dele na validação.

    Varre os candidatos de baixo para cima e guarda o primeiro que atinge o
    melhor F1. Empate fica com o limiar MAIS BAIXO de propósito: aqui um falso
    positivo é descartável — a LLM recebe o bloco daquela classe e pode
    descartá-la — enquanto um falso negativo é perda definitiva, porque a classe
    que o BERT não marcou nem chega a ser avaliada.

    Sem positivo na validação não há o que calibrar e a classe fica no corte
    neutro; o `motivo` registra isso, para que ninguém leia 0,5 como escolha.
    """
    from sklearn.metrics import f1_score

    y = np.asarray(y).reshape(len(y), -1) if len(y) else np.zeros((0, len(rotulos)))
    probabilidades = np.asarray(probabilidades)

    limiares, motivos = {}, {}
    for i, nome in enumerate(rotulos):
        if not len(y) or y[:, i].sum() == 0:
            limiares[nome] = LIMIAR_NEUTRO
            motivos[nome] = "sem positivo na validação: corte neutro"
            continue
        melhor, melhor_f1 = LIMIAR_NEUTRO, -1.0
        for candidato in LIMIARES_CANDIDATOS:
            f1 = f1_score(
                y[:, i], (probabilidades[:, i] >= candidato).astype(int),
                zero_division=0,
            )
            if f1 > melhor_f1 + 1e-12:
                melhor, melhor_f1 = candidato, f1
        limiares[nome] = melhor
        motivos[nome] = f"F1 {melhor_f1:.3f} na validação"
    return {"limiares": limiares, "motivos": motivos}


def _aplicar_limiares(probabilidades, rotulos: list[str],
                      limiares: dict[str, float]):
    """Cada coluna contra o SEU corte. É aqui que o multirrótulo acontece: a
    comparação é vetorial e todas as classes acima do próprio limiar entram, não
    só a de maior pontuação."""
    probabilidades = np.asarray(probabilidades)
    if not probabilidades.size:
        return np.zeros((0, len(rotulos)), dtype=int)
    cortes = np.array([limiares.get(n, LIMIAR_NEUTRO) for n in rotulos])
    return (probabilidades >= cortes).astype(int)


# --------------------------------------------------------------------------
# Ajuste de uma rede
# --------------------------------------------------------------------------

def _ajustar(modelo: str, hiper: dict, rotulos: list[str], conjuntos: dict,
             aparelho: str) -> dict:
    """Treina UMA rede e devolve as PROBABILIDADES de cada partição.

    Devolve probabilidade e não decisão: o corte vem depois, da calibração na
    validação. Uma rede com `len(rotulos) == 1` é uma cabeça binária da
    estratégia `por_indicador`; com oito, é o modelo multirrótulo inteiro. O
    código é o mesmo nos dois casos, e é por isso que as estratégias são
    comparáveis.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    tokenizador = AutoTokenizer.from_pretrained(MODELOS[modelo]["id"])
    rede = AutoModelForSequenceClassification.from_pretrained(
        MODELOS[modelo]["id"],
        num_labels=len(rotulos),
        problem_type="multi_label_classification",
    ).to(aparelho)

    class Lote(Dataset):
        def __init__(self, dados):
            self.codificado = tokenizador(
                dados["textos"], truncation=True, padding="max_length",
                max_length=hiper["max_length"], return_tensors="pt",
            )
            self.rotulos = torch.tensor(dados["rotulos"], dtype=torch.float)

        def __len__(self):
            return len(self.rotulos)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.codificado.items()}
            item["labels"] = self.rotulos[i]
            return item

    treino = conjuntos["treino"]
    carregador = DataLoader(
        Lote(treino), batch_size=hiper["batch_size"], shuffle=True
    )

    # A perda é montada aqui, e não deixada para o `transformers`, só por causa
    # do `pos_weight`: a perda interna é BCE sem peso, e com ela um indicador
    # raro é mais barato de ignorar do que de aprender.
    pesos = _pesos_positivos(treino["rotulos"]) if hiper.get("peso_classes", True) else None
    perda_de = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pesos, dtype=torch.float).to(aparelho)
        if pesos else None
    )

    otimizador = torch.optim.AdamW(
        rede.parameters(),
        lr=hiper["learning_rate"],
        weight_decay=hiper["weight_decay"],
    )
    total_passos = max(1, len(carregador) * hiper["epocas"])
    agendador = get_linear_schedule_with_warmup(
        otimizador, int(0.1 * total_passos), total_passos
    )

    def probabilidades(dados):
        """Sigmoide por indicador, nunca softmax: as classes não competem entre
        si, cada uma decide sozinha se passa do seu limiar."""
        if not dados["textos"]:
            return (
                np.zeros((0, len(rotulos))),
                np.zeros((0, len(rotulos)), dtype=int),
            )
        rede.eval()
        pontuadas, verdadeiros = [], []
        with torch.no_grad():
            for lote in DataLoader(Lote(dados), batch_size=16):
                alvos = lote.pop("labels")
                saida = rede(**{k: v.to(aparelho) for k, v in lote.items()})
                pontuadas.extend(torch.sigmoid(saida.logits).cpu().numpy())
                verdadeiros.extend(alvos.numpy().astype(int))
        return np.array(pontuadas), np.array(verdadeiros)

    historico = []
    for epoca in range(hiper["epocas"]):
        rede.train()
        perda_total = 0.0
        for lote in carregador:
            alvos = lote.pop("labels").to(aparelho)
            otimizador.zero_grad()
            logits = rede(**{k: v.to(aparelho) for k, v in lote.items()}).logits
            perda = perda_de(logits, alvos)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
            otimizador.step()
            agendador.step()
            perda_total += perda.item()

        registro = {
            "epoca": epoca + 1,
            "perda_treino": perda_total / max(1, len(carregador)),
        }
        p_val, y_val = probabilidades(conjuntos["validacao"])
        if len(y_val):
            # Curva de aprendizado no corte neutro, de propósito: o limiar
            # calibrado muda a cada época e a curva deixaria de mostrar o
            # modelo aprendendo para mostrar a calibração mexendo.
            registro["f1_macro_validacao"] = _metricas(
                y_val, (p_val >= LIMIAR_NEUTRO).astype(int), rotulos
            )["f1_macro"]
        historico.append(registro)

    saida = {
        "rede": rede,
        "tokenizador": tokenizador,
        "historico": historico,
        "pesos_positivos": dict(zip(rotulos, pesos)) if pesos else {},
        "probabilidades": {},
    }
    for particao in ("validacao", "teste"):
        p, y = probabilidades(conjuntos[particao])
        saida["probabilidades"][particao] = {"p": p, "y": y}
    return saida


def _descartar(cabeca: dict, aparelho: str) -> None:
    """Solta a rede da memória. Em `por_indicador` são sete redes em sequência:
    sem isto, a sétima não cabe na placa."""
    import torch

    cabeca.pop("rede", None)
    cabeca.pop("tokenizador", None)
    if aparelho == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Treino
# --------------------------------------------------------------------------

def _apurar_cluster(probabilidades_voto: dict, exemplos: dict,
                    rotulos: list[str]) -> dict:
    """Calibra cada rede do cluster e apura o voto no teste.

    Cada uma das três redes de um indicador tem o SEU limiar, calibrado na
    validação com os dados dela — e não um limiar do indicador. Isso é o ponto:
    se as três compartilhassem o corte, discordariam menos por construção, e a
    discordância é justamente o sinal que manda o caso para uma pessoa.

    Devolve o voto por indicador e a decisão em três faixas:
        0 votos            ausente
        1 ou 2 votos       vai para revisão humana
        3 votos            classificado
    """
    from sklearn.metrics import f1_score

    y = {
        particao: np.array(
            [
                _codificar_rotulo(e["rotulo"], rotulos)
                for e in exemplos[particao]
            ],
            dtype=int,
        ).reshape(len(exemplos[particao]), len(rotulos))
        for particao in ("validacao", "teste")
    }

    limiares: dict[str, float] = {}
    motivos: dict[str, str] = {}
    votos = np.zeros((len(exemplos["teste"]), len(rotulos)), dtype=int)
    votos_val = np.zeros((len(exemplos["validacao"]), len(rotulos)), dtype=int)

    for i, nome in enumerate(rotulos):
        for chave, por_particao in probabilidades_voto[nome].items():
            p_val = por_particao.get("validacao")
            alvo_val = y["validacao"][:, i] if len(y["validacao"]) else []

            if p_val is None or not len(alvo_val) or alvo_val.sum() == 0:
                limiares[chave] = LIMIAR_NEUTRO
                motivos[chave] = "sem positivo na validação: corte neutro"
            else:
                melhor, melhor_f1 = LIMIAR_NEUTRO, -1.0
                for candidato in LIMIARES_CANDIDATOS:
                    f1 = f1_score(
                        alvo_val, (p_val >= candidato).astype(int),
                        zero_division=0,
                    )
                    if f1 > melhor_f1 + 1e-12:
                        melhor, melhor_f1 = candidato, f1
                limiares[chave] = melhor
                motivos[chave] = f"F1 {melhor_f1:.3f} na validação"

            p_teste = por_particao.get("teste")
            if p_teste is not None and len(p_teste):
                votos[:, i] += (p_teste >= limiares[chave]).astype(int)
            # O mesmo voto na validação. Sem ele o cluster seria escolhido pelo
            # TESTE enquanto as outras estratégias são escolhidas pela
            # validação — e aí a comparação entre elas deixaria de valer.
            if p_val is not None and len(p_val):
                votos_val[:, i] += (p_val >= limiares[chave]).astype(int)

    precisa = votos_para_prosseguir(len(MODELOS_CLUSTER))
    classificado = (votos >= precisa).astype(int)
    duvidoso = ((votos > 0) & (votos < precisa)).astype(int)

    total_celulas = votos.size or 1
    return {
        "limiares": limiares,
        "motivos": motivos,
        "votos": votos,
        "y": y,
        "classificado": classificado,
        "classificado_validacao": (
            votos_val >= VOTOS_PARA_CLASSIFICAR
        ).astype(int),
        # O que este desenho CUSTA, medido: a fração de decisões que não é
        # automática — os votos que ficaram entre 1 e o corte. Sem este
        # número, exigir concordância parece de graça.
        "revisao": {
            "celulas_duvidosas": int(duvidoso.sum()),
            "celulas": int(total_celulas),
            "fracao": float(duvidoso.sum()) / total_celulas,
            "noticias_com_duvida": int((duvidoso.sum(axis=1) > 0).sum()),
            "noticias": int(votos.shape[0]),
        },
        "distribuicao_votos": {
            str(n): int((votos == n).sum())
            for n in range(len(MODELOS_CLUSTER) + 1)
        },
        "votos_para_prosseguir": precisa,
        "n_modelos": len(MODELOS_CLUSTER),
    }


def treinar(
    con,
    tarefa: str,
    modelo: str,
    hiperparametros: dict | None = None,
    dispositivo: str = "auto",
    salvar_modelo: bool = False,
    registrar: bool = True,
    classes: list[str] | None = None,
    estrategia: str = ESTRATEGIA_PADRAO,
    progresso=None,
) -> dict:
    """Treina uma estratégia numa tarefa e devolve as métricas no teste.

    O resultado tem o mesmo formato nas três estratégias — mesma partição de
    teste, mesmas métricas, mesma tabela — porque é essa igualdade que permite
    comparar "uma rede com oito saídas" contra "oito redes binárias" contra
    "o trio votando" sem ressalva.
    """
    if estrategia == "cluster":
        # O cluster usa os três modelos da constante; o argumento `modelo` só
        # serve de etiqueta na execução, e aceitar outro valor daria a impressão
        # falsa de que ele muda o que vai ser treinado.
        modelo = "cluster"
    elif modelo not in MODELOS:
        raise ValueError(f"Modelo desconhecido: {modelo}")
    if tarefa not in corpus.TAREFAS:
        raise ValueError(f"Tarefa desconhecida: {tarefa}")
    if estrategia not in ESTRATEGIAS:
        raise ValueError(
            f"Estratégia desconhecida: {estrategia}."
            f" Use uma de {', '.join(ESTRATEGIAS)}."
        )

    hiper = {**HIPERPARAMETROS_PADRAO, **(hiperparametros or {})}
    # `classes` restringe o treino a um subconjunto. Serve para medir uma
    # classe isolada sem que as outras disputem a capacidade do modelo, e para
    # treinar só o que já tem exemplo suficiente enquanto o corpus cresce. A
    # lista usada fica gravada no manifesto junto com os pesos, e é dela que a
    # inferência lê as posições do vetor.
    rotulos = corpus.rotulos_da_tarefa(tarefa, classes)
    aparelho = dispositivo_disponivel(dispositivo)
    inicio = time.monotonic()

    if not rotulos:
        raise RuntimeError("Nenhuma classe para treinar.")

    exemplos = {p: corpus.exemplos(con, tarefa, p) for p in corpus.PARTICOES}
    if not exemplos["treino"]:
        raise RuntimeError(
            "Não há exemplos de treino. Rotule notícias no repositório e rode a"
            " divisão em treino/validação/teste."
        )
    if not exemplos["teste"]:
        raise RuntimeError(
            "Não há exemplos de teste. Sem conjunto de teste não existe matriz"
            " de confusão para reportar."
        )

    # Uma "unidade" é uma rede a treinar, e cada uma carrega o seu próprio
    # modelo-base. Em `unico` é uma só, com todos os rótulos; em
    # `por_indicador`, uma por indicador; em `cluster`, uma por indicador E por
    # modelo-base — 3 x 8 = 24 redes hoje, treinadas e descartadas uma a uma,
    # que é o que as faz caber em 8 GB de VRAM.
    if estrategia == "unico":
        unidades = [("multirrotulo", rotulos, modelo)]
    elif estrategia == "por_indicador":
        unidades = [(nome, [nome], modelo) for nome in rotulos]
    else:
        # A POSIÇÃO entra no nome, não só o modelo. Sem ela, repetir um modelo
        # em MODELOS_CLUSTER — três sementes do mesmo, que é ensemble legítimo —
        # faria as três unidades do indicador terem o mesmo nome e se
        # sobrescreverem. Medido antes da correção: 21 treinadas, 7 gravadas.
        unidades = [
            (f"{nome}__{posicao}_{base}", [nome], base)
            for nome in rotulos
            for posicao, base in enumerate(MODELOS_CLUSTER, start=1)
        ]

    # No cluster, a probabilidade de cada voto fica guardada separada: o voto é
    # por REDE, e juntar as três numa coluna só (média, máximo) apagaria
    # exatamente a discordância que manda o caso para revisão humana. A chave é
    # o NOME DA UNIDADE, que é único; indexar pelo modelo-base perderia uma das
    # redes quando o mesmo modelo aparece duas vezes no trio.
    probabilidades_voto: dict[str, dict[str, dict]] = {
        nome: {} for nome in rotulos
    }
    if estrategia == "cluster":
        for nome_unidade, rotulos_unidade, _ in unidades:
            probabilidades_voto[rotulos_unidade[0]][nome_unidade] = {}

    # Matriz conjunta: uma coluna por indicador, preenchida pela unidade que o
    # treinou. Em `por_indicador` é aqui que as sete saídas binárias voltam a
    # ser um vetor multirrótulo — e é por isso que o resultado final das duas
    # estratégias tem exatamente a mesma forma.
    conjunta = {
        particao: {
            "p": np.zeros((len(exemplos[particao]), len(rotulos))),
            "y": np.array(
                [
                    _codificar_rotulo(e["rotulo"], rotulos)
                    for e in exemplos[particao]
                ],
                dtype=int,
            ).reshape(len(exemplos[particao]), len(rotulos)),
        }
        for particao in ("validacao", "teste")
    }

    cabecas: dict[str, dict] = {}
    pesos_usados: dict[str, float] = {}
    salvas: dict[str, str] = {}
    historico_unico: list[dict] = []

    for ordem, (nome_unidade, rotulos_unidade, base_unidade) in enumerate(
        unidades, start=1
    ):
        if progresso:
            progresso(ordem, len(unidades), nome_unidade)

        conjuntos = {
            particao: _conjunto(exemplos[particao], rotulos_unidade)
            for particao in corpus.PARTICOES
        }
        ajustada = _ajustar(
            base_unidade, hiper, rotulos_unidade, conjuntos, aparelho
        )
        colunas = [rotulos.index(n) for n in rotulos_unidade]
        for particao in ("validacao", "teste"):
            previstas = ajustada["probabilidades"][particao]["p"]
            if previstas.size:
                if estrategia == "cluster":
                    probabilidades_voto[rotulos_unidade[0]][nome_unidade][
                        particao
                    ] = previstas[:, 0]
                else:
                    conjunta[particao]["p"][:, colunas] = previstas

        pesos_usados.update(ajustada["pesos_positivos"])
        positivos = int(np.asarray(conjuntos["treino"]["rotulos"]).sum())
        if estrategia == "unico":
            historico_unico = ajustada["historico"]
        else:
            cabecas[nome_unidade] = {
                "historico": ajustada["historico"],
                "n_positivos_treino": positivos,
                "peso_positivo": ajustada["pesos_positivos"].get(nome_unidade),
            }

        if salvar_modelo:
            destino = _pasta_da_unidade(tarefa, estrategia, nome_unidade)
            destino.mkdir(parents=True, exist_ok=True)
            ajustada["rede"].save_pretrained(destino)
            ajustada["tokenizador"].save_pretrained(destino)
            salvas[nome_unidade] = destino.relative_to(
                PASTA_MODELOS / tarefa
            ).as_posix()

        _descartar(ajustada, aparelho)

    if estrategia == "cluster":
        # Treinou 21 e gravou 7? Isso já aconteceu, por colisão de nome, e
        # passou calado. Nenhuma rede ajustada pode desaparecer sem erro: o
        # tempo de GPU foi gasto e o resultado seria um cluster incompleto se
        # fazendo passar por completo.
        if salvar_modelo and len(salvas) != len(unidades):
            raise RuntimeError(
                f"O cluster ajustou {len(unidades)} redes mas gravou"
                f" {len(salvas)}. Alguma unidade sobrescreveu outra — confira se"
                " há nome repetido em MODELOS_CLUSTER."
            )
        apuradas = sum(len(v) for v in probabilidades_voto.values())
        if apuradas != len(unidades):
            raise RuntimeError(
                f"O cluster ajustou {len(unidades)} redes mas só"
                f" {apuradas} entraram na apuração do voto."
            )
        apuracao = _apurar_cluster(probabilidades_voto, exemplos, rotulos)
        metricas = _metricas(
            apuracao["y"]["teste"], apuracao["classificado"], rotulos
        )
        metricas["limiares"] = apuracao["limiares"]
        metricas["limiares_motivo"] = apuracao["motivos"]
        metricas["pesos_positivos"] = pesos_usados
        metricas["revisao_humana"] = apuracao["revisao"]
        metricas["distribuicao_votos"] = apuracao["distribuicao_votos"]
        metricas["votos_para_classificar"] = apuracao["votos_para_prosseguir"]
        metricas["n_modelos"] = apuracao["n_modelos"]
        # O que seria exigindo UNANIMIDADE. Fica ao lado porque a decisão de
        # baixar para 2 de 3 precisa poder ser revista com número, e não de
        # memória.
        metricas["com_unanimidade"] = _metricas(
            apuracao["y"]["teste"],
            (apuracao["votos"] >= len(MODELOS_CLUSTER)).astype(int),
            rotulos,
        )
        # Qual seria o resultado se bastasse UM voto. Com `com_unanimidade`
        # ao lado, os três números cercam a escolha do corte: 1, 2 e 3
        # votos medidos no mesmo teste.
        metricas["com_um_voto"] = _metricas(
            apuracao["y"]["teste"],
            (apuracao["votos"] >= 1).astype(int),
            rotulos,
        )
        metricas["cabecas"] = cabecas
        metricas["f1_macro_validacao_final"] = (
            _metricas(
                apuracao["y"]["validacao"],
                apuracao["classificado_validacao"],
                rotulos,
            )["f1_macro"]
            if len(apuracao["y"]["validacao"]) else None
        )

        caminho = None
        if salvar_modelo:
            caminho = str(
                _gravar_manifesto(
                    tarefa, estrategia, modelo, rotulos,
                    apuracao["limiares"], hiper, salvas,
                )
            )
        resultado = {
            "tarefa": tarefa,
            "estrategia": estrategia,
            "modelo": "cluster(" + "+".join(MODELOS_CLUSTER) + ")",
            "hiperparametros": hiper,
            "metricas": metricas,
            "rotulos": rotulos,
            "limiares": apuracao["limiares"],
            "historico": [],
            "n_treino": len(exemplos["treino"]),
            "n_validacao": len(exemplos["validacao"]),
            "n_teste": len(exemplos["teste"]),
            "n_redes": len(unidades),
            "duracao_s": time.monotonic() - inicio,
            "dispositivo": aparelho,
            "caminho_modelo": caminho,
        }
        if registrar:
            resultado["id"] = registrar_execucao(con, resultado)
        return resultado

    # Calibra na validação e mede no teste. Nunca o contrário.
    calibracao = (
        _calibrar_limiares(
            conjunta["validacao"]["y"], conjunta["validacao"]["p"], rotulos
        )
        if hiper.get("calibrar_limiar", True)
        else {
            "limiares": {n: LIMIAR_NEUTRO for n in rotulos},
            "motivos": {n: "calibração desligada" for n in rotulos},
        }
    )
    limiares = calibracao["limiares"]

    y_teste = conjunta["teste"]["y"]
    p_teste = conjunta["teste"]["p"]
    metricas = _metricas(
        y_teste, _aplicar_limiares(p_teste, rotulos, limiares), rotulos
    )
    metricas["limiares"] = limiares
    metricas["limiares_motivo"] = calibracao["motivos"]
    metricas["pesos_positivos"] = pesos_usados
    # O mesmo modelo com todos os cortes em 0,5. A diferença entre este número
    # e o de cima é o efeito da calibração, medido — e é ele que mostra, preto
    # no branco, quantos indicadores o corte fixo estava engolindo.
    metricas["sem_calibracao"] = _metricas(
        y_teste, (p_teste >= LIMIAR_NEUTRO).astype(int), rotulos
    )
    if cabecas:
        metricas["cabecas"] = cabecas

    y_val = conjunta["validacao"]["y"]
    metricas["f1_macro_validacao_final"] = (
        _metricas(
            y_val, _aplicar_limiares(conjunta["validacao"]["p"], rotulos, limiares),
            rotulos,
        )["f1_macro"]
        if len(y_val) else None
    )

    caminho = None
    if salvar_modelo:
        caminho = str(
            _gravar_manifesto(
                tarefa, estrategia, modelo, rotulos, limiares, hiper, salvas
            )
        )

    resultado = {
        "tarefa": tarefa,
        "estrategia": estrategia,
        "modelo": modelo,
        "hiperparametros": hiper,
        "metricas": metricas,
        "rotulos": rotulos,
        "limiares": limiares,
        "historico": historico_unico,
        "n_treino": len(exemplos["treino"]),
        "n_validacao": len(exemplos["validacao"]),
        "n_teste": len(exemplos["teste"]),
        "n_redes": len(unidades),
        "duracao_s": time.monotonic() - inicio,
        "dispositivo": aparelho,
        "caminho_modelo": caminho,
    }

    if registrar:
        resultado["id"] = registrar_execucao(con, resultado)
    return resultado


# --------------------------------------------------------------------------
# Pesos em disco
# --------------------------------------------------------------------------

def _pasta_da_unidade(tarefa: str, estrategia: str, nome_unidade: str) -> Path:
    """Cada estratégia grava na sua própria subpasta.

    Separadas porque as duas podem conviver em disco: treinar `por_indicador`
    não apaga o modelo `unico` que estava em produção, e o manifesto é o que
    diz qual dos dois responde. Trocar de estratégia é reescrever o manifesto,
    não apagar pesos.
    """
    base = PASTA_MODELOS / tarefa
    if estrategia == "unico":
        return base / "unico"
    if estrategia == "cluster":
        # `nome_unidade` já vem como `indicador__modelo`: uma pasta por rede, e
        # o nome diz qual indicador e qual modelo-base, que é o que a inferência
        # precisa para montar o voto de novo.
        return base / "cluster" / nome_unidade
    return base / "por_indicador" / nome_unidade


def _gravar_manifesto(tarefa: str, estrategia: str, modelo: str,
                      rotulos: list[str], limiares: dict[str, float],
                      hiper: dict, pastas: dict[str, str]) -> Path:
    """O contrato entre o treino e a inferência.

    Grava a ordem dos rótulos, o limiar de cada um, o `max_length` usado e onde
    estão os pesos. A inferência não deduz nada disso: um indicador criado em
    /gestao depois do treino não existe nos pesos, e ler o catálogo atual
    deslocaria todas as posições do vetor.
    """
    destino = PASTA_MODELOS / tarefa
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / MANIFESTO
    arquivo.write_text(
        json.dumps(
            {
                "tarefa": tarefa,
                "estrategia": estrategia,
                "modelo": modelo,
                # No cluster não há UM modelo-base: há três por indicador, e o
                # catálogo não tem entrada "cluster". A lista dos ids vai no
                # lugar, que é o que torna o manifesto auditável.
                "modelo_id": (
                    [MODELOS[m]["id"] for m in MODELOS_CLUSTER]
                    if estrategia == "cluster"
                    else MODELOS[modelo]["id"]
                ),
                "modelos_cluster": (
                    MODELOS_CLUSTER if estrategia == "cluster" else None
                ),
                "votos_para_classificar": (
                    votos_para_prosseguir(len(MODELOS_CLUSTER))
                    if estrategia == "cluster" else None
                ),
                "n_modelos": (
                    len(MODELOS_CLUSTER) if estrategia == "cluster" else 1
                ),
                "rotulos": rotulos,
                "limiares": limiares,
                "max_length": hiper["max_length"],
                "hiperparametros": hiper,
                "pastas": pastas,
                "treinado_em": agora(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return arquivo


def ler_manifesto(tarefa: str) -> dict | None:
    arquivo = PASTA_MODELOS / tarefa / MANIFESTO
    if not arquivo.exists():
        return None
    try:
        return json.loads(arquivo.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------
# Registro no banco
# --------------------------------------------------------------------------

def registrar_execucao(con, resultado: dict) -> int:
    metricas = resultado["metricas"]
    cursor = con.execute(
        "INSERT INTO treino_execucao (tarefa, estrategia, modelo, hiperparametros,"
        " metricas, matriz_confusao, rotulos, limiares, historico, n_treino,"
        " n_validacao, n_teste, n_redes, duracao_s, dispositivo, caminho_modelo,"
        " erro, criado_em) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            resultado["tarefa"],
            resultado.get("estrategia", ESTRATEGIA_PADRAO),
            resultado["modelo"],
            json.dumps(resultado["hiperparametros"], ensure_ascii=False),
            json.dumps(metricas, ensure_ascii=False),
            json.dumps(
                metricas.get("matriz_confusao") or metricas.get("matrizes_por_classe"),
                ensure_ascii=False,
            ),
            json.dumps(resultado["rotulos"], ensure_ascii=False),
            json.dumps(resultado.get("limiares") or {}, ensure_ascii=False),
            json.dumps(resultado["historico"], ensure_ascii=False),
            resultado["n_treino"],
            resultado["n_validacao"],
            resultado["n_teste"],
            resultado.get("n_redes"),
            resultado["duracao_s"],
            resultado["dispositivo"],
            resultado["caminho_modelo"],
            resultado.get("erro"),
            agora(),
        ),
    )
    return cursor.lastrowid


def registrar_falha(con, tarefa: str, estrategia: str, modelo: str, hiper: dict,
                    erro: str) -> None:
    """Uma falha também é resultado: sem registrá-la, a tabela do TCC daria a
    entender que o modelo não foi testado."""
    con.execute(
        "INSERT INTO treino_execucao (tarefa, estrategia, modelo, hiperparametros,"
        " erro, criado_em) VALUES (?,?,?,?,?,?)",
        (tarefa, estrategia, modelo, json.dumps(hiper, ensure_ascii=False), erro,
         agora()),
    )


# --------------------------------------------------------------------------
# Comparação
# --------------------------------------------------------------------------

def _chave_de_selecao(resultado: dict) -> float:
    """Escolhe pela validação, não pelo teste. Quando não há validação (classe
    rara demais para sobrar exemplo), cai para o teste e isso fica registrado,
    porque o número deixa de ser uma estimativa limpa."""
    metricas = resultado["metricas"]
    validacao = metricas.get("f1_macro_validacao_final")
    return validacao if validacao is not None else metricas["f1_macro"]


def placar(execucoes: list[dict]) -> list[dict]:
    """O melhor resultado de CADA par (estratégia, modelo), já ordenado.

    A tabela completa lista toda execução da grade: com seis modelos e quatro
    configurações são 24 linhas por estratégia, e descobrir ali qual MODELO
    ganha exige comparar à mão as quatro linhas de cada um. Esta reduz a uma
    linha por modelo — a melhor configuração dele — e ordena pelo mesmo critério
    que o treino usa para escolher os pesos que vão para produção: F1 macro na
    VALIDAÇÃO, nunca no teste.

    Aceita tanto o resultado de `treinar` quanto a linha lida do banco: as duas
    formas têm `estrategia`, `modelo` e `metricas`.
    """
    melhores: dict[tuple[str, str], dict] = {}
    quantas: dict[tuple[str, str], int] = {}

    for execucao in execucoes:
        metricas = execucao.get("metricas") or {}
        if metricas.get("f1_macro") is None:
            # Execução que falhou: não tem métrica para entrar no placar, e já
            # aparece na lista de falhas.
            continue
        chave = (execucao.get("estrategia") or ESTRATEGIA_PADRAO, execucao["modelo"])
        quantas[chave] = quantas.get(chave, 0) + 1
        atual = melhores.get(chave)
        if atual is None or _chave_de_selecao(execucao) > _chave_de_selecao(atual):
            melhores[chave] = execucao

    linhas = [
        {
            "estrategia": chave[0],
            "modelo": chave[1],
            "execucao": execucao,
            "n_execucoes": quantas[chave],
            # Deixa explícito quando o número não é uma estimativa limpa: sem
            # validação, a escolha caiu para o próprio teste.
            "criterio": (
                "validação"
                if execucao["metricas"].get("f1_macro_validacao_final") is not None
                else "teste (não havia validação)"
            ),
        }
        for chave, execucao in melhores.items()
    ]
    linhas.sort(key=lambda linha: -_chave_de_selecao(linha["execucao"]))
    return linhas


def contar_treinos(modelos: list[str] | None, grade: list[dict] | None,
                   estrategias: list[str] | None, n_classes: int) -> int:
    """Quantas redes uma comparação vai ajustar. Serve para avisar ANTES: com
    `unico` e `por_indicador`, três modelos e quatro configurações, são 12
    redes contra 96 — `por_indicador` sozinha responde por 8 de cada 9."""
    modelos = modelos or list(MODELOS)
    grade = grade or GRADE
    estrategias = estrategias or [ESTRATEGIA_PADRAO]
    por_estrategia = {"unico": 1, "por_indicador": max(1, n_classes)}
    total = 0
    for estrategia in estrategias:
        if estrategia == "cluster":
            # O cluster não varia por modelo-base: ele USA os três. São
            # 3 x classes redes por configuração da grade, uma vez só.
            total += len(grade) * len(MODELOS_CLUSTER) * max(1, n_classes)
        else:
            total += (
                len(modelos) * len(grade) * por_estrategia.get(estrategia, 1)
            )
    return total


def rodar_comparacao(
    con,
    tarefa: str,
    modelos: list[str] | None = None,
    grade: list[dict] | None = None,
    dispositivo: str = "auto",
    salvar_melhor: bool = True,
    progresso=None,
    classes: list[str] | None = None,
    estrategias: list[str] | None = None,
) -> dict:
    """Treina cada modelo com cada configuração, em cada estratégia pedida.

    Com mais de uma estratégia na lista, a tabela devolvida as compara na
    MESMA partição de teste — é a resposta medida para "vale um modelo por
    indicador?", e não uma opinião.

    Falhas (falta de memória na GPU, modelo indisponível) são registradas e não
    interrompem a comparação: um modelo que não cabe na placa é um resultado
    relevante para o TCC, não um motivo para perder as outras execuções.
    """
    modelos = modelos or list(MODELOS)
    grade = grade or GRADE
    estrategias = estrategias or [ESTRATEGIA_PADRAO]
    for estrategia in estrategias:
        if estrategia not in ESTRATEGIAS:
            raise ValueError(f"Estratégia desconhecida: {estrategia}")

    execucoes, falhas = [], []
    # Quantas EXECUÇÕES (não redes): o cluster é uma execução por configuração,
    # porque ele não varia por modelo-base — repetir por modelo treinaria três
    # vezes exatamente a mesma coisa.
    total = sum(
        len(grade) if e == "cluster" else len(modelos) * len(grade)
        for e in estrategias
    )
    feitos = 0

    for estrategia in estrategias:
        # Uma volta só no cluster; o nome é etiqueta, os pesos vêm dos três.
        candidatos = ["cluster"] if estrategia == "cluster" else modelos
        for modelo in candidatos:
            for hiper in grade:
                feitos += 1
                if progresso:
                    progresso(feitos, total, estrategia, modelo, hiper)
                try:
                    execucoes.append(
                        treinar(con, tarefa, modelo, hiper, dispositivo,
                                salvar_modelo=False, classes=classes,
                                estrategia=estrategia)
                    )
                except Exception as erro:
                    detalhe = f"{type(erro).__name__}: {erro}"
                    registrar_falha(con, tarefa, estrategia, modelo, hiper, detalhe)
                    falhas.append({
                        "estrategia": estrategia, "modelo": modelo,
                        "hiperparametros": hiper, "erro": str(erro),
                    })

    melhor = None
    if execucoes:
        melhor = max(execucoes, key=_chave_de_selecao)
        if salvar_melhor:
            # Retreina a configuração vencedora só para gravar os pesos; a
            # métrica reportada continua sendo a da execução original. Em
            # `por_indicador` isso significa ajustar as sete redes de novo — é
            # o preço de não guardar sete checkpoints de cada execução.
            treinar(
                con, tarefa, melhor["modelo"], melhor["hiperparametros"],
                dispositivo, salvar_modelo=True, registrar=False,
                classes=classes, estrategia=melhor["estrategia"],
            )

    return {
        "tarefa": tarefa,
        "execucoes": execucoes,
        "falhas": falhas,
        "melhor": melhor,
        "estrategias": estrategias,
        "ambiente": descrever_ambiente(),
    }
