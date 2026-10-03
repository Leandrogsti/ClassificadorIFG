"""Treino e avaliação dos classificadores, antes da camada de LLM.

Duas tarefas encadeadas, descritas em corpus.py: `violencia` decide se a notícia
relata disparo de arma de fogo, e só o que passa por ela segue para `motivacao`
e `indicadores`.

Cada execução treina UM modelo com UM conjunto de hiperparâmetros e mede no
conjunto de teste, que é fixo (gravado em corpus_noticia.particao). Os
hiperparâmetros são escolhidos pelo F1 macro na VALIDAÇÃO, nunca no teste —
olhar o teste para escolher a configuração vaza informação e infla o resultado
que vai para a banca.

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

# Os seis modelos da comparação. A escolha cobre três eixos que importam para o
# TCC: monolíngue português vs multilíngue, tamanho (base vs large) e modelo
# destilado vs completo.
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
    "xlm-roberta-base": {
        "id": "FacebookAI/xlm-roberta-base",
        "rotulo": "XLM-RoBERTa base",
        "parametros": "279M",
        "tipo": "multilíngue (100 idiomas)",
    },
    "mbert": {
        "id": "google-bert/bert-base-multilingual-cased",
        "rotulo": "mBERT",
        "parametros": "178M",
        "tipo": "multilíngue (104 idiomas)",
    },
    "distilbert-multilingual": {
        "id": "distilbert/distilbert-base-multilingual-cased",
        "rotulo": "DistilBERT multilingual",
        "parametros": "134M",
        "tipo": "multilíngue destilado",
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

HIPERPARAMETROS_PADRAO = GRADE[0]


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


def _metricas(tarefa: str, y_verdadeiro, y_previsto, rotulos: list[str]) -> dict:
    """Acurácia, precisão, recall e F1 — macro, ponderado e por classe.

    O macro é o número a reportar: ele dá o mesmo peso a cada classe, então uma
    categoria rara mal classificada aparece. A acurácia sozinha engana num
    corpus desbalanceado, onde chutar sempre a classe maior já acerta muito.
    """
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, precision_recall_fscore_support,
    )

    if tarefa == "indicadores":
        from sklearn.metrics import multilabel_confusion_matrix

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
        matrizes = multilabel_confusion_matrix(y_verdadeiro, y_previsto)
        return {
            "acuracia_subconjunto": float(
                accuracy_score(y_verdadeiro, y_previsto)
            ),
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
        }

    macro = precision_recall_fscore_support(
        y_verdadeiro, y_previsto, average="macro", zero_division=0
    )
    ponderado = precision_recall_fscore_support(
        y_verdadeiro, y_previsto, average="weighted", zero_division=0
    )
    por_classe = precision_recall_fscore_support(
        y_verdadeiro, y_previsto, average=None, zero_division=0,
        labels=list(range(len(rotulos))),
    )
    return {
        "acuracia": float(accuracy_score(y_verdadeiro, y_previsto)),
        "precisao_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precisao_ponderada": float(ponderado[0]),
        "recall_ponderado": float(ponderado[1]),
        "f1_ponderado": float(ponderado[2]),
        "por_classe": {
            rotulos[i]: {
                "precisao": float(por_classe[0][i]),
                "recall": float(por_classe[1][i]),
                "f1": float(por_classe[2][i]),
                "suporte": int(por_classe[3][i]),
            }
            for i in range(len(rotulos))
        },
        "matriz_confusao": confusion_matrix(
            y_verdadeiro, y_previsto, labels=list(range(len(rotulos)))
        ).tolist(),
    }


def _codificar_rotulo(tarefa: str, rotulo, rotulos: list[str]):
    if tarefa == "violencia":
        return int(rotulo)
    if tarefa == "motivacao":
        return rotulos.index(rotulo)
    vetor = [0.0] * len(rotulos)
    for nome in rotulo:
        vetor[rotulos.index(nome)] = 1.0
    return vetor


def _conjunto(con, tarefa: str, particao: str, rotulos: list[str]) -> dict:
    exemplos = corpus.exemplos(con, tarefa, particao)
    return {
        "textos": [e["texto"] for e in exemplos],
        "rotulos": [_codificar_rotulo(tarefa, e["rotulo"], rotulos) for e in exemplos],
        "ids": [e["id"] for e in exemplos],
    }


def treinar(
    con,
    tarefa: str,
    modelo: str,
    hiperparametros: dict | None = None,
    dispositivo: str = "auto",
    salvar_modelo: bool = False,
    registrar: bool = True,
) -> dict:
    """Treina um modelo numa tarefa e devolve as métricas no conjunto de teste."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup,
    )

    if modelo not in MODELOS:
        raise ValueError(f"Modelo desconhecido: {modelo}")
    if tarefa not in corpus.TAREFAS:
        raise ValueError(f"Tarefa desconhecida: {tarefa}")

    hiper = {**HIPERPARAMETROS_PADRAO, **(hiperparametros or {})}
    rotulos = corpus.rotulos_da_tarefa(tarefa)
    aparelho = dispositivo_disponivel(dispositivo)
    inicio = time.monotonic()

    treino = _conjunto(con, tarefa, "treino", rotulos)
    validacao = _conjunto(con, tarefa, "validacao", rotulos)
    teste = _conjunto(con, tarefa, "teste", rotulos)

    if not treino["textos"]:
        raise RuntimeError(
            "Não há exemplos de treino. Rotule notícias no repositório e rode a"
            " divisão em treino/validação/teste."
        )
    if not teste["textos"]:
        raise RuntimeError(
            "Não há exemplos de teste. Sem conjunto de teste não existe matriz"
            " de confusão para reportar."
        )

    multilabel = tarefa == "indicadores"
    tokenizador = AutoTokenizer.from_pretrained(MODELOS[modelo]["id"])
    rede = AutoModelForSequenceClassification.from_pretrained(
        MODELOS[modelo]["id"],
        num_labels=len(rotulos),
        problem_type="multi_label_classification" if multilabel else None,
    ).to(aparelho)

    class Lote(Dataset):
        def __init__(self, dados):
            self.codificado = tokenizador(
                dados["textos"], truncation=True, padding="max_length",
                max_length=hiper["max_length"], return_tensors="pt",
            )
            self.rotulos = torch.tensor(
                dados["rotulos"], dtype=torch.float if multilabel else torch.long
            )

        def __len__(self):
            return len(self.rotulos)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.codificado.items()}
            item["labels"] = self.rotulos[i]
            return item

    carregador = DataLoader(
        Lote(treino), batch_size=hiper["batch_size"], shuffle=True
    )
    otimizador = torch.optim.AdamW(
        rede.parameters(),
        lr=hiper["learning_rate"],
        weight_decay=hiper["weight_decay"],
    )
    total_passos = len(carregador) * hiper["epocas"]
    agendador = get_linear_schedule_with_warmup(
        otimizador, int(0.1 * total_passos), total_passos
    )

    def prever(dados):
        if not dados["textos"]:
            return None, None
        rede.eval()
        previstos, verdadeiros = [], []
        with torch.no_grad():
            for lote in DataLoader(Lote(dados), batch_size=16):
                alvos = lote.pop("labels")
                saida = rede(**{k: v.to(aparelho) for k, v in lote.items()})
                if multilabel:
                    previstos.extend(
                        (torch.sigmoid(saida.logits).cpu().numpy() >= 0.5).astype(int)
                    )
                else:
                    previstos.extend(saida.logits.argmax(-1).cpu().numpy())
                alvos_np = alvos.numpy()
                verdadeiros.extend(
                    alvos_np.astype(int) if multilabel else alvos_np
                )
        return np.array(verdadeiros), np.array(previstos)

    historico = []
    for epoca in range(hiper["epocas"]):
        rede.train()
        perda_total = 0.0
        for lote in carregador:
            otimizador.zero_grad()
            saida = rede(**{k: v.to(aparelho) for k, v in lote.items()})
            saida.loss.backward()
            torch.nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
            otimizador.step()
            agendador.step()
            perda_total += saida.loss.item()

        registro = {
            "epoca": epoca + 1,
            "perda_treino": perda_total / max(1, len(carregador)),
        }
        y_val, p_val = prever(validacao)
        if y_val is not None:
            registro["f1_macro_validacao"] = _metricas(
                tarefa, y_val, p_val, rotulos
            )["f1_macro"]
        historico.append(registro)

    y_teste, p_teste = prever(teste)
    metricas = _metricas(tarefa, y_teste, p_teste, rotulos)
    metricas["f1_macro_validacao_final"] = (
        historico[-1].get("f1_macro_validacao") if historico else None
    )

    caminho = None
    if salvar_modelo:
        caminho = PASTA_MODELOS / tarefa
        caminho.mkdir(parents=True, exist_ok=True)
        rede.save_pretrained(caminho)
        tokenizador.save_pretrained(caminho)
        (caminho / "rotulos.json").write_text(
            json.dumps(rotulos, ensure_ascii=False), encoding="utf-8"
        )
        caminho = str(caminho)

    duracao = time.monotonic() - inicio
    resultado = {
        "tarefa": tarefa,
        "modelo": modelo,
        "hiperparametros": hiper,
        "metricas": metricas,
        "rotulos": rotulos,
        "historico": historico,
        "n_treino": len(treino["textos"]),
        "n_validacao": len(validacao["textos"]),
        "n_teste": len(teste["textos"]),
        "duracao_s": duracao,
        "dispositivo": aparelho,
        "caminho_modelo": caminho,
    }

    if registrar:
        resultado["id"] = registrar_execucao(con, resultado)

    del rede
    if aparelho == "cuda":
        torch.cuda.empty_cache()
    return resultado


def registrar_execucao(con, resultado: dict) -> int:
    metricas = resultado["metricas"]
    cursor = con.execute(
        "INSERT INTO treino_execucao (tarefa, modelo, hiperparametros, metricas,"
        " matriz_confusao, rotulos, historico, n_treino, n_validacao, n_teste,"
        " duracao_s, dispositivo, caminho_modelo, erro, criado_em)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            resultado["tarefa"],
            resultado["modelo"],
            json.dumps(resultado["hiperparametros"], ensure_ascii=False),
            json.dumps(metricas, ensure_ascii=False),
            json.dumps(
                metricas.get("matriz_confusao") or metricas.get("matrizes_por_classe"),
                ensure_ascii=False,
            ),
            json.dumps(resultado["rotulos"], ensure_ascii=False),
            json.dumps(resultado["historico"], ensure_ascii=False),
            resultado["n_treino"],
            resultado["n_validacao"],
            resultado["n_teste"],
            resultado["duracao_s"],
            resultado["dispositivo"],
            resultado["caminho_modelo"],
            resultado.get("erro"),
            agora(),
        ),
    )
    return cursor.lastrowid


def _chave_de_selecao(resultado: dict) -> float:
    """Escolhe pela validação, não pelo teste. Quando não há validação (classe
    rara demais para sobrar exemplo), cai para o teste e isso fica registrado,
    porque o número deixa de ser uma estimativa limpa."""
    metricas = resultado["metricas"]
    validacao = metricas.get("f1_macro_validacao_final")
    return validacao if validacao is not None else metricas["f1_macro"]


def rodar_comparacao(
    con,
    tarefa: str,
    modelos: list[str] | None = None,
    grade: list[dict] | None = None,
    dispositivo: str = "auto",
    salvar_melhor: bool = True,
    progresso=None,
) -> dict:
    """Treina cada modelo com cada configuração da grade e devolve a tabela.

    Falhas (falta de memória na GPU, modelo indisponível) são registradas e não
    interrompem a comparação: um modelo que não cabe na placa é um resultado
    relevante para o TCC, não um motivo para perder as outras 23 execuções.
    """
    modelos = modelos or list(MODELOS)
    grade = grade or GRADE
    execucoes, falhas = [], []
    total = len(modelos) * len(grade)
    feitos = 0

    for modelo in modelos:
        for hiper in grade:
            feitos += 1
            if progresso:
                progresso(feitos, total, modelo, hiper)
            try:
                execucoes.append(
                    treinar(con, tarefa, modelo, hiper, dispositivo, salvar_modelo=False)
                )
            except Exception as erro:
                registrar_falha(con, tarefa, modelo, hiper, f"{type(erro).__name__}: {erro}")
                falhas.append({"modelo": modelo, "hiperparametros": hiper, "erro": str(erro)})

    melhor = None
    if execucoes:
        melhor = max(execucoes, key=_chave_de_selecao)
        if salvar_melhor:
            # Retreina a configuração vencedora só para gravar os pesos; a
            # métrica reportada continua sendo a da execução original.
            treinar(
                con, tarefa, melhor["modelo"], melhor["hiperparametros"],
                dispositivo, salvar_modelo=True, registrar=False,
            )

    return {
        "tarefa": tarefa,
        "execucoes": execucoes,
        "falhas": falhas,
        "melhor": melhor,
        "ambiente": descrever_ambiente(),
    }


def registrar_falha(con, tarefa: str, modelo: str, hiper: dict, erro: str) -> None:
    """Uma falha também é resultado: sem registrá-la, a tabela do TCC daria a
    entender que o modelo não foi testado."""
    con.execute(
        "INSERT INTO treino_execucao (tarefa, modelo, hiperparametros, erro,"
        " criado_em) VALUES (?,?,?,?,?)",
        (tarefa, modelo, json.dumps(hiper, ensure_ascii=False), erro, agora()),
    )
