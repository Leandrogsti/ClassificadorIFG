"""Roda a comparação de modelos pela linha de comando.

Exemplos:

    # divide o corpus e compara os 6 modelos na tarefa binária, na GPU se houver
    python rodar_treino.py --dividir --tarefa violencia

    # força CPU, para reproduzir o trabalho em máquina sem placa de vídeo
    python rodar_treino.py --tarefa violencia --dispositivo cpu

    # uma única combinação, para inspecionar rápido
    python rodar_treino.py --tarefa motivacao --modelo bertimbau-base --grade 1

    # as três tarefas em sequência
    python rodar_treino.py --tarefa todas
"""
import argparse
import json

import corpus
import db
import treinamento


def main() -> None:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument(
        "--tarefa", default="violencia",
        choices=[*corpus.TAREFAS, "todas"],
        help="qual classificador treinar",
    )
    analisador.add_argument(
        "--modelo", action="append", choices=list(treinamento.MODELOS),
        help="limita a comparação a estes modelos (pode repetir a opção)",
    )
    analisador.add_argument(
        "--grade", type=int, metavar="N",
        help="usa só a N-ésima configuração da grade (1 a %d)" % len(treinamento.GRADE),
    )
    analisador.add_argument(
        "--dispositivo", default="auto", choices=["auto", "cuda", "cpu"],
        help="auto usa a GPU quando existe; cpu força processador",
    )
    analisador.add_argument(
        "--dividir", action="store_true",
        help="sorteia treino/validação/teste antes de treinar",
    )
    analisador.add_argument(
        "--semente", type=int, default=42, help="semente do sorteio da divisão",
    )
    analisador.add_argument(
        "--nao-salvar", action="store_true",
        help="não grava os pesos do melhor modelo em modelos/<tarefa>",
    )
    argumentos = analisador.parse_args()

    db.iniciar()
    con = db.conectar()

    ambiente = treinamento.descrever_ambiente()
    print(f"torch {ambiente['torch']} | dispositivo: ", end="")
    print(
        f"{ambiente['gpu']} (CUDA)" if ambiente["cuda_disponivel"] else "CPU",
        f"| pedido: {argumentos.dispositivo}",
    )

    if argumentos.dividir:
        with con:
            contagem = corpus.dividir(con, semente=argumentos.semente)
        print(f"Divisão (semente {argumentos.semente}): {contagem}")

    estado = corpus.estatisticas(con)
    print(
        f"Corpus: {estado['total']} notícias | {estado['positivos']} positivas |"
        f" {estado['negativos']} negativas | {estado['nao_rotulados']} sem rótulo"
    )

    tarefas = list(corpus.TAREFAS) if argumentos.tarefa == "todas" else [argumentos.tarefa]
    grade = (
        [treinamento.GRADE[argumentos.grade - 1]] if argumentos.grade else treinamento.GRADE
    )

    for tarefa in tarefas:
        prontidao = estado["prontidao"][tarefa]
        print(f"\n=== {corpus.TAREFAS[tarefa]} ===")
        if not prontidao["pode_treinar"]:
            print(f"  pulada: {prontidao['motivo']}")
            continue
        if prontidao["motivo"] != "ok":
            print(f"  aviso: {prontidao['motivo']}")

        def progresso(feito, total, modelo, hiper):
            print(
                f"  [{feito}/{total}] {modelo} lr={hiper['learning_rate']}"
                f" bs={hiper['batch_size']} ep={hiper['epocas']}", flush=True,
            )

        with con:
            resultado = treinamento.rodar_comparacao(
                con, tarefa,
                modelos=argumentos.modelo,
                grade=grade,
                dispositivo=argumentos.dispositivo,
                salvar_melhor=not argumentos.nao_salvar,
                progresso=progresso,
            )

        for falha in resultado["falhas"]:
            print(f"  FALHOU {falha['modelo']}: {falha['erro'][:120]}")

        if not resultado["melhor"]:
            print("  nenhuma execução concluída")
            continue

        print(f"\n  {'modelo':<26} {'lr':>7} {'bs':>3} {'ep':>3} {'F1 macro':>9} {'acur.':>7}")
        for execucao in sorted(
            resultado["execucoes"], key=lambda e: -e["metricas"]["f1_macro"]
        ):
            hiper, metricas = execucao["hiperparametros"], execucao["metricas"]
            acuracia = metricas.get("acuracia", metricas.get("acuracia_subconjunto", 0))
            print(
                f"  {execucao['modelo']:<26} {hiper['learning_rate']:>7.0e}"
                f" {hiper['batch_size']:>3} {hiper['epocas']:>3}"
                f" {metricas['f1_macro']:>9.3f} {acuracia:>7.3f}"
            )

        melhor = resultado["melhor"]
        print(
            f"\n  Melhor (por F1 macro na validação): {melhor['modelo']}"
            f" {json.dumps(melhor['hiperparametros'])}"
        )
        if not argumentos.nao_salvar:
            print(f"  Pesos salvos em modelos/{tarefa}")


if __name__ == "__main__":
    main()
