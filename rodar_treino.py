"""Roda a comparação de modelos pela linha de comando.

Treina o classificador MULTIRRÓTULO de indicadores: dado o texto da notícia,
quais dos oito indicadores se aplicam — podem ser vários ao mesmo tempo, ou
nenhum. É o modelo que a aplicação usa em produção para marcar a notícia antes
de chamar a LLM.

Há três estratégias para chegar a essa saída, e elas podem ser comparadas na
mesma partição de teste:

    unico          uma rede com uma saída sigmoide por indicador (padrão)
    por_indicador  uma rede binária por indicador (um-contra-resto)

Exemplos:

    # divide o corpus e compara os 6 modelos na estratégia padrão
    python rodar_treino.py --dividir

    # compara as DUAS estratégias: é este o experimento que responde se vale
    # um modelo por indicador (atenção: multiplica o número de treinos por 8)
    python rodar_treino.py --estrategia ambas --grade 1

    # um modelo por indicador, uma configuração só
    python rodar_treino.py --estrategia por_indicador --modelo bertimbau-base --grade 1

    # força CPU, para reproduzir o trabalho em máquina sem placa de vídeo
    python rodar_treino.py --dispositivo cpu

    # treina só duas classes, para medi-las sem as outras disputando o modelo
    python rodar_treino.py --classe mulheres_baleadas --classe lgbtqiapn

    # desliga as duas correções de desbalanceamento, para medir o efeito delas
    python rodar_treino.py --sem-peso-classes --sem-calibrar-limiar --grade 1
"""
import argparse
import json

import corpus
import db
import indicadores
import treinamento


def main() -> None:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument(
        "--tarefa", default="indicadores", choices=list(corpus.TAREFAS),
        help="qual classificador treinar (hoje só existe um)",
    )
    analisador.add_argument(
        "--estrategia", default="unico",
        choices=[*treinamento.ESTRATEGIAS, "ambas", "todas"],
        help="unico: uma rede multirrótulo; por_indicador: uma rede binária por"
             " indicador; cluster: três modelos por indicador, votando, e só"
             " unanimidade classifica; ambas: unico + por_indicador; todas:"
             " as três",
    )
    analisador.add_argument(
        "--modelo", action="append", choices=list(treinamento.MODELOS),
        help="limita a comparação a estes modelos (pode repetir a opção)",
    )
    analisador.add_argument(
        "--classe", action="append", choices=indicadores.nomes(),
        help="treina só estas classes, em vez das %d (pode repetir a opção)"
             % len(indicadores.nomes()),
    )
    analisador.add_argument(
        "--grade", type=int, metavar="N",
        help="usa só a N-ésima configuração da grade (1 a %d)" % len(treinamento.GRADE),
    )
    analisador.add_argument(
        "--sem-peso-classes", action="store_true",
        help="perda sem pos_weight; serve para MEDIR o efeito do peso das"
             " classes, não para treinar o modelo de produção",
    )
    analisador.add_argument(
        "--sem-calibrar-limiar", action="store_true",
        help="corta todos os indicadores em 0,5 em vez de calibrar cada um na"
             " validação; também é uma medição, não um modo de uso",
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
            contagem = corpus.dividir(
                con, semente=argumentos.semente, refazer=True
            )
        print(f"Divisão REFEITA do zero (semente {argumentos.semente}):"
              f" {contagem}")
        print("  AVISO: o conjunto de teste mudou. As execuções anteriores"
              " deixaram de ser comparáveis com as desta rodada.")
    else:
        # Automático: notícia anotada depois do último sorteio não pode ficar
        # fora do treino sem ninguém notar. Distribui só as pendentes, em lote,
        # e não mexe no que já estava sorteado.
        pendentes = corpus.pendentes_de_particao(con)
        if pendentes:
            with con:
                contagem = corpus.dividir(con, semente=argumentos.semente)
            print(f"{pendentes} notícia(s) sem partição foram distribuídas:"
                  f" {contagem}")

    estado = corpus.estatisticas(con)
    print(
        f"Corpus: {estado['total']} notícias | {estado['rotuladas']} anotadas |"
        f" {estado['treinaveis']} APROVADAS (só estas treinam) |"
        f" {estado['negativas']} sem nenhum indicador |"
        f" {estado['pendentes']} na fila"
    )
    if estado["rotuladas_sem_aprovar"]:
        print(
            f"    ATENÇÃO: {estado['rotuladas_sem_aprovar']} notícia(s)"
            " rotuladas esperando revisão. Elas NÃO entram neste treino —"
            " aprove-as em /corpus."
        )
    for nome, n in sorted(estado["indicadores"].items(), key=lambda x: -x[1]):
        print(f"    {nome:<30} {n:>4}")

    classes = argumentos.classe or indicadores.nomes()
    if argumentos.classe:
        print(f"Treinando só {len(classes)} de {len(indicadores.nomes())} classes:"
              f" {', '.join(classes)}")

    # A MESMA trava da tela, sobre as classes REALMENTE pedidas. Sem isto, a
    # linha de comando seria um desvio silencioso do mínimo por classe: a tela
    # recusaria e `python rodar_treino.py` treinaria do mesmo jeito.
    prontidao_pedida = corpus.estatisticas(con, classes)["prontidao"]
    prontidao = prontidao_pedida["indicadores"]
    if not prontidao["pode_treinar"]:
        print(f"\nTREINO NÃO INICIADO: {prontidao['motivo']}")
        if prontidao.get("prontas"):
            print(
                "\nClasses que já alcançaram o mínimo — dá para treinar só"
                " elas com:\n    python rodar_treino.py "
                + " ".join(f"--classe {n}" for n in prontidao["prontas"])
            )
        if estado["rotuladas_sem_aprovar"]:
            print(
                f"\nHá {estado['rotuladas_sem_aprovar']} notícia(s) rotuladas"
                " esperando revisão em /corpus. Aprovar o que já está certo"
                " pode ser o bastante."
            )
        # SystemExit, e não `return`: `main()` é chamada direto no rodapé do
        # arquivo e um return não viraria código de saída. Treino recusado tem
        # de sair diferente de zero, senão um script que chame este aqui vai
        # achar que treinou.
        raise SystemExit(1)

    # `ambas` continua querendo dizer as duas de rede única por treino, para
    # que um comando antigo não dispare de repente uma comparação três vezes
    # maior. Quem quer o cluster na mesma tabela pede `todas`.
    if argumentos.estrategia == "ambas":
        estrategias = ["unico", "por_indicador"]
    elif argumentos.estrategia == "todas":
        estrategias = list(treinamento.ESTRATEGIAS)
    else:
        estrategias = [argumentos.estrategia]
    grade = [
        {
            **hiper,
            "peso_classes": not argumentos.sem_peso_classes,
            "calibrar_limiar": not argumentos.sem_calibrar_limiar,
        }
        for hiper in (
            [treinamento.GRADE[argumentos.grade - 1]] if argumentos.grade
            else treinamento.GRADE
        )
    ]

    print("\nEstratégias: " + ", ".join(
        f"{e} ({treinamento.ESTRATEGIAS[e]})" for e in estrategias
    ))
    redes = treinamento.contar_treinos(
        argumentos.modelo, grade, estrategias, len(classes)
    )
    print(
        f"Serão ajustadas {redes} rede(s) no total"
        + (
            f" — `por_indicador` ajusta {len(classes)} redes por combinação"
            if "por_indicador" in estrategias else ""
        )
    )
    if argumentos.sem_peso_classes:
        print("AVISO: pos_weight desligado. Num corpus desbalanceado, o modelo"
              " tende a não marcar as classes raras.")
    if argumentos.sem_calibrar_limiar:
        print("AVISO: calibração desligada. Todos os indicadores cortam em 0,5,"
              " o que esconde as classes raras.")

    for tarefa in [argumentos.tarefa]:
        # A prontidão das classes PEDIDAS, não a do catálogo inteiro. `estado`
        # foi apurado sem `classes`: usá-lo aqui faria uma classe rara de fora
        # do pedido pular um treino que `--classe` tinha acabado de liberar.
        prontidao = prontidao_pedida[tarefa]
        print(f"\n=== {corpus.TAREFAS[tarefa]} ===")
        if not prontidao["pode_treinar"]:
            print(f"  pulada: {prontidao['motivo']}")
            continue
        if prontidao["motivo"] != "ok":
            print(f"  aviso: {prontidao['motivo']}")

        def progresso(feito, total, estrategia, modelo, hiper):
            print(
                f"  [{feito}/{total}] {estrategia} · {modelo} lr={hiper['learning_rate']}"
                f" bs={hiper['batch_size']} ep={hiper['epocas']}", flush=True,
            )

        with con:
            resultado = treinamento.rodar_comparacao(
                con, tarefa,
                modelos=argumentos.modelo,
                grade=grade,
                classes=classes,
                estrategias=estrategias,
                dispositivo=argumentos.dispositivo,
                salvar_melhor=not argumentos.nao_salvar,
                progresso=progresso,
                semente=argumentos.semente,
            )

        for falha in resultado["falhas"]:
            print(f"  FALHOU {falha['estrategia']}/{falha['modelo']}:"
                  f" {falha['erro'][:120]}")

        if not resultado["melhor"]:
            print("  nenhuma execução concluída")
            continue

        print(
            f"\n  {'estratégia':<14} {'modelo':<24} {'lr':>7} {'bs':>3} {'ep':>3}"
            f" {'F1 macro':>9} {'em 0,5':>8} {'marcados':>9}"
        )
        for execucao in sorted(
            resultado["execucoes"], key=lambda e: -e["metricas"]["f1_macro"]
        ):
            hiper, metricas = execucao["hiperparametros"], execucao["metricas"]
            sem_calibrar = (metricas.get("sem_calibracao") or {}).get("f1_macro")
            marcacoes = metricas.get("marcacoes_por_noticia") or {}
            print(
                f"  {execucao['estrategia']:<14} {execucao['modelo']:<24}"
                f" {hiper['learning_rate']:>7.0e} {hiper['batch_size']:>3}"
                f" {hiper['epocas']:>3} {metricas['f1_macro']:>9.3f}"
                f" {sem_calibrar if sem_calibrar is not None else 0:>8.3f}"
                f" {marcacoes.get('media_prevista', 0):>9.2f}"
            )
        print(
            "  (`em 0,5` é o MESMO modelo com o corte fixo, para medir o efeito"
            " da calibração;\n   `marcados` é a média de indicadores por notícia"
            f" — o gabarito do teste tem"
            f" {(resultado['melhor']['metricas'].get('marcacoes_por_noticia') or {}).get('media_verdadeira', 0):.2f})"
        )

        placar = treinamento.placar(resultado["execucoes"])
        if len(placar) > 1:
            print(
                f"\n  Placar — melhor configuração de cada modelo, ordenado pelo"
                f" F1 macro na VALIDAÇÃO:"
            )
            print(f"  {'#':>2} {'estratégia':<14} {'modelo':<24}"
                  f" {'F1 teste':>9} {'F1 valid.':>9} {'execs':>6}")
            for posicao, linha in enumerate(placar, start=1):
                metricas = linha["execucao"]["metricas"]
                validacao = metricas.get("f1_macro_validacao_final")
                print(
                    f"  {posicao:>2} {linha['estrategia']:<14} {linha['modelo']:<24}"
                    f" {metricas['f1_macro']:>9.3f}"
                    f" {validacao if validacao is not None else float('nan'):>9.3f}"
                    f" {linha['n_execucoes']:>6}"
                )

        melhor = resultado["melhor"]
        print(
            f"\n  Melhor (por F1 macro na validação): {melhor['estrategia']} /"
            f" {melhor['modelo']} {json.dumps(melhor['hiperparametros'])}"
        )
        print("  Limiar calibrado de cada indicador:")
        for nome, corte in melhor["limiares"].items():
            por_classe = melhor["metricas"]["por_classe"].get(nome, {})
            print(
                f"    {nome:<30} corte {corte:>4.2f} · F1 {por_classe.get('f1', 0):.3f}"
                f" · suporte {por_classe.get('suporte', 0)}"
            )
        if not argumentos.nao_salvar:
            print(
                f"  Pesos salvos em modelos/{tarefa} (manifesto"
                f" {treinamento.MANIFESTO}, estratégia {melhor['estrategia']})"
            )


if __name__ == "__main__":
    main()
