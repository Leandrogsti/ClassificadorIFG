"""Mede a extração da LLM contra um conjunto-ouro, campo a campo.

POR QUE ISTO EXISTE. O corpus e as métricas de treino medem só as CLASSES que o
BERT aprende. Tudo o que a LLM extrai — nome, idade, gênero, tipo, local,
circunstância, corporação, patente, situação de serviço, unidade — não tinha
nenhum conjunto de referência: cada erro era encontrado a olho, um registro de
cada vez, e cada mudança de prompt era aprovada por impressão.

Com isto, mexer no prompt vira medida. Rode antes e depois da mudança e compare
os números.

USO
    python avaliar_extracao.py                 # todos os casos
    python avaliar_extracao.py --limite 1      # só o primeiro
    python avaliar_extracao.py --caso vaz-lobo # por nome, busca parcial
    python avaliar_extracao.py --intervalo 0   # sem pausa (vai estourar o teto)

O RITMO. O plano da Groq tem teto de tokens por MINUTO, e uma ocorrência de três
notícias gasta perto de 6.300. Duas chamadas seguidas são recusadas com 413
ainda que cada uma caiba sozinha. Por isso o padrão é esperar entre os casos;
não é lentidão à toa.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import classificador
import indicadores
from classificador import ErroClassificacao

FONTE = Path(__file__).parent / "ouro_extracao.json"
INTERVALO_PADRAO = 65

# Campos da vítima que valem como identidade na hora de casar o esperado com o
# obtido. Nome quando há; sem nome, o par gênero+status, que é o que distingue
# as pessoas de uma mesma ocorrência na prática.
def _identidade(vitima: dict) -> tuple:
    nome = (vitima.get("nome") or "").strip().lower()
    if nome:
        return ("nome", nome)
    return ("perfil", vitima.get("genero"), vitima.get("status"))


def _igual(esperado, obtido) -> bool:
    """Comparação por campo, tolerante só ao que não muda o sentido."""
    if isinstance(esperado, list):
        return sorted(esperado) == sorted(obtido or [])
    if isinstance(esperado, str) and isinstance(obtido, str):
        return esperado.strip().lower() == obtido.strip().lower()
    return esperado == obtido


def _comparar_vitima(esperada: dict, obtida: dict | None, placar: dict) -> list:
    """Confere só os campos que o ouro declara. O que não está no ouro não é
    cobrado: um conjunto-ouro parcial é melhor que nenhum, e obrigar a preencher
    tudo faria ninguém escrever o primeiro caso."""
    erros = []
    for campo, valor in esperada.items():
        if campo == "vitimizacao_policial":
            continue
        chave = f"vitima.{campo}"
        placar.setdefault(chave, [0, 0])
        placar[chave][1] += 1
        achado = (obtida or {}).get(campo)
        if _igual(valor, achado):
            placar[chave][0] += 1
        else:
            erros.append(f"      {campo}: esperado {valor!r}, veio {achado!r}")

    for campo, valor in (esperada.get("vitimizacao_policial") or {}).items():
        chave = f"agente.{campo}"
        placar.setdefault(chave, [0, 0])
        placar[chave][1] += 1
        achado = ((obtida or {}).get("vitimizacao_policial") or {}).get(campo)
        if _igual(valor, achado):
            placar[chave][0] += 1
        else:
            erros.append(f"      agente.{campo}: esperado {valor!r}, veio {achado!r}")
    return erros


def avaliar(caso: dict, placar: dict) -> dict:
    print(f"\n{'=' * 72}\n{caso['nome']}  ({len(caso['noticias'])} notícia(s))\n{'=' * 72}")
    if caso.get("observacao"):
        print(f"  {caso['observacao']}\n")

    try:
        resultado = classificador.classificar_ocorrencia(list(caso["noticias"]))
    except ErroClassificacao as erro:
        print(f"  FALHOU: {erro}")
        return {"falhou": True}

    esperado = caso["esperado"]
    erros = []

    if "fenomenos" in esperado:
        placar.setdefault("fenomenos", [0, 0])
        placar["fenomenos"][1] += 1
        # A lista que o classificador devolve é a da LEITURA, antes das regras.
        # Os derivados — vitimização policial, chacina, criança baleada — só
        # entram depois, calculados sobre as vítimas, e é `recalcular_fenomenos`
        # que faz isso em produção. Comparar sem este passo cobraria do ouro uma
        # resposta que o sistema nunca dá nesse ponto.
        obtidos = indicadores.calcular(
            resultado.get("vitimas") or [], resultado.get("fenomenos") or []
        )
        if _igual(esperado["fenomenos"], obtidos):
            placar["fenomenos"][0] += 1
        else:
            erros.append(
                f"    fenomenos: esperado {sorted(esperado['fenomenos'])},"
                f" veio {sorted(obtidos)}"
            )

    esperadas = esperado.get("vitimas") or []
    obtidas = list(resultado.get("vitimas") or [])

    placar.setdefault("n_vitimas", [0, 0])
    placar["n_vitimas"][1] += 1
    if len(esperadas) == len(obtidas):
        placar["n_vitimas"][0] += 1
    else:
        erros.append(
            f"    nº de vítimas: esperado {len(esperadas)}, veio {len(obtidas)}"
        )

    disponiveis = list(obtidas)
    for vitima in esperadas:
        alvo = _identidade(vitima)
        casada = next(
            (v for v in disponiveis if _identidade(v) == alvo),
            None,
        )
        # Sem casamento por identidade, cai na ordem: é melhor comparar com a
        # vítima errada e ver os campos divergirem do que não comparar nada.
        if casada is None and disponiveis:
            casada = disponiveis[0]
        if casada is not None:
            disponiveis.remove(casada)
        rotulo = vitima.get("nome") or f"{vitima.get('genero')}/{vitima.get('status')}"
        falhas = _comparar_vitima(vitima, casada, placar)
        if falhas:
            erros.append(f"    vítima {rotulo}:")
            erros.extend(falhas)

    if erros:
        print("  DIVERGÊNCIAS:")
        for linha in erros:
            print(linha)
        # A ficha inteira que veio, não só o que divergiu: sem ela não há como
        # saber se a LLM entendeu outra coisa ou simplesmente perdeu alguém.
        print("  FICHA OBTIDA:")
        print(f"    fenomenos: {sorted(indicadores.calcular(obtidas, resultado.get('fenomenos') or []))}")
        for vitima in obtidas:
            resumo = {
                campo: vitima.get(campo)
                for campo in ("nome", "status", "genero", "idade", "tipo_vitima",
                              "circunstancias", "local")
            }
            print(f"    {resumo}")
        if resultado.get("divergencias"):
            print(f"    divergencias: {resultado['divergencias']}")
    else:
        print("  tudo conferido")
    return {"falhou": False, "erros": len(erros)}


def main() -> int:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--limite", type=int, help="quantos casos rodar")
    analisador.add_argument("--caso", help="roda só os casos cujo nome contenha isto")
    analisador.add_argument(
        "--intervalo", type=int, default=INTERVALO_PADRAO,
        help=f"segundos de espera entre casos (padrão {INTERVALO_PADRAO})",
    )
    analisador.add_argument(
        "--repeticoes", type=int, default=1,
        help="quantas vezes rodar cada caso, para medir a VARIAÇÃO da LLM"
             " (ela não é determinística, nem com temperature=0)",
    )
    argumentos = analisador.parse_args()

    if not FONTE.exists():
        print(f"{FONTE.name} não existe. Crie-o com os casos de referência.")
        return 1

    casos = json.loads(FONTE.read_text(encoding="utf-8"))["casos"]
    if argumentos.caso:
        casos = [c for c in casos if argumentos.caso.lower() in c["nome"].lower()]
    if argumentos.limite:
        casos = casos[: argumentos.limite]
    if not casos:
        print("Nenhum caso selecionado.")
        return 1

    # Uma rodada por caso NÃO basta para decidir se uma mudança de prompt
    # ajudou: a mesma notícia já devolveu duas vítimas numa execução e uma na
    # seguinte, com o mesmo prompt e temperature=0. Com `--repeticoes`, o
    # placar passa a contar acertos sobre todas as rodadas, e um campo que
    # oscila aparece como fração em vez de passar por acerto ou por erro
    # conforme a sorte do dia.
    execucoes = [
        (caso, rodada)
        for caso in casos
        for rodada in range(argumentos.repeticoes)
    ]

    placar: dict[str, list[int]] = {}
    falhas = 0
    for indice, (caso, rodada) in enumerate(execucoes):
        if indice and argumentos.intervalo:
            print(f"\n(esperando {argumentos.intervalo}s pelo teto por minuto)")
            time.sleep(argumentos.intervalo)
        if argumentos.repeticoes > 1:
            print(f"\n--- rodada {rodada + 1} de {argumentos.repeticoes} ---")
        if avaliar(caso, placar).get("falhou"):
            falhas += 1

    print(f"\n{'=' * 72}\nPLACAR POR CAMPO\n{'=' * 72}")
    largura = max((len(c) for c in placar), default=10)
    acertos = total = 0
    for campo in sorted(placar):
        certo, cobrado = placar[campo]
        acertos += certo
        total += cobrado
        marca = "ok" if certo == cobrado else "<<"
        print(f"  {campo:<{largura}}  {certo}/{cobrado}  {marca}")
    if total:
        print(f"\n  GERAL: {acertos}/{total} = {100 * acertos / total:.0f}%")
    if falhas:
        print(f"  {falhas} caso(s) nem chegaram a rodar (erro de API)")
    return 0 if acertos == total and not falhas else 2


if __name__ == "__main__":
    sys.exit(main())
