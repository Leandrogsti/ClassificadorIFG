"""Predição com o modelo treinado: o BERT identificando os indicadores.

É a primeira etapa da classificação em produção e também o que o playground
mostra. Fica ANTES da camada de LLM, e é isso que permite julgar se a LLM está
corrigindo o classificador ou apenas repetindo-o.

O resultado é MULTIRRÓTULO e é sempre a lista inteira: cada indicador passa pela
sua própria sigmoide, é comparado com o SEU limiar e entra ou não entra sozinho.
Nenhuma classe exclui outra, não há `argmax` e a varredura não para no primeiro
indicador que passa — uma notícia pode sair com nenhum, com um ou com os sete.
Nenhum indicador acima do limiar é uma resposta legítima: quer dizer que a
notícia está fora do escopo.

Os limiares vêm do treino, um por indicador, calibrados na validação (ver
treinamento.py). Um indicador raro cujo limiar ficou em 0,30 seria invisível sob
o corte fixo de 0,5 — é essa calibração que o faz aparecer ao lado dos outros em
vez de ser engolido por eles.

Três estratégias de treino respondem por aqui, e o manifesto `modelo.json` diz
qual delas está em produção:

- `unico` — uma rede com uma saída por indicador, uma passagem por notícia.
- `por_indicador` — uma rede binária por indicador. Todas são consultadas
  para a mesma notícia e as respostas são reunidas num vetor só, o que custa
  uma passagem e uma rede na memória por indicador — perto de 3,5 GB com o
  BERTimbau base e os oito atuais. Mesmo formato de saída da outra estratégia.

Os modelos ficam em memória depois do primeiro uso, porque carregar um BERT a
cada requisição levaria segundos.
"""
import json
import os
from pathlib import Path

import corpus
import indicadores
import treinamento

TAREFA = "indicadores"

# Só usado quando o treino não deixou limiar calibrado para a classe (ou quando
# alguém pede um corte único, como faz o controle do playground). O corte
# calibrado do manifesto tem precedência sobre ele.
LIMIAR_NEUTRO = float(os.getenv("BERTIMBAU_LIMIAR", "0.5"))

_carregados: dict[str, dict] = {}


def caminho_do_modelo(tarefa: str = TAREFA) -> Path:
    return treinamento.PASTA_MODELOS / tarefa


def _descricao(tarefa: str = TAREFA) -> dict | None:
    """O que existe em disco para esta tarefa, normalizado.

    Primeiro o manifesto gravado pelo treino. Se não houver, aceita o formato
    antigo (pesos soltos na raiz da pasta, com `rotulos.json` ao lado) para que
    um checkpoint treinado antes desta mudança continue respondendo — ali não
    havia limiar por classe, então todos ficam no corte neutro.
    """
    manifesto = treinamento.ler_manifesto(tarefa)
    if manifesto and manifesto.get("rotulos"):
        return manifesto

    base = caminho_do_modelo(tarefa)
    if not (base / "config.json").exists():
        return None

    antigos = base / "rotulos.json"
    rotulos = (
        json.loads(antigos.read_text(encoding="utf-8"))
        if antigos.exists()
        else corpus.rotulos_da_tarefa(tarefa)
    )
    return {
        "tarefa": tarefa,
        "estrategia": "unico",
        "rotulos": rotulos,
        "limiares": {},
        "max_length": 256,
        "pastas": {"multirrotulo": "."},
        "formato": "anterior ao manifesto",
    }


def treinado(tarefa: str = TAREFA) -> bool:
    return _descricao(tarefa) is not None


def estrategia(tarefa: str = TAREFA) -> str | None:
    descricao = _descricao(tarefa)
    return descricao["estrategia"] if descricao else None


def limiares(tarefa: str = TAREFA) -> dict[str, float]:
    """O corte de cada indicador, para que a tela possa mostrar a pontuação ao
    lado do limiar que ela precisava vencer."""
    descricao = _descricao(tarefa)
    if not descricao:
        return {}
    gravados = descricao.get("limiares") or {}
    return {
        nome: float(gravados.get(nome, LIMIAR_NEUTRO))
        for nome in descricao["rotulos"]
    }


def defasagem(tarefa: str = TAREFA) -> dict:
    """O que o modelo em produção NÃO conhece do catálogo de hoje, e vice-versa.

    Existe porque o desencontro era silencioso e caro. A inferência lê os
    rótulos do MANIFESTO do modelo, não do catálogo: um modelo treinado com
    oito classes continua prevendo aquelas oito depois de alguém criar a nona,
    e nunca a marca. Nenhuma tela dizia isso. Pior, notícia que só sustentava
    a classe nova caía no `arquivo` como se não tivesse nada — o indicador
    existia no catálogo, na tela e no prompt, e não existia para o
    classificador.

    - `ausentes`: estão no catálogo e o modelo não conhece. É o caso grave: o
      indicador não é marcado por ninguém até alguém retreinar.
    - `obsoletos`: o modelo conhece e o catálogo não tem mais. Inofensivo na
      prática (a previsão é descartada na validação), mas sinaliza que o
      checkpoint é de outra época.

    Sem modelo treinado devolve tudo vazio e `defasado` falso: não há
    desencontro quando não há modelo — há ausência, que a tela já mostra.
    """
    descricao = _descricao(tarefa)
    if descricao is None:
        return {"defasado": False, "ausentes": [], "obsoletos": [], "modelo": []}

    do_modelo = list(descricao["rotulos"])
    do_catalogo = indicadores.nomes()
    ausentes = [n for n in do_catalogo if n not in set(do_modelo)]
    obsoletos = [n for n in do_modelo if n not in set(do_catalogo)]
    return {
        "defasado": bool(ausentes or obsoletos),
        "ausentes": ausentes,
        "obsoletos": obsoletos,
        "modelo": do_modelo,
    }


def resumo(tarefa: str = TAREFA) -> dict | None:
    """O que está em produção, ou None quando não há modelo treinado.

    Serve para a tela de modelos apontar QUAL linha da tabela é a que responde
    hoje. Sem isso, a tabela mostra a melhor execução e deixa implícito que é
    ela a que está no ar — o que deixa de ser verdade assim que alguém treina
    outra coisa depois.
    """
    descricao = _descricao(tarefa)
    if descricao is None:
        return None
    return {
        "estrategia": descricao["estrategia"],
        "modelo": descricao.get("modelo"),
        "hiperparametros": descricao.get("hiperparametros") or {},
        "limiares": limiares(tarefa),
        "treinado_em": descricao.get("treinado_em"),
        "redes": len(descricao.get("pastas") or {}),
    }


def esquecer(tarefa: str | None = None) -> None:
    """Descarta o modelo em memória após um retreino, senão a aplicação
    continuaria respondendo com os pesos antigos."""
    if tarefa is None:
        _carregados.clear()
    else:
        _carregados.pop(tarefa, None)


def _carregar(tarefa: str) -> dict:
    if tarefa in _carregados:
        return _carregados[tarefa]

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    descricao = _descricao(tarefa)
    if descricao is None:
        raise RuntimeError(
            f"A tarefa '{tarefa}' ainda não tem modelo treinado. Rotule notícias"
            " no corpus e rode o treino."
        )

    base = caminho_do_modelo(tarefa)
    aparelho = treinamento.dispositivo_disponivel()
    # Os rótulos vêm do manifesto, não do catálogo atual: um indicador criado
    # em /gestao depois do treino não existe nos pesos, e ler a lista nova
    # deslocaria todas as posições do vetor.
    rotulos = descricao["rotulos"]

    if descricao["estrategia"] == "cluster":
        # NADA é carregado aqui. Três modelos por indicador dão 21 redes e ~15 GB,
        # e a placa tem 8. O que se guarda é o mapa; quem carrega é `prever`, um
        # grupo de três por vez — cerca de 2,2 GB, que caba com folga.
        grupos: dict[str, list[dict]] = {nome: [] for nome in rotulos}
        for nome_unidade, relativo in (descricao.get("pastas") or {}).items():
            indicador = nome_unidade.split("__")[0]
            if indicador not in grupos:
                continue
            caminho = (base / relativo).resolve()
            if not (caminho / "config.json").exists():
                raise RuntimeError(
                    f"O manifesto de '{tarefa}' aponta para {caminho}, que não"
                    " tem pesos. Rode o treino de novo."
                )
            grupos[indicador].append({
                "unidade": nome_unidade,
                "caminho": caminho,
                "limiar": float(
                    (descricao.get("limiares") or {}).get(
                        nome_unidade, LIMIAR_NEUTRO
                    )
                ),
            })
        faltando = [nome for nome, redes in grupos.items() if not redes]
        if faltando:
            raise RuntimeError(
                "O manifesto do cluster não tem rede para: "
                + ", ".join(faltando) + ". Rode o treino de novo."
            )
        _carregados[tarefa] = {
            "redes": [],
            "grupos": grupos,
            "rotulos": rotulos,
            "limiares": {},
            "estrategia": "cluster",
            "modelo": descricao.get("modelo"),
            "max_length": int(descricao.get("max_length") or 256),
            "dispositivo": aparelho,
            "torch": torch,
            "n_modelos": int(
                descricao.get("n_modelos") or len(treinamento.MODELOS_CLUSTER)
            ),
            "votos_para_classificar": int(
                descricao.get("votos_para_classificar")
                or treinamento.votos_para_prosseguir(
                    int(descricao.get("n_modelos")
                        or len(treinamento.MODELOS_CLUSTER))
                )
            ),
        }
        return _carregados[tarefa]

    redes = []
    for nome_unidade, relativo in (descricao.get("pastas") or {}).items():
        caminho = (base / relativo).resolve()
        if not (caminho / "config.json").exists():
            raise RuntimeError(
                f"O manifesto de '{tarefa}' aponta para {caminho}, que não tem"
                " pesos. Rode o treino de novo para regravar os checkpoints."
            )
        rede = AutoModelForSequenceClassification.from_pretrained(caminho).to(aparelho)
        rede.eval()
        # Em `por_indicador` cada rede responde por UM indicador, e é o nome da
        # pasta que diz qual: a coluna 0 daquela rede é a posição desse
        # indicador no vetor final.
        colunas = (
            rotulos
            if descricao["estrategia"] == "unico"
            else [nome_unidade]
        )
        redes.append({
            "rede": rede,
            "tokenizador": AutoTokenizer.from_pretrained(caminho),
            "colunas": colunas,
        })

    if not redes:
        raise RuntimeError(
            f"O manifesto de '{tarefa}' não lista nenhum checkpoint. Rode o"
            " treino de novo."
        )

    _carregados[tarefa] = {
        "redes": redes,
        "rotulos": rotulos,
        "limiares": {
            nome: float((descricao.get("limiares") or {}).get(nome, LIMIAR_NEUTRO))
            for nome in rotulos
        },
        "estrategia": descricao["estrategia"],
        "modelo": descricao.get("modelo"),
        "max_length": int(descricao.get("max_length") or 256),
        "dispositivo": aparelho,
        "torch": torch,
    }
    return _carregados[tarefa]


def _prever_cluster(texto: str, modelo: dict, limiar: float | None) -> dict:
    """Um grupo de três por vez: carrega, pontua, libera, passa ao próximo.

    É mais lento que manter tudo na memória — é um ciclo de carga por
    indicador, oito hoje — e é o preço de cabermos em 8 GB de VRAM com as
    24 redes do trio. O ciclo é
    por INDICADOR porque o voto também é: as três redes de um indicador têm de
    estar vivas ao mesmo tempo para votar, as dos outros não.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch = modelo["torch"]
    precisa = modelo["votos_para_classificar"]

    votos: dict[str, int] = {}
    confianca: dict[str, float] = {}
    por_rede: dict[str, dict] = {}

    for indicador, redes in modelo["grupos"].items():
        pontos = {}
        for ficha in redes:
            rede = AutoModelForSequenceClassification.from_pretrained(
                ficha["caminho"]
            ).to(modelo["dispositivo"])
            rede.eval()
            tokenizador = AutoTokenizer.from_pretrained(ficha["caminho"])
            entrada = tokenizador(
                texto, truncation=True, padding="max_length",
                max_length=modelo["max_length"], return_tensors="pt",
            )
            with torch.no_grad():
                logits = rede(
                    **{k: v.to(modelo["dispositivo"]) for k, v in entrada.items()}
                ).logits[0]
            pontos[ficha["unidade"]] = {
                "confianca": float(torch.sigmoid(logits).cpu().numpy()[0]),
                "limiar": float(limiar) if limiar is not None else ficha["limiar"],
            }
            # Liberar aqui, e não no fim do laço de indicadores, é o que mantém
            # o pico de memória no tamanho de um grupo.
            del rede, tokenizador
        if modelo["dispositivo"] == "cuda":
            torch.cuda.empty_cache()

        votos[indicador] = sum(
            1 for p in pontos.values() if p["confianca"] >= p["limiar"]
        )
        # A confiança do indicador é a MÉDIA das três: serve para ordenar a fila
        # de revisão, nunca para decidir — quem decide é a contagem de votos.
        confianca[indicador] = (
            sum(p["confianca"] for p in pontos.values()) / len(pontos)
        )
        por_rede[indicador] = pontos

    rotulos = modelo["rotulos"]
    return {
        "previsto": [n for n in rotulos if votos.get(n, 0) >= precisa],
        # Abaixo do corte mas acima de zero: nem ausente nem decidido. Vira
        # ALERTA (laranja) e passa por pessoa antes de valer.
        "duvidosos": [
            n for n in rotulos if 0 < votos.get(n, 0) < precisa
        ],
        "n_modelos": modelo["n_modelos"],
        # `concordancia` é o nome que a coluna de `entrada_classificacao` já
        # usa, e é por ele que `auditoria.registrar_entrada` grava quantas das
        # três redes marcaram o indicador.
        "concordancia": votos,
        "confianca": confianca,
        "por_rede": por_rede,
        "votos_para_classificar": precisa,
    }


def prever(texto: str, limiar: float | None = None, tarefa: str = TAREFA) -> dict:
    """Devolve TODOS os indicadores marcados, com a confiança de CADA um.

    `limiar` como None usa o corte calibrado de cada indicador, que é o
    comportamento de produção. Um número substitui todos os cortes por ele —
    serve para o playground mostrar o que mudaria com outro corte, e nada mais.

    A confiança é parte do resultado, não um detalhe: um indicador marcado com
    0,51 e outro com 0,99 dizem coisas muito diferentes sobre quando confiar no
    filtro e quando deixar a decisão para a LLM. Vêm todas as classes, inclusive
    as que não passaram — ao revisar um erro, saber que a classe certa ficou em
    0,48 aponta o limiar, e saber que ficou em 0,02 aponta falta de exemplo.
    """
    modelo = _carregar(tarefa)
    if modelo["estrategia"] == "cluster":
        saida = _prever_cluster(texto, modelo, limiar)
        return {
            "tarefa": tarefa,
            "estrategia": "cluster",
            "dispositivo": modelo["dispositivo"],
            "limiar": float(limiar) if limiar is not None else None,
            **saida,
        }

    torch = modelo["torch"]
    rotulos = modelo["rotulos"]

    pontuacoes: dict[str, float] = {}
    for unidade in modelo["redes"]:
        entrada = unidade["tokenizador"](
            texto, truncation=True, padding="max_length",
            max_length=modelo["max_length"], return_tensors="pt",
        )
        with torch.no_grad():
            logits = unidade["rede"](
                **{k: v.to(modelo["dispositivo"]) for k, v in entrada.items()}
            ).logits[0]
        # Sigmoide, nunca softmax: as classes não competem entre si. Com softmax
        # as pontuações somariam 1 e marcar dois indicadores exigiria que um
        # cedesse espaço ao outro.
        for i, valor in enumerate(torch.sigmoid(logits).cpu().numpy()):
            if i < len(unidade["colunas"]):
                pontuacoes[unidade["colunas"][i]] = float(valor)

    cortes = (
        {nome: float(limiar) for nome in rotulos}
        if limiar is not None
        else dict(modelo["limiares"])
    )

    # A lista completa, em UMA passada por todos os indicadores: cada um é
    # comparado com o seu próprio corte e nada interrompe a varredura.
    previsto = [
        nome for nome in rotulos
        if pontuacoes.get(nome, 0.0) >= cortes.get(nome, LIMIAR_NEUTRO)
    ]

    return {
        "tarefa": tarefa,
        "previsto": previsto,
        "confianca": {nome: pontuacoes.get(nome, 0.0) for nome in rotulos},
        "limiares": cortes,
        # Um número só quando o corte é o mesmo para todos; None quando cada
        # indicador tem o seu, porque aí nenhum número único descreveria a
        # decisão.
        "limiar": float(limiar) if limiar is not None else (
            next(iter(set(cortes.values()))) if len(set(cortes.values())) == 1 else None
        ),
        "estrategia": modelo["estrategia"],
        "dispositivo": modelo["dispositivo"],
        # Com uma rede só, a concordância é 0 ou 1 em 1 — o que, pela fração de
        # 2/3, já basta para seguir. A fila precisa desses dois campos em
        # QUALQUER estratégia: sem eles ela teria um caminho para o cluster e
        # outro para as demais, e dois caminhos divergem com o tempo.
        "concordancia": {
            nome: (1 if nome in previsto else 0) for nome in rotulos
        },
        "n_modelos": 1,
        "duvidosos": [],
    }


# --------------------------------------------------------------------------
# Marcação em produção
# --------------------------------------------------------------------------

_externo = None
_externo_carregado = False


def _checkpoint_externo():
    """BERTIMBAU_MODELO no .env aponta um checkpoint de fora — outra pasta ou
    um repositório do Hugging Face. Só use com um modelo AJUSTADO para estas
    classes: um BERTimbau puro não tem cabeça de classificação treinada e
    devolveria rótulos aleatórios, o que é pior que não filtrar."""
    global _externo, _externo_carregado
    if _externo_carregado:
        return _externo
    _externo_carregado = True
    if not os.getenv("BERTIMBAU_MODELO"):
        return None
    from transformers import pipeline

    _externo = pipeline(
        "text-classification", model=os.getenv("BERTIMBAU_MODELO"),
        top_k=None, truncation=True, max_length=512,
    )
    return _externo


def origem() -> str | None:
    """Qual modelo responde, para que a tela não deixe isso implícito."""
    descricao = _descricao()
    if descricao:
        quantas = len(descricao.get("pastas") or {})
        return (
            f"modelo treinado na aplicação ({caminho_do_modelo()}) ·"
            f" estratégia {descricao['estrategia']} · {quantas} rede(s)"
        )
    externo = os.getenv("BERTIMBAU_MODELO")
    return f"checkpoint externo ({externo})" if externo else None


def marcar(texto: str, limiar: float | None = None) -> dict | None:
    """A marcação de produção, ou None quando não há modelo nenhum.

    Devolve o resultado inteiro, não só a lista: as confianças são gravadas no
    banco de classificações para que, ao revisar um erro, se veja se o modelo
    errou com 0,51 ou com 0,98 — a primeira é um limiar mal escolhido, a
    segunda é falta de exemplo no corpus.

    A LLM continua sendo quem certifica: esta lista decide QUAIS prompts entram
    na chamada, nunca o resultado final. E são todos os indicadores marcados,
    não o primeiro: é a lista completa que monta o prompt.
    """
    if treinado():
        resultado = prever(texto, limiar=limiar)
        resultado["modelo"] = str(caminho_do_modelo())
        return resultado

    modelo = _checkpoint_externo()
    if modelo is None:
        return None
    corte = LIMIAR_NEUTRO if limiar is None else float(limiar)
    pontuacoes = {r["label"]: float(r["score"]) for r in modelo(texto)[0]}
    return {
        "tarefa": TAREFA,
        "previsto": [n for n, p in pontuacoes.items() if p >= corte],
        "confianca": pontuacoes,
        "limiares": {n: corte for n in pontuacoes},
        "limiar": corte,
        "estrategia": "externo",
        "modelo": os.getenv("BERTIMBAU_MODELO"),
        "dispositivo": "externo",
    }
