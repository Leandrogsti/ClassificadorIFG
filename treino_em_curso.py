"""Lança o treino em segundo plano, a partir da aplicação.

POR QUE EM SUBPROCESSO. O treino leva minutos e carrega um BERT inteiro na
memória; rodá-lo dentro da requisição deixaria o navegador pendurado e
derrubaria o servidor de desenvolvimento junto com ele, se desse errado. Em
processo separado, a aplicação segue respondendo, a saída vai para um arquivo
que a tela mostra, e um treino que estoura a memória morre sozinho sem levar o
resto.

NÃO É FILA. É um treino por vez, e é o suficiente: a máquina tem uma placa de
vídeo só, e dois treinos simultâneos disputariam os mesmos 8 GB de VRAM e
terminariam os dois mais devagar do que um depois do outro.
"""
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).parent
PASTA = RAIZ / "modelos"
REGISTRO = PASTA / "treino_em_curso.log"
PID = PASTA / "treino_em_curso.pid"


def _vivo(pid: int) -> bool:
    """Se o processo ainda existe. Em Windows não há sinal 0 como no POSIX,
    então pergunta-se ao sistema pela lista de tarefas."""
    try:
        if sys.platform == "win32":
            saida = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            return str(pid) in saida
        import os

        os.kill(pid, 0)
        return True
    except Exception:
        return False


def em_andamento() -> int | None:
    """O PID do treino que está rodando, ou None.

    O arquivo de PID sozinho não basta: um treino interrompido à força deixa o
    arquivo para trás, e a tela diria "em andamento" para sempre.
    """
    if not PID.exists():
        return None
    try:
        pid = int(PID.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        PID.unlink(missing_ok=True)
        return None
    if _vivo(pid):
        return pid
    PID.unlink(missing_ok=True)
    return None


def iniciar(opcoes: dict) -> dict:
    """Dispara `rodar_treino.py` com as opções escolhidas na tela."""
    if em_andamento():
        return {"iniciado": False, "motivo": "Já há um treino em andamento."}

    PASTA.mkdir(parents=True, exist_ok=True)
    comando = [sys.executable, str(RAIZ / "rodar_treino.py")]
    if opcoes.get("estrategia"):
        comando += ["--estrategia", opcoes["estrategia"]]
    for modelo in opcoes.get("modelos") or []:
        comando += ["--modelo", modelo]
    if opcoes.get("grade"):
        comando += ["--grade", str(opcoes["grade"])]
    if opcoes.get("dispositivo"):
        comando += ["--dispositivo", opcoes["dispositivo"]]
    if opcoes.get("dividir"):
        comando.append("--dividir")

    registro = REGISTRO.open("w", encoding="utf-8", buffering=1)
    registro.write(f"$ {' '.join(comando)}\n\n")
    processo = subprocess.Popen(
        comando, cwd=str(RAIZ), stdout=registro, stderr=subprocess.STDOUT,
        # Sem console novo em Windows: o treino é serviço da aplicação, não uma
        # janela que o analista tem de manter aberta.
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    PID.write_text(str(processo.pid), encoding="utf-8")
    return {"iniciado": True, "pid": processo.pid, "comando": " ".join(comando)}


def ultimas_linhas(quantas: int = 40) -> list[str]:
    """O fim do registro, que é onde está o que importa enquanto roda."""
    if not REGISTRO.exists():
        return []
    try:
        linhas = REGISTRO.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return linhas[-quantas:]


def situacao() -> dict:
    pid = em_andamento()
    return {
        "rodando": pid is not None,
        "pid": pid,
        "registro": ultimas_linhas(),
        "tem_registro": REGISTRO.exists(),
    }
