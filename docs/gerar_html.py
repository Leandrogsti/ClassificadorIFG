"""Transforma um .md com diagramas Mermaid num HTML que abre no navegador.

Existe porque o Markdown com Mermaid só renderiza dentro de um editor que o
entenda (VS Code, GitHub, Obsidian). Para mostrar o fluxograma a alguém que não
tem nenhum deles — uma banca, por exemplo — o caminho é um arquivo que o
navegador abre sozinho.

    python docs/gerar_html.py                      # todos os .md da pasta docs
    python docs/gerar_html.py fluxograma.md        # só um

Precisa de internet na primeira abertura: o Mermaid vem de CDN. Para arquivar
sem depender da rede, abra o HTML e use "Salvar página como > completa", ou
exporte cada diagrama em <https://mermaid.live>.
"""
import html
import re
import sys
from pathlib import Path

PASTA = Path(__file__).parent

MOLDE = """<!doctype html>
<html lang="pt-BR">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{titulo}</title>
<style>
  :root {{
    --fundo: #fbfbfd; --texto: #1c2430; --suave: #5c6878;
    --borda: #dfe4ec; --cartao: #fff;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --fundo: #14181e; --texto: #e6eaf0; --suave: #9aa6b5;
      --borda: #2b323c; --cartao: #1b2027;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 16px 80px; background: var(--fundo);
    color: var(--texto); font: 16px/1.6 -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  main {{ max-width: 920px; margin: 0 auto; }}
  h1 {{ font-size: 1.9rem; line-height: 1.25; margin: 0 0 4px; }}
  h2 {{
    font-size: 1.3rem; margin: 44px 0 12px; padding-top: 18px;
    border-top: 1px solid var(--borda);
  }}
  h3 {{ font-size: 1.05rem; margin: 28px 0 8px; }}
  p, li {{ color: var(--texto); }}
  code {{
    background: var(--cartao); border: 1px solid var(--borda);
    border-radius: 4px; padding: 1px 5px; font-size: 0.88em;
  }}
  .diagrama {{
    background: var(--cartao); border: 1px solid var(--borda);
    border-radius: 10px; padding: 18px; margin: 18px 0; overflow-x: auto;
  }}
  .diagrama svg {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
  .aviso {{
    background: var(--cartao); border: 1px solid var(--borda);
    border-left: 3px solid #b0561a; border-radius: 6px;
    padding: 10px 14px; color: var(--suave); font-size: 0.92rem;
  }}
  hr {{ display: none; }}
  @media print {{
    body {{ padding: 0; }}
    h2 {{ page-break-before: always; border-top: 0; }}
    .diagrama {{ page-break-inside: avoid; border-color: #ccc; }}
    .aviso {{ display: none; }}
  }}
</style>
<main>
<p class="aviso">
  Gerado de <code>{origem}</code>. Para salvar um diagrama como imagem, clique
  nele com o botão direito e escolha salvar, ou use Ctrl&nbsp;+&nbsp;P para PDF.
</p>
{corpo}
</main>
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  const escuro = window.matchMedia("(prefers-color-scheme: dark)").matches;
  mermaid.initialize({{
    startOnLoad: true,
    theme: escuro ? "dark" : "default",
    flowchart: {{ useMaxWidth: true, htmlLabels: true }},
  }});
</script>
</html>
"""


def _inline(texto: str) -> str:
    """O mínimo de Markdown que estes documentos usam."""
    texto = html.escape(texto)
    texto = re.sub(r"`([^`]+)`", r"<code>\1</code>", texto)
    texto = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", texto)
    # Link de arquivo vira texto: o HTML gerado não acompanha os .md ao lado.
    texto = re.sub(r"\[([^\]]+)\]\((?!https?:)[^)]+\)", r"<code>\1</code>", texto)
    texto = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', texto)
    texto = re.sub(r"&lt;(https?://[^&]+)&gt;", r'<a href="\1">\1</a>', texto)
    return texto


def converter(caminho: Path) -> Path:
    linhas = caminho.read_text(encoding="utf-8").splitlines()
    partes, titulo = [], caminho.stem
    i, dentro, buffer = 0, False, []

    while i < len(linhas):
        linha = linhas[i]
        if linha.strip().startswith("```mermaid"):
            dentro, buffer = True, []
        elif dentro and linha.strip() == "```":
            partes.append(
                '<div class="diagrama"><pre class="mermaid">'
                + html.escape("\n".join(buffer))
                + "</pre></div>"
            )
            dentro = False
        elif dentro:
            buffer.append(linha)
        elif linha.startswith("# "):
            titulo = linha[2:].strip()
            partes.append(f"<h1>{_inline(linha[2:].strip())}</h1>")
        elif linha.startswith("### "):
            partes.append(f"<h3>{_inline(linha[4:].strip())}</h3>")
        elif linha.startswith("## "):
            partes.append(f"<h2>{_inline(linha[3:].strip())}</h2>")
        elif linha.strip() == "---":
            pass
        elif linha.strip():
            # Junta as linhas do parágrafo: o .md quebra em 79 colunas, e
            # respeitar essas quebras no HTML deixaria o texto picado.
            paragrafo = [linha.strip()]
            while i + 1 < len(linhas) and linhas[i + 1].strip() and not (
                linhas[i + 1].startswith(("#", "```", "---"))
            ):
                i += 1
                paragrafo.append(linhas[i].strip())
            partes.append(f"<p>{_inline(' '.join(paragrafo))}</p>")
        i += 1

    destino = caminho.with_suffix(".html")
    destino.write_text(
        MOLDE.format(
            titulo=html.escape(titulo),
            origem=html.escape(caminho.name),
            corpo="\n".join(partes),
        ),
        encoding="utf-8",
    )
    return destino


def main() -> int:
    alvos = (
        [PASTA / nome for nome in sys.argv[1:]]
        if len(sys.argv) > 1
        else sorted(PASTA.glob("*.md"))
    )
    for caminho in alvos:
        if not caminho.exists():
            print(f"não existe: {caminho}")
            continue
        destino = converter(caminho)
        diagramas = destino.read_text(encoding="utf-8").count('class="mermaid"')
        print(f"{caminho.name} -> {destino.name}  ({diagramas} diagramas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
