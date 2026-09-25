# ClassificadorIFG

Registra notícias de violência armada em um formulário e as classifica em
indicadores, alimentando um banco de dados estruturado. Substitui a triagem
manual, que limita o volume de notícias analisadas e a velocidade de
atualização dos indicadores.

## Como rodar

```bash
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
cp .env.example .env        # e preencha a GROQ_API_KEY
python app.py
```

A aplicação sobe em http://localhost:5000 e cria o `classificador.db` no
primeiro uso.

## Fluxo

1. **Abertura da ocorrência** — data e endereço (cidade, bairro, localidade).
2. **Inclusão de notícia** — link e texto colados; o analista clica em OK.
3. **Pré-processamento** — `preprocessar_noticia.resumir` descarta as frases sem
   relação com a ocorrência antes de gastar tokens com elas.
4. **Classificação com BERTimbau** — filtro rápido dos indicadores de texto.
5. **Validação com LLM** — a cada OK a LLM relê **todas** as notícias juntas,
   confirma os indicadores pela skill de cada um e refaz a lista de vítimas do
   zero.
6. **Verificação das vítimas** — o analista confere e corrige; as correções são
   mantidas nos processamentos seguintes.
7. **Regras dos indicadores** — os que dependem das vítimas são calculados sobre
   a lista final.
8. **Registro não aprovado** — gravado com a classificação e as notícias
   vinculadas.
9. **Aprovação** — o responsável libera para a comunicação ou devolve ao analista.
10. **Atualização** — notícia nova ou correção manual volta ao fluxo, fica no
    histórico e passa de novo pela aprovação.

Em paralelo, um alerta aponta possíveis duplicidades (mesmo dia e bairro da
mesma cidade).

## Contagem de vítimas

A contagem é de **pessoas distintas**, nunca a soma das notícias. O nome
identifica a pessoa; sem nome, o sistema compara idade, gênero e tipo de vítima,
e um valor "não informado" nunca contradiz um informado. "Um homem baleado" em
um jornal e "dois homens baleados" em outro resultam em 2 vítimas, não 3.

Descrições no lugar do nome ("Adolescente de 16 anos") não contam como nome: a
comparação cai para o perfil. Sem isso, a mesma pessoa vira duas assim que outra
notícia a descreve de outro jeito — e, contadas as duas como mortas, apareceria
uma chacina que não houve.

Só entra na lista quem foi **atingido** por disparo: a situação é `ferida` ou
`morta`, e não existe vítima ilesa.

Quando a LLM discorda de uma vítima já corrigida pelo analista — um ferido que
morreu, por exemplo —, a divergência vira uma **proposta** para ele aceitar, em
vez de sobrescrever a correção.

A **circunstância** é uma lista fechada: Feminicídio/tentativa, Suicídio,
Acidente, Trajeto escolar, Bala perdida, Chacina, LGBTQIAPN+, Tribunal do crime,
Vítima de agente de segurança e Não se aplica.

## Telas de consulta

**Ocorrências** — busca por período (data do fato), filtro por status, paginação
de 20 em 20 e um botão que baixa **toda a base** como `.xlsx`, uma aba por tabela.

**Vítimas** — todas as pessoas atingidas, com a circunstância e o local da
ocorrência. Busca por período, cidade, bairro, localidade (parcial),
circunstância e situação, e um botão que baixa a **planilha de vítimas**
respeitando a busca ativa na tela.

**Acompanhamento** — feridos em checagem periódica. Sem filtro, mostra os
últimos 90 dias; com um período informado, olha fora dessa janela.

A planilha de vítimas traz cidade, bairro e localidade junto de cada pessoa, para
que dê para buscar pelo local sem cruzar abas na mão. Ela é a mesma aba `vitima`
do download da base completa.

## Banco (SQLite)

`ocorrencia` e `vitima` são as duas tabelas centrais, ligadas por
`vitima.ocorrencia_id → ocorrencia.id`. As demais apoiam o fluxo:

| tabela | papel |
| --- | --- |
| `ocorrencia` | data, endereço, analista, status, motivação e indicadores |
| `vitima` | pessoas atingidas, ligadas à ocorrência pelo id |
| `noticia` | textos colados e sua versão pré-processada |
| `vitima_historico` | valor anterior, novo, quem alterou, quando e com qual fonte |
| `alerta_duplicidade` | checagem paralela de registros repetidos, com desfazer |

## Indicadores

Os de **regra** são recalculados sobre a lista final de vítimas a cada
alteração — um ferido que morre pode transformar a ocorrência em chacina. Os de
**texto** vêm da classificação das notícias.

| indicador | como é identificado |
| --- | --- |
| Criança ou adolescente baleado | regra: vítima com menos de 18 anos (ECA) |
| Mulheres baleadas | regra: vítima do gênero feminino |
| Vitimização policial | regra: vítima agente de segurança |
| Político baleado | regra: vítima com cargo ou candidatura política |
| Chacina | regra: 3 ou mais mortos na mesma ocorrência |
| Ação policial | texto: BERTimbau e skill na LLM |
| Crime organizado | texto: BERTimbau e skill na LLM |
| Violência de gênero | texto: BERTimbau e skill na LLM |

A tela **Gestão** cadastra indicadores e edita as skills sem mexer no código,
gravando em `indicadores_fonte.json`.

## BERTimbau

O filtro só entra em ação com um checkpoint **ajustado** para os indicadores,
apontado por `BERTIMBAU_MODELO` no `.env`. O BERTimbau puro não tem cabeça de
classificação treinada e devolveria rótulos aleatórios, o que é pior que não
filtrar — por isso, enquanto não houver checkpoint, a classificação fica a cargo
da LLM. Indicadores de texto novos exigem exemplos rotulados (cadastrados na
tela de Gestão) para o retreino.

## Arquivos de configuração

- `categorias_fonte.json` — motivações; rode `python gerar_dicionario.py` após
  editar, para regerar o `dicionario_categorias.txt` usado no prompt.
- `indicadores_fonte.json` — indicadores e skills (editável pela tela de Gestão).
- `locais.json` — listas fixas de cidade e bairro, para evitar grafias
  diferentes que quebrariam o alerta de duplicidade e as estatísticas.
