# ClassificadorIFC

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

1. **Recebimento** (`/recebidas`) — a notícia entra com link, data e texto,
   antes de existir ocorrência. É aqui que o classificador olha primeiro.
2. **Limpeza** — tira o ruído de portal (assinatura, data de publicação,
   "Compartilhe no Whatsapp", legenda de foto repetida) e desfaz a ofuscação
   de rede social: `m0rto a tir0s` volta a ser `morto a tiros`. **O BERT e a
   LLM leem este mesmo texto**, e é isso que os faz concordar.
3. **Marcação pelo BERT** — o modelo lê e marca, em multirrótulo, quais das
   **oito** classes se aplicam. A marcação decide quais prompts a LLM vai
   carregar e **autoriza** as circunstâncias condicionadas.
4. **Triagem por concordância** — na estratégia de comitê, conta-se quantos dos
   três modelos marcaram cada indicador:

   | Votos | Estado | O que acontece |
   |---|---|---|
   | 0 em tudo | `arquivo` | vai para o acervo; **não** chama a LLM e **não** é pendência |
   | abaixo do corte | `alerta` | uma pessoa verifica antes de seguir |
   | no corte (2 de 3) ou acima | `com_fenomeno` | segue para virar ocorrência |

   O arquivo é relido quando houver modelo ou indicador novo, pelo botão
   *Reavaliar o arquivo*. É onde fica o tiroteio comum, sem nenhuma das
   características que os indicadores descrevem.
5. **Abertura da ocorrência** — a notícia com fenômeno vira registro, pelo
   analista **júnior**. Só aqui se pede o endereço: estado, cidade e bairro em
   listas encadeadas, nessa ordem, porque 232 nomes de município se repetem
   entre estados. A *localidade* (sub-bairro, favela, conjunto) é texto livre.
6. **Aprofundamento pela LLM** — a LLM relê **todas** as notícias da ocorrência
   juntas, recebendo **só os prompts das classes marcadas**. Ela certifica ou
   descarta cada uma, monta a lista de vítimas, preenche a ficha de cada uma e
   as tabelas complementares. Não acrescenta classe que o BERT não marcou.
7. **Verificação das vítimas** — o analista confere e corrige; as correções são
   mantidas nos processamentos seguintes.
8. **Regras dos indicadores** — os que dependem das vítimas são calculados
   sobre a lista final, e com eles a circunstância `chacina` de cada vítima.
9. **Registro não aprovado** — gravado com a classificação e as notícias.
10. **Revisão pelo sênior** (`/aprovacao`) — ele libera o registro e, com isso,
    **autoriza as notícias dele a treinar o classificador**. É aqui que o ciclo
    fecha, e é a única porta que entra no corpus já aprovada.
11. **Treino** (`/modelos`) — em subprocesso, com o que está **aprovado**.

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
Agente de segurança agressor e Não se aplica. `Não se aplica` nunca convive com
outra: é a ausência de circunstância, não uma a mais.

`Chacina` é a única **calculada**: a LLM não a marca e a tela não a oferece como
caixa. A conta é **quem morreu, menos o agente de segurança que estava em
serviço** — três civis mortos é chacina; dois civis mais um agente morto é
chacina dos três se o agente estava de folga, aposentado ou sem status
informado, e não é se ele estava em serviço, porque aí ele está do outro lado do
mesmo fato. A circunstância e o fenômeno saem da mesma conta, e é isso que
impede um fenômeno que nenhuma vítima sustenta. Ela também **sai** sozinha:
quando um morto se revela ferido, ou o status de serviço é corrigido, desaparece
de todas as vítimas.

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
| `ocorrencia` | ID_CASO, data e hora do caso, faixa horária, endereço, fenômeno observado, status |
| `vitima` | ID_VÍTIMA, nome, status, gênero, cor/raça, grupo etário, idade, tipo (múltiplo), circunstância (múltipla), local |
| `vitimizacao_policial` | corporação, cargo, status do agente, unidade — uma linha por vítima agente |
| `politico_baleado` | partido, cargo, status — uma linha por vítima política |
| `acao_policial` | contexto, corporações presentes, unidades — uma linha por ocorrência |
| `crime_organizado` | contextos, grupos armados citados — uma linha por ocorrência |
| `noticia` | textos colados e sua versão pré-processada |
| `vitima_historico` | valor anterior, novo, quem alterou, quando e com qual fonte |
| `alerta_duplicidade` | checagem paralela de registros repetidos, com desfazer |
| `recebida` | a fila: link, data, texto, estado da triagem e os votos de cada indicador |
| `corpus_noticia` | o corpus de treino: texto, rótulos, partição e a aprovação que autoriza o treino |
| `entrada` | todo texto que entra no sistema, gravado antes de qualquer decisão |
| `entrada_classificacao` | a confiança de cada classe naquela entrada |
| `classificacao_bert` | a mesma marcação em lista JSON, para a auditoria de `/classificacoes` |
| `treino_execucao` | cada treino: hiperparâmetros, métricas, limiares e onde ficaram os pesos |

## Fenômenos e circunstâncias

O BERT classifica **oito classes** a partir do texto, em multirrótulo: uma
notícia pode receber várias, ou nenhuma. Seis vão para FENÔMENO OBSERVADO da
ocorrência e duas são circunstâncias da vítima, mas o modelo é um só e as lê
juntas. Outros dois fenômenos existem sem passar pelo BERT.

Duas perguntas separadas, e confundi-las é o erro fácil: **onde o resultado
vai** (fenômeno da ocorrência ou circunstância da vítima) e **quem decide**.

| classe | vai para | quem decide | conferência pelas vítimas |
| --- | --- | --- | --- |
| Criança ou adolescente baleado | fenômeno | BERT | idade abaixo de 18 (ECA) |
| Mulheres baleadas | fenômeno | BERT | gênero mulher cis, trans ou travesti |
| Ação policial | fenômeno | BERT | — |
| Crime organizado | fenômeno | BERT | — |
| Político baleado | fenômeno | BERT | vítima com tipo `politico` |
| Suicídio | circunstância | BERT | — |
| LGBTQIAPN+ | circunstância | BERT | — |
| Violência de gênero | fenômeno | **LLM** | — |
| Vitimização policial | fenômeno | BERT | vítima com tipo `agente` |
| Chacina | fenômeno **e** circunstância | **regra** | 3+ mortos, sem contar o agente que estava em serviço |

**BERT** são as oito classes do treino e do corpus: o modelo marca, a LLM
certifica. Para as que também têm regra, ela serve de conferência — o fenômeno
entra por texto OU por regra, porque o texto pode ser vago sobre a idade e a
lista de vítimas não, e a lista muda depois, quando um ferido morre.

**LLM** não tem classe própria: violência de gênero é avaliada dentro dos
blocos de `mulheres_baleadas` e `lgbtqiapn`, que são onde a pergunta aparece.

**Regra** é calculado sobre a lista final de vítimas e nem BERT nem LLM o
marcam. Hoje só a chacina. É dado que a LLM já preencheu na vítima, e derivá-lo
é exato — pedir ao modelo que o marque de novo só criaria uma segunda resposta
capaz de contradizer a primeira (uma chacina com dois mortos).

`vitimizacao_policial` já foi só regra e deixou de ser em 09/10. O motivo é o
portão da LLM: a regra roda DEPOIS da LLM e o portão acontece ANTES, então
policial baleado de folga — que não é ação policial — não disparava classe
nenhuma e ia para o arquivo em silêncio. **Indicador que só existe como regra
não pode controlar o portão da LLM.**

### A marcação autoriza

Quatro circunstâncias só são avaliadas pela LLM quando o BERT marcou a classe
correspondente:

| circunstância | só se o BERT marcar |
| --- | --- |
| Feminicídio/tentativa | `mulheres_baleadas` |
| Agente de segurança agressor | `mulheres_baleadas` |
| LGBTQIAPN+ | `lgbtqiapn` |
| Suicídio | `suicidio` |

`agente_seguranca_agressor` depende de `mulheres_baleadas`, e não de
`acao_policial`, de propósito: ela marca a MULHER agredida por um agente. Presa
a `acao_policial`, a circunstância caía no policial que havia sido *baleado* —
quem leva o tiro é vítima, nunca o agressor. As quatro restantes —
`bala_perdida`, `trajeto_escolar`, `acidente` e `tribunal_do_crime` — estão
sempre disponíveis: reconhecê-las na leitura não depende de uma classe própria.

Sem a marcação, a instrução nem viaja no prompt, e uma marcação que a LLM
devolva assim mesmo é descartada na validação. É o que impede a LLM de sair
procurando feminicídio em notícia sem mulher baleada.

### Banco de classificações (`/classificacoes`)

Toda marcação do BERT é gravada antes de a LLM ver qualquer coisa, com o link,
o texto, a confiança de **cada** classe — não só das marcadas — e o **limiar de
cada uma**. Sem esse registro o erro do modelo desaparece assim que a LLM o
corrige, e nunca se saberia que uma classe precisa de mais exemplos. O limiar
por classe é o que torna a revisão legível: uma classe marcada com 0,42 não é
erro de leitura se o corte calibrado dela é 0,30.

A tela mostra o que está por conferir e, nas já conferidas, onde o modelo erra:
`faltou marcar` costuma ser falta de exemplo no corpus, `marcou à toa` costuma
ser limiar baixo demais. Corrigir ali **não altera a ocorrência** — ela já foi
classificada. O que a correção produz é exemplo de treino: um botão a envia ao
corpus já rotulada, e é assim que o erro de hoje vira o acerto do próximo
treino.

### Onde cada prompt mora (`/prompts` e `/gestao`)

Nenhum texto que vai à LLM está escondido no código. A chamada é a soma de cinco
partes, e cada uma tem um lugar para ser editada:

| # | Parte | Onde se edita | Quando viaja |
|---|---|---|---|
| 1 | Seções do dicionário: **papel** e **recorte de análise**, e só essas duas | `categorias_fonte.json` → gera `dicionario_categorias.txt` | sempre |
| 2 | **Prompt mestre** — ocorrência e vítimas | `/prompts` | sempre |
| 3 | **Formato de saída** (o JSON) | `/prompts` | sempre |
| 4 | **Cabeçalho da tarefa de classes** (um dos três) | `/prompts` | sempre |
| 5 | **Skill** de cada indicador | `/gestao` | só quando o BERT marca aquele indicador |

O prompt mestre é o que manda a LLM preencher a ocorrência inteira: a lista de
vítimas e, em cada uma, nome, status, gênero, grupo etário, idade, tipo (civil,
agente ou político), local, circunstâncias, **data de morte** e divergências
entre fontes. **Cor/raça ficou fora de propósito**: a imprensa quase nunca a
informa, e pedi-la à LLM produziria palpite. A coluna existe no banco e se
preenche à mão na tela de Vítimas.

Os três cabeçalhos cobrem as três situações: o BERT marcou classes, o BERT leu e
não marcou nenhuma (o **pré-prompt padrão**), e não há modelo treinado.

`/prompts` grava em `prompts_fonte.json` **só o que difere do padrão** — o bloco
que você não personalizou continua vindo de `prompts.py`, e assim uma melhoria
futura no padrão chega sem apagar o seu texto. Há um botão de restaurar por
bloco. Os marcadores (`{generos}`, `{circunstancias}`…) são trocados por
substituição direta, nunca por `str.format`, porque o formato de saída é um JSON
cheio de chaves literais.

### Um prompt por classe

Cada classe carrega o **seu próprio prompt** (campo `skill`), as circunstâncias
que destrava e o JSON da tabela complementar que exige. Só os blocos das
classes marcadas entram na chamada: mandar os dez blocos e o formato de todas
as tabelas em toda notícia seria pagar por instrução que não se aplica. Medido
hoje, sem o texto da notícia:

| o que o BERT marcou | prompt | ~tokens |
| --- | --- | --- |
| nada (sem modelo treinado): catálogo inteiro | 23.757 car. | ~6.750 |
| duas classes | 16.273 car. | ~4.620 |
| nenhuma classe (leu e descartou) | 12.448 car. | ~3.540 |

A razão medida para este conteúdo é de **3,52 caracteres por token**, e o teto
da conta é de **8.000 tokens por minuto** — é essa a folga que a marcação
compra. O núcleo vai primeiro, para que a API consiga mantê-lo em cache.

O texto da notícia entra em cima disso, e é o que a **limpeza** (passo 2 do
fluxo) reduz. Medido nas notícias do banco: **-19,8% de caracteres**. Numa
ocorrência real de três notícias, a chamada caiu de ~7.794 para ~7.576 tokens —
218 de folga num teto de 8.000.

A regra da poda é **invertida de propósito**: uma linha só sai quando casa com
um padrão de ruído de portal **E** não carrega marcador de fato nenhum (palavra
do domínio, idade, contagem de pessoas, horário). A versão anterior fazia o
contrário — mantinha o que parecia relevante — e por isso apagava fato material:
saíram "Mulher estava atrás de viatura", que posiciona a vítima em relação à
polícia, e "os dois estavam conscientes e sem gravidade aparente". Era por isso
que ela vivia desligada. Deixar lixo passar custa ~40 tokens; deixar fato sair
custa uma classificação errada, então o erro tem de cair sempre para o lado
barato.

A limpeza também **desofusca**: `m0rto a tir0s`, como as redes sociais escrevem
para furar moderação automática, volta a ser `morto a tiros`. A troca de dígito
por letra só acontece quando a palavra mistura letra e dígito **e** o resultado
é um termo do domínio — `45 anos` e `41º BPM` não têm letra junto do dígito e
nunca são tocados, porque idade adulterada seria pior que a ofuscação.

Quatro blocos saem vazios de propósito e não viajam: `crianca_adolescente_baleado`
(sai da idade, que o prompt mestre já pede), `politico_baleado` (sai do
`tipo_vitima`), `chacina` (sai da contagem) e `violencia_genero` (o juízo dela
está dentro de `mulheres_baleadas` e `lgbtqiapn`, que é onde a pergunta
aparece). Título e definição sozinhos não dizem à LLM o que FAZER.

A tela **Gestão** cadastra indicadores e edita as skills sem mexer no código,
gravando em `indicadores_fonte.json`.

## Classificador antes da LLM

Uma tarefa só, multirrótulo, treinada sobre um corpus rotulado à mão: quais das
oito classes a notícia sustenta. Não há filtro binário antes dela — a notícia
fora do escopo é simplesmente a que não recebe nenhuma classe, e esses exemplos
negativos são tão necessários quanto os positivos: sem eles o modelo nunca
aprende a recusar nada.

**Multirrótulo, não multiclasse.** A pergunta não é *qual* é a classe da
notícia, é *quais* das oito ela sustenta. Cada classe tem a sua própria saída,
passa pela sua própria **sigmoide** e é comparada com o **seu próprio limiar**:
nenhuma exclui outra, não há `softmax` nem `argmax`, e a leitura não para na
primeira que passa. Uma notícia pode sair com nenhuma classe, com uma ou com as
oito, e todas as marcadas vão juntas para a LLM.

Duas coisas fazem um modelo multirrótulo *parecer* que só marca uma classe, e as
duas são tratadas no treino:

- **Peso das classes na perda** (`pos_weight`). Com 20 exemplos de uma classe
  contra 400 sem ela, a perda não ponderada já é quase mínima num modelo que
  nunca a marca. O peso de cada classe é a razão entre os negativos e os
  positivos dela no treino, com teto de 10.
- **Limiar por classe**, calibrado na **validação**. A saída de uma classe rara
  costuma ficar abaixo de 0,5 mesmo quando ela é a certa — sob o corte fixo ela
  nunca aparece, por bem ordenadas que estejam as pontuações. Cada classe recebe
  o limiar que maximiza o seu F1 na validação, e esse limiar é gravado junto com
  os pesos. As métricas trazem também o F1 do **mesmo** modelo com todos os
  cortes em 0,5: a diferença entre os dois é o efeito medido da calibração.

A ficha da vítima **não** se anota no corpus — o rótulo é só a lista de
indicadores, porque é só isso que o BERT aprende do texto. Nome, idade, gênero,
local e circunstância de cada vítima quem preenche é a LLM, depois, usando os
indicadores marcados como recorte.

### Corpus (`/corpus`)

Repositório de notícias rotuladas, com link e texto guardados para que a origem
de cada rótulo seja auditável. Mostra a distribuição por classe e diz se já há
exemplos suficientes para uma medição honesta. Importa e exporta planilha
`.xlsx`. O título não é um campo à parte: cola-se na primeira linha do texto.

#### Estar no corpus não é estar no treino

Duas perguntas, duas colunas:

| coluna | pergunta |
| --- | --- |
| `rotulada` | alguém leu e decidiu quais indicadores se aplicam |
| `aprovada` | alguém **revisou** a decisão e assumiu que ela está certa |

**Só o aprovado treina o modelo.** O motivo é a divisão de trabalho real: quem
cadastra a ocorrência é o analista júnior, quem revisa é o sênior. Um rótulo
errado do júnior não produz um erro visível — produz um modelo pior, que erra em
notícia que ninguém está olhando. É o tipo de defeito que não aparece em tela
nenhuma, e por isso a trava fica na consulta de treino (`corpus.TREINAVEL`,
escrita uma vez só) e não na confiança.

As três portas entram diferente:

| porta | entra |
| --- | --- |
| Rotulagem na fila de `/recebidas` (júnior) | sem aprovação |
| **Aprovação de ocorrência** em `/aprovacao` (sênior) | **aprovada** |
| Importação de planilha | sem aprovação, salvo se marcar *já aprovar* |

O que não está aprovado **não é descartado**: fica visível e editável em
`/corpus`, filtrável por *Esperando revisão*, com um botão de aprovar por linha.
É a fila de trabalho do sênior. A tela mostra os dois números lado a lado —
`aprovadas: estas treinam` e `esperando revisão` — para que a diferença seja
visível em vez de surpresa na hora de treinar.

Mudar o rótulo **derruba a aprovação** e libera a vaga de partição: rótulo novo
é afirmação nova, e quem assinou a anterior não assinou esta. O formulário de
cada notícia tem a caixa de aprovação no mesmo envio, para que corrigir e
assumir a correção sejam um clique só.

Aprovar uma ocorrência tem a caixa **servir de exemplo de treino** marcada por
padrão. Desmarcá-la grava a notícia no corpus **sem** aprovação, em vez de não
gravar: serve ao registro que está certo mas é exemplo ruim — texto truncado,
duas ocorrências misturadas, fonte duvidosa.

> **Limite conhecido:** a aplicação não tem login nem papéis. A separação
> júnior/sênior é garantida pelo **fluxo** (quem cadastra não é quem aprova a
> ocorrência), não por permissão. Quem abrir `/corpus` pode aprovar.

#### A trava dos 30 positivos

**O treino só roda quando CADA classe do catálogo tem 30 positivos aprovados**
(`corpus.MINIMO_POR_CLASSE`). Vale para a tela e para a linha de comando, que
sai com código 1 em vez de treinar.

Até 14/10 esse número só avisava, e o que o aviso deixava passar era pior que o
incômodo da trava: uma classe com 2 exemplos — ou com **zero**, no caso de um
indicador recém-criado — treinava para dizer sempre não e saía com F1 0. Esse
zero parece defeito do modelo e é falta de dado.

A conta é sobre **todas** as classes, inclusive as que ainda não têm exemplo
nenhum. Criar um indicador novo portanto **barra o treino** até ele ser
anotado. A saída, quando uma classe rara segura o resto, é treinar só o
subconjunto pronto — a tela imprime o comando já montado:

```bash
python rodar_treino.py --classe mulheres_baleadas --classe acao_policial
```

#### Acrescentar um indicador novo

O catálogo é dado, não código: `/gestao` cadastra e todo o resto se ajusta. Mas
duas coisas não são automáticas, e as duas eram silenciosas antes de 14/10:

| armadilha | o que o sistema faz agora |
| --- | --- |
| O modelo treinado **ignora** a classe nova — a inferência lê os rótulos do manifesto dele, não do catálogo | `inferencia.defasagem()` compara os dois, e `/modelos` e `/recebidas` mostram um aviso vermelho nomeando o que o modelo não conhece |
| O corpus que já existe vira exemplo **negativo** da classe nova, porque aquelas notícias nunca foram lidas com a pergunta | o indicador guarda `criado_em`; `/corpus` conta quantas notícias são anteriores a ele e oferece o filtro *Reler por causa de* |

O caminho completo: cadastrar com definição e skill → reler o corpus antigo
pelo filtro → juntar 30 positivos aprovados → treinar → **Reavaliar o arquivo**
em `/recebidas`, que relê o acervo com o modelo novo. O último passo é o que
recupera a notícia que foi arquivada antes de o indicador existir.

A divisão treino/validação/teste é **estratificada por indicador**, feita só
sobre o que está **aprovado**, e **gravada no banco**: refazer o sorteio a cada
treino mudaria o conjunto de teste e tornaria os modelos incomparáveis.
Distribui uma notícia por vez, sempre começando pelo indicador com menos
exemplos ainda por distribuir, e a notícia sem nenhum indicador conta como um
estrato dela. É isso que garante exemplo de *cada*
indicador nas três partições — sem exemplo na validação não há limiar para
calibrar, e sem exemplo no teste o indicador sai com F1 zero, que parece erro do
modelo e é falta de dado. A tela mostra a contagem por indicador em cada
partição, e o botão mostra o código exato no balão.

### Treino (`rodar_treino.py`)

Três estratégias chegam à mesma saída multirrótulo, e o treino implementa as
três para que a escolha seja um resultado medido:

| estratégia | o que é | redes | custo |
|---|---|---|---|
| `unico` (padrão) | uma rede com uma saída sigmoide por indicador | 1 | um treino, um checkpoint, uma passagem por notícia |
| `por_indicador` | uma rede binária por indicador (um-contra-resto) | 8 | 8 treinos, 8 checkpoints em disco, ~3 GB de RAM na inferência |
| `cluster` | os três modelos por indicador, votando | 24 | ~17 GB em disco; carrega um trio por vez, ~2,1 GB |

Em `unico` o encoder é compartilhado, então o que o modelo aprende sobre a
linguagem das notícias serve a todos os indicadores. Em `por_indicador` cada
indicador ganha uma rede inteira, com o seu próprio ponto de parada, e um
indicador raro não disputa a capacidade da rede com os frequentes — ao preço de
multiplicar treino, disco e memória por oito. Em `cluster` são três modelos
diferentes por indicador: **2 de 3 votos** classificam, 1 voto manda o caso a
uma pessoa e 0 arquiva. O corte é a fração `2/3` e não o número 2, porque em
`unico` e `por_indicador` há UMA rede — com o 2 cravado, nenhuma classe
passaria nessas estratégias e a LLM nunca seria chamada.

**Qual vence não se decide por argumento:** as três são medidas na mesma
partição de teste e aparecem como linhas da mesma tabela em `/modelos`. O
cluster reporta ainda os três cortes no mesmo teste — `com_um_voto`, o número
principal (2 de 3) e `com_unanimidade` — mais `revisao_humana`, a fração de
decisões que ficou com 1 voto e precisa de pessoa.

```bash
python rodar_treino.py --dividir            # divide e compara os 3 modelos
python rodar_treino.py --dispositivo cpu    # força CPU
python rodar_treino.py --modelo bertimbau-base --grade 1

# compara as DUAS estratégias na mesma partição de teste
python rodar_treino.py --estrategia ambas --grade 1

# uma rede binária por indicador
python rodar_treino.py --estrategia por_indicador --grade 1

# treina só um subconjunto, para medir uma classe sem as outras disputando
python rodar_treino.py --classe mulheres_baleadas --classe lgbtqiapn

# mede o efeito das correções de desbalanceamento, desligando-as
python rodar_treino.py --sem-peso-classes --sem-calibrar-limiar --grade 1
```

Três modelos comparados, todos **monolíngues de português** — BERTimbau base,
BERTimbau large e Albertina PT-BR —, cada um com quatro configurações de
hiperparâmetros. O recorte é deliberado: o corpus é de notícias brasileiras, e um
vocabulário treinado em português gasta menos subpalavras em "baleado" ou
"milícia". Em troca, o trabalho não afirma que o português vence o multilíngue —
essa comparação ficou fora. Os hiperparâmetros são escolhidos pelo **F1 macro na validação**,
nunca no teste: olhar o teste para escolher a configuração vaza informação e
infla o resultado. A comparação completa ajusta 12 redes em `unico` e 96 em
`por_indicador` (3 modelos × 4 configurações × 8 indicadores), e é por isso que
a comparação entre as estratégias costuma rodar com `--grade 1`.

Cada execução grava acurácia, precisão, recall e F1 (macro, micro e por classe),
a matriz 2×2 de cada indicador, o limiar calibrado de cada um, o F1 do mesmo
modelo com todos os cortes em 0,5 e a média de indicadores marcados por notícia
ao lado da média do gabarito — é esta última que responde direto a "ele está
marcando vários?". Falhas também são registradas — um modelo que não cabe na
placa de vídeo é resultado, não motivo para omitir a linha.

Os pesos do melhor modelo vão para `modelos/indicadores`, com um `modelo.json`
que declara a estratégia, a ordem dos indicadores, o limiar de cada um e o
`max_length` usado. As duas estratégias convivem em disco, em subpastas
separadas: treinar uma não apaga a outra, e é o manifesto que decide qual
responde em produção.

### Resultados (`/modelos`) e testes (`/testar`)

`/modelos` traz duas tabelas de resultado. O **placar** reduz a uma linha por
modelo — a melhor configuração dele —, ordenado pelo **F1 macro na validação**,
que é o mesmo critério com que o treino decide quais pesos gravar; a primeira
linha leva as etiquetas `melhor` e `em produção`. A tabela de **todas as
execuções** fica abaixo, ordenada pelo F1 no teste, com a estratégia, o número de
redes e o efeito da calibração em colunas próprias. Quando a primeira linha de
uma não é a da outra, isso é informação: a configuração com melhor número no
teste não era a que a validação apontava, e adotá-la seria escolher pelo teste.
Também há as matrizes de confusão por indicador e o aprendizado por época. `/testar` é o playground: cole uma notícia
e veja **todos** os indicadores marcados, a confiança de cada classe e o limiar
que cada uma precisava vencer, **antes de qualquer LLM** — é assim que se julga
se a LLM está corrigindo o classificador ou apenas repetindo-o. O campo de
limiar troca todos os cortes por um número só, para ver o que apareceria ou
desapareceria. Um texto que o modelo errou pode ser enviado ao corpus com um
clique.

### GPU ou CPU

`--dispositivo auto` (padrão) usa a GPU quando existe e cai para CPU quando não.
Em CPU o treino é muito mais lento, mas roda na máquina que houver. Veja
`requirements.txt` para instalar o torch em cada modo.

O que a semente (`--semente`, padrão 42) garante: a mesma divisão do corpus, os
mesmos pesos iniciais da cabeça de classificação e a mesma ordem dos lotes. Em
CPU, repetir a mesma configuração dá o mesmo número. Em GPU não dá: os núcleos
acumulam em ordem variável e as métricas batem só nas primeiras casas decimais.
A comparação entre modelos vale nos dois casos; a igualdade dígito a dígito,
só em CPU.

## Ajuda embutida (botão `? Ajuda`)

Toda tela tem um botão de ajuda na barra superior. Ele abre um painel que não
aparece sozinho nunca — é consulta, não tutorial, e quem já sabe usar a tela
não pode ser interrompido por ela.

O painel tem três partes:

| parte | o que traz |
| --- | --- |
| **Parte por parte** | cada seção da tela, na ordem em que ela aparece, explicada para quem nunca viu o sistema |
| **Por que esta tela é assim** | a decisão de projeto por trás dela: o erro que a motivou, a alternativa descartada, o limite assumido |
| **Palavras e siglas desta tela** | o glossário, filtrado para os termos que aquela tela usa |

São **16 telas e 54 termos**, cerca de 13 mil palavras. O glossário cobre desde
o vocabulário técnico (BERT, F1 macro, limiar, época, sigmoide, VRAM) até o do
domínio (chacina, vitimização policial, BPM, UPP, ECA, LGBTQIAPN+, cis e
travesti) — nenhuma sigla aparece numa tela sem ter explicação ali.

### Onde o texto mora

Em [`ajuda.py`](ajuda.py), e não dentro dos templates. Texto explicativo preso
no HTML é texto que ninguém revisa: some numa refatoração, contradiz a tela ao
lado e não há como conferir se alguma tela ficou sem. Como dado, ele é
desenhado pelo `base.html` sozinho, corrigido num lugar só, e **auditável**:

- `ajuda.para(endpoint)` resolve a ajuda da rota aberta, com o glossário já
  filtrado. Devolve `None` quando não há, e aí o botão simplesmente não existe;
- `ajuda.conferir(rotas)` aponta rota sem ajuda, ajuda apontando para rota que
  não existe mais, seção vazia, e termo citado numa tela que não está no
  glossário — que é o erro mais comum, porque faz a tela prometer uma
  explicação que não aparece;
- `ajuda.SEM_TELA` lista as sete rotas que de propósito não têm ajuda: três
  baixam arquivo, duas alimentam os campos encadeados de endereço e duas
  redirecionam. Nenhuma desenha tela.

### Como é desenhado

Um `<dialog>` nativo: o navegador dá o fundo escuro, o fechar com `Esc` e o
foco preso dentro do painel sem script nenhum para isso. O script tem doze
linhas e só faz abrir, fechar e fechar ao clicar fora. No celular o painel
ocupa a tela inteira e o glossário empilha — 170px de coluna fixa não cabem ao
lado do texto em 360px de largura.

## Documentação do TCC

`docs/TCC-Classificador-Violencia-Armada.docx` traz método, implementação e
protocolo de avaliação: fundamentação, construção do corpus, divisão amostral,
modelos comparados, busca de hiperparâmetros, métricas, reprodutibilidade em
GPU e CPU, limitações e ameaças à validade, mais apêndices com as categorias,
os indicadores e as diretrizes de anotação.

```bash
pip install python-docx
python gerar_documentacao.py
```

As tabelas de apêndice são geradas a partir de `categorias_fonte.json`,
`indicadores_fonte.json` e do catálogo de modelos — alterar uma categoria e
regerar mantém o documento em dia com o sistema, em vez de deixar os dois
divergirem em silêncio. A seção de resultados é um esqueleto a preencher com a
saída real do treino.

## Arquivos de configuração

Nenhum deles é obrigatório para subir a aplicação: os que faltam caem no padrão
do código. Eles existem para que uma decisão de curadoria não precise de um
commit.

- `indicadores_fonte.json` — indicadores, definições e skills. Editável em
  `/gestao`. É o arquivo que decide quais classes o BERT aprende.
- `prompts_fonte.json` — os blocos de prompt que você personalizou em
  `/prompts`, **e só eles**. Não existe até a primeira edição; o que não está
  aqui vem de `prompts.py`, para que uma melhoria futura no padrão chegue sem
  apagar o seu texto.
- `vocabulario_fonte.json` — as sete listas editáveis em `/locais`: corporações,
  cargos de agente, status de serviço do agente, cargos políticos, status do
  político e os contextos de ação policial e de crime organizado. Guarda só a
  diferença em relação ao padrão do código.
- `locais.json` — 27 estados, 5.571 municípios e os bairros cadastrados. Lista
  fechada e encadeada (estado → cidade → bairro) para evitar grafias diferentes
  que quebrariam o alerta de duplicidade e as estatísticas. CRUD em `/locais`.
- `partidos.json` — siglas de partido para a vítima política. Sigla fora da
  lista é preservada como veio: o TSE muda a lista e o texto não espera.
- `categorias_fonte.json` — **fora de uso desde 06/10**, quando a motivação saiu
  do sistema. O que ainda sai dele e viaja no prompt são as seções PAPEL e
  RECORTE DE ANÁLISE: edite o JSON e rode `python gerar_dicionario.py` para
  regerar o `dicionario_categorias.txt`. As outras seções do arquivo gerado
  (as 15 categorias, os dois níveis, as regras de desempate) **não são lidas** —
  ficam como registro do conceito.
- `ajuda.py` — o texto de ajuda de cada tela e o glossário de 54 termos. É
  código, não JSON, porque é texto longo com parágrafos: editar prosa dentro
  de um JSON de uma linha só é pior para quem escreve do que editar prosa em
  Python. Ver a seção *Ajuda embutida*.
- `ouro_extracao.json` — o gabarito campo a campo de `avaliar_extracao.py`, que
  mede a extração da LLM contra leitura humana. Aceita `--repeticoes` para medir
  a variação entre chamadas, que é o que revela extração instável.

A limpeza do texto não tem arquivo de configuração: os padrões de ruído moram em
`preprocessar_noticia.PADROES_RUIDO`, e cada um deles saiu de uma linha real das
notícias do banco. `PODAR = False` desliga a poda e deixa só a normalização, que
nunca remove linha de conteúdo.
