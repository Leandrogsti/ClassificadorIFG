# Fluxograma do treinamento

Companheiro de [fluxograma.md](fluxograma.md), que descreve o caminho de uma
notícia até virar registro. Aqui é o outro ciclo: como o classificador aprende.

Fonte em Mermaid; renderiza no GitHub e na pré-visualização do VS Code. Para
exportar imagem, <https://mermaid.live>.

Mantenha junto do código: um fluxograma que descreve uma versão antiga do fluxo
é pior que nenhum.

---

## 1. De onde vem o rótulo

Antes de treinar é preciso ter exemplo rotulado, e há **três portas** por onde
ele entra. Nenhuma delas pede trabalho extra: todas aproveitam decisão que
alguém já tomou por outro motivo.

```mermaid
flowchart TD
    R["Fila de recebidas<br/><small>júnior rotula a notícia que chegou</small>"]
    A["Aprovação de ocorrência<br/><small>o sênior confirma: entra APROVADA</small>"]
    I["Importação de planilha<br/><small>corpus que já existia fora daqui</small>"]

    D{"Esse texto<br/>já está no corpus?"}
    CORPUS[("corpus_noticia<br/><small>texto · rótulos · partição</small>")]
    DESC["Devolve o id do que já existe<br/><small>nada é duplicado</small>"]

    R --> D
    A --> D
    I --> D
    D -- "não" --> CORPUS
    D -- "sim" --> DESC
    CORPUS --> AP{"Aprovada por<br/>um sênior?"}
    AP -- "não" --> ESP["Espera revisão em /corpus<br/><small>não treina, não se perde</small>"]
    ESP -- "o sênior aprova" --> AP
    AP -- "sim" --> P["dividir()<br/><small>estratifica o corpus APROVADO, antes de cada treino</small>"]
    P --> CORPUS

    classDef banco fill:#fff,stroke:#b6bfcd,stroke-dasharray:5 4;
    classDef nao fill:#fbe9e8,stroke:#b3261e;
    class CORPUS banco;
    class DESC nao;
    classDef aviso fill:#fdf1e3,stroke:#b0561a;
    class ESP aviso;
```

**Por que a verificação de repetição.** A mesma notícia chega por mais de uma
porta — o analista rotula na fila e depois aprova a ocorrência feita dela. Sem
a comparação por `impressao` (sha1 do texto normalizado), o mesmo exemplo
entraria duas vezes e poderia cair **em treino e em teste ao mesmo tempo**.
Isso é vazamento: a métrica sobe sem o modelo melhorar.

**Por que a partição é atribuída em LOTE, e não na inserção.** A notícia nasce
com `particao` em branco e quem a preenche é `dividir()`, que o treino chama
sozinho quando encontra exemplo pendente — ninguém precisa clicar em
"sortear".

Sortear na hora de inserir foi tentado e é **pior**, por um motivo que só
aparece medindo: a estratificação olha quanto falta de cada indicador para
equilibrar as três partições, e uma notícia por vez não tem esse panorama. Na
medição, três classes ficaram sem nenhum exemplo na validação e o teste ficou
sem nenhum exemplo negativo. Em lote, o algoritmo começa sempre pelo indicador
com menos exemplos ainda por distribuir, e é isso que garante exemplo de cada
um nas três partições.

A divisão feita fica **gravada no banco**: refazer o sorteio a cada treino
mudaria o conjunto de teste e tornaria dois modelos incomparáveis.

**Rótulo vazio é rótulo.** A notícia lida e sem nenhum indicador é o exemplo
negativo, e sem ele o modelo nunca aprende a recusar o que está fora do escopo.
A coluna `rotulada` existe só para distinguir "li e não há nada" de "ninguém
olhou ainda".

**Rotulado não é aprovado, e só o aprovado treina.** `aprovada` é a segunda
coluna, e a razão dela é a divisão de trabalho real: quem cadastra a ocorrência
é o analista júnior, quem revisa é o sênior. Um rótulo errado do júnior não
produz um erro visível — produz um modelo pior, errando em notícia que ninguém
está olhando. É o defeito que não aparece em tela nenhuma, e por isso a trava
fica na consulta de treino (`corpus.TREINAVEL`), escrita uma vez só: espalhar
`rotulada = 1` por sete consultas é o que permitia esquecer uma.

Mudar o rótulo derruba a aprovação e libera a vaga de partição — rótulo novo é
afirmação nova, e um exemplo que saiu do treino não guarda lugar reservado.

**E o treino só roda com 30 positivos aprovados de CADA classe**
(`corpus.MINIMO_POR_CLASSE`), na tela e na linha de comando. Antes isso era um
aviso, e o que ele deixava passar era pior que o incômodo da trava: uma classe
com 2 exemplos — ou com zero, num indicador recém-criado — treinava para dizer
sempre não e saía com F1 0, que parece defeito do modelo e é falta de dado. A
saída, quando uma classe rara segura o resto, é `rodar_treino.py --classe`, que
aplica a mesma trava só ao subconjunto pedido.

---

## 2. O treinamento, passo a passo

```mermaid
flowchart TD
    START["/modelos · Treinar agora"]
    SUB["Subprocesso em segundo plano<br/><small>rodar_treino.py</small>"]
    PRONTO{"Prontidão:<br/>3+ exemplos por classe?"}
    PARA["Recusa e diz o que falta"]

    DIV["Divisão 70/15/15<br/><small>só o que ainda não tem partição</small>"]
    LER["Lê treino, validação e teste<br/><small>só rotulada = 1</small>"]
    COD["Codifica: lista de nomes → vetor de 0 e 1"]

    LOOP["Para cada UNIDADE a treinar"]
    PESO["Calcula pos_weight<br/><small>negativos ÷ positivos, teto 10</small>"]
    FIT["Ajusta a rede<br/><small>AdamW · warmup 10% · BCEWithLogitsLoss</small>"]
    PROB["Guarda as PROBABILIDADES<br/><small>validação e teste</small>"]
    FREE["Descarta a rede da memória"]

    CAL["Calibra o limiar de cada indicador<br/><small>na VALIDAÇÃO</small>"]
    MED["Mede no TESTE<br/><small>F1 macro, micro, por indicador</small>"]
    CMP["Compara com o corte fixo 0,5"]
    REG[("treino_execucao<br/><small>métricas · hiperparâmetros · duração</small>")]
    SEL{"Melhor F1 macro<br/>NA VALIDAÇÃO"}
    SALVA["Retreina a vencedora e grava os pesos<br/><small>+ modelo.json</small>"]

    START --> SUB --> PRONTO
    PRONTO -- "não" --> PARA
    PRONTO -- "sim" --> DIV --> LER --> COD --> LOOP
    LOOP --> PESO --> FIT --> PROB --> FREE
    FREE -- "próxima unidade" --> LOOP
    FREE -- "acabaram" --> CAL --> MED --> CMP --> REG --> SEL --> SALVA

    classDef modelo fill:#e4edf8,stroke:#1f5fa8,stroke-width:2px;
    classDef banco fill:#fff,stroke:#b6bfcd,stroke-dasharray:5 4;
    classDef nao fill:#fbe9e8,stroke:#b3261e;
    class FIT modelo;
    class REG banco;
    class PARA nao;
```

### Por que cada passo está ali

**Subprocesso em segundo plano.** O treino leva minutos e carrega um BERT
inteiro. Dentro da requisição, ele penduraria o navegador e derrubaria o
servidor junto se estourasse a memória. Em processo separado, a aplicação segue
respondendo e um treino que falha morre sozinho.

**Prontidão antes de tudo.** São precisos ao menos 3 exemplos da classe mais
rara — um para cada partição. Abaixo de 30, a métrica daquela classe é ruído:
com 4 exemplos no teste, acertar 3 ou 4 move o recall de 0,75 para 1,00. O
número não trava nada, só avisa.

**Divisão 70/15/15, e só do que falta.** Resortear tudo a cada treino mudaria o
conjunto de teste, e dois modelos medidos em testes diferentes não são
comparáveis — um modelo avaliado na semana passada teria sido medido num teste
que já não existe. A estratificação é por classe e uma notícia por vez,
começando pela **mais rara**: é o que garante exemplo dela nas três partições.
Sem exemplo na validação não há limiar para calibrar; sem exemplo no teste não
há métrica para reportar.

**Codificação em vetor.** Lista de nomes vira vetor de 0 e 1, uma posição por
indicador. É multirrótulo, não multiclasse: a pergunta não é "qual é o
indicador desta notícia" e sim "quais destes oito ela sustenta". Vetor todo
zero é o exemplo negativo, e é válido.

**`pos_weight` na perda.** Com 20 exemplos de um indicador contra 400 sem ele,
a perda não ponderada é quase minimizada por um modelo que **nunca o marca** —
e o sintoma disso é o clássico "só sai um indicador". O peso é a razão entre
negativos e positivos, com teto de 10 para que uma classe com 2 exemplos não
domine o gradiente.

**A rede devolve probabilidade, não decisão.** O corte vem depois, da
calibração. É isso que permite medir o mesmo modelo com cortes diferentes sem
retreinar.

**Descartar a rede antes da próxima.** É o que torna possível treinar as 24
redes do cluster numa placa de 8 GB: só uma vive por vez.

**Calibrar na VALIDAÇÃO, medir no TESTE.** Cortar todo mundo em 0,5 é palpite:
a saída de um indicador raro costuma ficar abaixo disso mesmo quando é o rótulo
certo, e aí ele nunca aparece. Cada indicador recebe o limiar que maximiza o F1
*dele* na validação. Calibrar no teste inflaria o número que vai para a banca.
Empate fica com o limiar mais baixo de propósito: aqui um falso positivo é
descartável — a LLM recebe o bloco daquela classe e pode descartá-la — enquanto
um falso negativo é perda definitiva, porque a classe que o BERT não marcou nem
chega a ser avaliada.

**A comparação com 0,5.** As métricas trazem também o F1 que o *mesmo* modelo
teria com todos os cortes em 0,5. A diferença entre os dois é o efeito medido
da calibração — não uma afirmação de que ela ajuda.

**Seleção pela validação.** O melhor é escolhido pelo F1 macro na validação,
não no teste. Escolher pelo teste seria usá-lo duas vezes e o número deixaria
de ser estimativa limpa.

**Retreinar a vencedora para gravar.** As execuções da grade rodam sem salvar
pesos; só a vencedora é refeita com `salvar_modelo=True`. É o preço de não
guardar um checkpoint de cada execução — com 4 configurações × 3 modelos, seria
muito disco para jogar fora.

---

## 3. As três estratégias

O que muda entre elas é só o que conta como **unidade** no laço acima.

```mermaid
flowchart TD
    U["unico<br/><small>1 rede · 8 saídas sigmoide</small>"]
    P["por_indicador<br/><small>8 redes binárias</small>"]
    C["cluster<br/><small>24 redes · 3 modelos × 8 indicadores</small>"]

    U --> U1["encoder compartilhado:<br/>o que aprende sobre a língua<br/>serve a todos os indicadores"]
    P --> P1["indicador raro não disputa<br/>capacidade com os frequentes;<br/>cada um com seu ponto de parada"]
    C --> C1["votação: 2 de 3 classificam;<br/>1 voto vira alerta para<br/>uma pessoa"]

    classDef modelo fill:#e4edf8,stroke:#1f5fa8,stroke-width:2px;
    class U,P,C modelo;
```

Qual é melhor é **resultado medido**, não palpite: `--estrategia todas` treina
as três na mesma partição de teste, e é isso que responde a pergunta.

### O voto do cluster

```mermaid
flowchart LR
    T["Texto"]
    G["Carrega as 3 redes<br/>de UM indicador<br/><small>~2,1 GB</small>"]
    V{"Quantas<br/>marcaram?"}
    Z["0 · ausente"]
    D["1 · ALERTA, uma pessoa verifica"]
    M["2 ou 3 · classificado"]
    N["Libera e passa<br/>ao próximo indicador"]

    T --> G --> V
    V -- "0" --> Z
    V -- "1" --> D
    V -- "2-3" --> M
    Z --> N
    D --> N
    M --> N

    classDef duvida fill:#fbeedf,stroke:#b0561a,stroke-width:2px;
    classDef nao fill:#fbe9e8,stroke:#b3261e;
    class D duvida;
    class Z nao;
```

**Por que um grupo de três por vez.** As 24 redes somam ~17 GB em disco e não
cabem juntas nos 8 GB da placa. Um grupo de três ocupa ~2,1 GB. O ciclo é por
indicador porque o **voto** é por indicador: as três de um precisam estar vivas
ao mesmo tempo para votar, as dos outros não. São oito ciclos de carga por
notícia — mais lento, e é esse o preço de caber.

**Por que cada rede tem o seu próprio limiar.** Se as três compartilhassem o
corte, discordariam menos por construção — e a discordância é justamente o
sinal que manda o caso para uma pessoa.

**Por que 2 de 3, e não unanimidade.** O corte sai de
`FRACAO_PARA_PROSSEGUIR = 2/3`, e é fração de propósito: cravado o número 3,
as estratégias de UMA rede — `unico` e `por_indicador` — nunca passariam, e a
LLM jamais seria chamada nelas. Em 1 de 1 a fração dá 1 voto, que é o
comportamento certo.

Unanimidade ficou de fora porque ela custa recall, e aqui o falso positivo é
descartável: a LLM recebe o bloco daquela classe e pode descartá-la. O falso
negativo, não — a notícia vai para o arquivo e ninguém mais a vê.

As métricas do cluster deixam a escolha auditável, com os três cortes medidos
no MESMO teste: `com_um_voto` (1 voto), o número principal (2 de 3) e
`com_unanimidade` (3 de 3). `revisao_humana` é a fração de decisões que ficou
com 1 voto e precisa de pessoa. Sem esses números, o corte seria palpite.

---

## 4. O ciclo da aprovação

Existe, e é a segunda porta do diagrama 1. Passo a passo:

```mermaid
flowchart TD
    A["Sênior clica em Aprovar<br/><small>caixa 'servir de treino' marcada</small>"]
    AL{"Há alerta de<br/>duplicidade aberto?"}
    BLOQ["Recusa: resolva o alerta antes"]
    ST["Grava status = aprovado<br/><small>quem aprovou, quando, observação</small>"]
    FEN["Lê os fenômenos CONFIRMADOS<br/>da ocorrência"]
    FILTRO["Mantém só as classes do BERT<br/><small>descarta as derivadas de regra</small>"]
    EACH["Para cada notícia da ocorrência"]
    DUP{"Texto já<br/>no corpus?"}
    PULA["Conta como repetida"]
    INS["Grava no corpus<br/><small>rotulada = 1 · aprovada = 1</small>"]
    MSG["Avisa quantas foram e com quais rótulos"]

    A --> AL
    AL -- "sim" --> BLOQ
    AL -- "não" --> ST --> FEN --> FILTRO --> EACH --> DUP
    DUP -- "sim" --> PULA
    DUP -- "não" --> INS
    PULA --> MSG
    INS --> MSG

    classDef nao fill:#fbe9e8,stroke:#b3261e;
    class BLOQ,PULA nao;
```

### Por que cada passo está ali

**Tudo na mesma transação.** A gravação do status e o envio ao corpus são uma
operação só. Não pode existir registro aprovado sem o exemplo de treino
correspondente — se o segundo falhar, o primeiro também desfaz.

**Só as classes do BERT viram rótulo.** Hoje a única de fora é `chacina`, que
existe apenas como regra sobre a lista de vítimas. Ensiná-la ao classificador
seria pedir que ele adivinhasse do texto algo que a contagem já sabe de forma
exata, e criar uma segunda resposta capaz de contradizer a primeira.

`vitimizacao_policial` já esteve nesse grupo e **saiu dele** em 09/10: ela tem
classe de BERT e regra. O motivo é o portão da LLM — a regra roda DEPOIS da
LLM, e o portão acontece ANTES, então policial baleado de folga não disparava
classe nenhuma e ia para o arquivo em silêncio.

**Uma linha de corpus por notícia, não por ocorrência.** Em produção o BERT lê
**uma notícia de cada vez**; treiná-lo com várias notícias concatenadas
ensinaria um formato que ele nunca vai encontrar.

**Limite conhecido e assumido.** O rótulo é da *ocorrência*, e cada notícia
recebe o conjunto inteiro. Numa ocorrência com várias fontes, uma notícia curta
pode receber uma classe que ela sozinha não sustenta — uma que não menciona a
polícia receberia `acao_policial` assim mesmo. Por isso elas entram editáveis
em `/corpus`, e não como verdade intocável.

**Reaprovar não duplica.** A verificação por impressão faz a segunda aprovação
não acrescentar nada, e a mensagem diz quantas já estavam lá.

**Esta é a única porta que já entra aprovada**, e é para isso que ela existe:
quem aprova a ocorrência é o sênior que revisou o trabalho do júnior, então a
decisão dele é a assinatura que o treino exige. Desmarcar a caixa *servir de
exemplo de treino* grava a notícia sem a assinatura, em vez de não gravar —
serve ao registro que está certo mas é exemplo ruim: texto truncado, duas
ocorrências misturadas, fonte duvidosa.

---

## 5. O ciclo inteiro, de volta ao começo

```mermaid
flowchart LR
    N["Notícia chega"] --> F["Fila de recebidas"]
    F --> B{"BERT marcou?"}
    B -- "não há modelo" --> H["Analista rotula"]
    B -- "0 voto em tudo" --> ARQ["Arquivo<br/><small>acervo, sem LLM</small>"]
    B -- "1 voto" --> H
    B -- "no corte ou acima" --> O["Vira ocorrência"]
    H --> CO[("corpus")]
    H -- "tem fenômeno" --> O
    ARQ -- "modelo novo:<br/>reavaliar o arquivo" --> B
    O --> L["LLM aprofunda<br/><small>só os blocos marcados</small>"]
    L --> AP["Analista confere e aprova"]
    AP --> CO
    CO --> TR["Treino"]
    TR -- "modelo novo" --> B

    classDef banco fill:#fff,stroke:#b6bfcd,stroke-dasharray:5 4;
    classDef modelo fill:#e4edf8,stroke:#1f5fa8,stroke-width:2px;
    classDef frio fill:#eef0f3,stroke:#6b7584;
    class CO banco;
    class B,TR modelo;
    class ARQ frio;
```

As duas setas que chegam no corpus são as duas portas: a rotulagem na fila
(inclusive a que conclui "não há fenômeno") e a aprovação da ocorrência. As
duas alimentam o mesmo treino, e o modelo que sai dele volta para o começo.

**O que esse ciclo compra além do treino.** Enquanto o BERT não marca nada, a
chamada à LLM leva o catálogo inteiro de indicadores: ~6.750 tokens de prompt,
contra um teto de 8.000 por minuto — e o texto da notícia ainda entra em cima
disso. Com o BERT marcando duas classes, o prompt cai para ~4.620, e com
nenhuma, para ~3.540. É essa folga que permite subir o `reasoning_effort` da
LLM, que é o que estabiliza a extração da lista de vítimas: medido em três
leituras do mesmo texto, `low` devolveu 2, 0 e 1 vítima, e `medium` devolveu 2,
2 e 2. O corpus não serve só ao BERT — ele é o que torna a chamada da LLM
melhor.
