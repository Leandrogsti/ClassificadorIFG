"""
O TEXTO DE AJUDA DE CADA TELA, escrito para quem nunca viu o sistema.

POR QUE ESTE ARQUIVO EXISTE

A ajuda não está espalhada dentro dos templates, e isso é de propósito. Texto
explicativo preso no HTML é texto que ninguém revisa: some numa refatoração,
contradiz a tela ao lado e não dá para conferir se alguma tela ficou sem. Aqui
ele é DADO — uma estrutura que o `base.html` desenha sozinho, que o script de
auditoria consegue percorrer e que se corrige num lugar só.

COMO LER A ESTRUTURA

`TELAS` é um dicionário com uma entrada por rota (o `endpoint` do Flask, que é
o nome da função que responde por aquele endereço). Cada entrada tem:

- `titulo`   — o nome da tela, como ela aparece no menu;
- `resumo`   — uma frase: o que esta tela é, para quem abriu pela primeira vez;
- `secoes`   — uma lista de (nome da parte, explicação). A ordem acompanha a
               ordem em que as partes aparecem na tela, para que a pessoa possa
               ler com a tela ao lado;
- `construcao` — POR QUE a tela é assim. É a parte que não se descobre usando:
               a decisão de projeto, o erro que a motivou, a alternativa
               descartada. É o que um trabalho de conclusão precisa registrar;
- `termos`   — as chaves do `GLOSSARIO` que importam nesta tela.

`GLOSSARIO` é compartilhado: o mesmo termo é explicado uma vez e aparece em
todas as telas que o usam. Explicar "F1" de novo em cada tela produziria cinco
explicações que, com o tempo, deixariam de concordar entre si.

PARA QUEM ESTE TEXTO É ESCRITO

Para quem não é da área. Nada de "ajuste fino do encoder" sem dizer o que é um
encoder; nada de sigla solta. Se uma frase só faz sentido para quem já sabe, a
frase está errada, não o leitor.
"""

# --------------------------------------------------------------------------
# GLOSSÁRIO
# --------------------------------------------------------------------------
# Toda sigla e todo jargão que aparece em alguma tela. A regra: a explicação
# tem de bastar sozinha, sem mandar o leitor procurar outra entrada antes.

GLOSSARIO: dict[str, tuple[str, str]] = {
    # ---------------------------------------------------------- o essencial
    "bert": (
        "BERT",
        "Um tipo de programa que aprende a ler. Em vez de seguir regras que"
        " alguém escreveu, ele vê milhares de exemplos já classificados por"
        " pessoas e descobre sozinho os padrões. O nome é a sigla em inglês de"
        " uma técnica publicada pelo Google em 2018, e virou nome comum para"
        " essa família de programas. Aqui ele faz UMA coisa: lê a notícia e diz"
        " quais indicadores ela sustenta. Ele não escreve, não resume e não"
        " preenche ficha de vítima.",
    ),
    "bertimbau_albertina": (
        "BERTimbau e Albertina",
        "Os dois BERT que este sistema compara. Os dois já vêm treinados em"
        " português por universidades (o BERTimbau na USP, o Albertina em"
        " Lisboa e no Porto) e aqui recebem um treino adicional, curto, só com"
        " notícias de violência armada. É como contratar quem já sabe"
        " português e ensinar o vocabulário específico do trabalho, em vez de"
        " alfabetizar do zero.",
    ),
    "llm": (
        "LLM",
        "Sigla em inglês para 'modelo de linguagem grande'. É o tipo de"
        " programa que conversa e escreve texto — a mesma família do ChatGPT."
        " Aqui ele quase nunca decide quais indicadores a notícia tem: isso é"
        " trabalho do BERT, e o LLM entra depois para APROFUNDAR dentro do que"
        " o BERT já marcou — montar a lista de vítimas, preencher nome, idade,"
        " gênero e as circunstâncias de cada uma. A exceção é"
        " `violencia_genero`, que não tem classe no BERT: feminicídio e"
        " lgbtfobia são decididos pelo LLM, dentro dos blocos de"
        " mulheres_baleadas e lgbtqiapn.",
    ),
    "prompt": (
        "Prompt",
        "O texto de instruções que o sistema envia ao LLM junto com a notícia."
        " É o equivalente ao briefing que se daria a um estagiário: o que ele"
        " é, o que tem de procurar, em que formato responder e o que NÃO pode"
        " inventar. Quanto mais longo o prompt, mais caro e mais lento fica"
        " cada pedido — por isso ele é montado em pedaços e só os pedaços"
        " necessários viajam.",
    ),
    "skill": (
        "Skill",
        "O pedaço de prompt que pertence a UM indicador. A skill de"
        " 'mulheres baleadas', por exemplo, é o parágrafo que ensina o LLM a"
        " decidir se houve feminicídio. Ela só é enviada quando o BERT marcou"
        " aquele indicador — mandar todas sempre seria pagar por instrução que"
        " não se aplica àquela notícia.",
    ),
    "indicador": (
        "Indicador",
        "Uma característica da ocorrência que o instituto acompanha: criança"
        " ou adolescente baleado, mulheres baleadas, ação policial, e assim"
        " por diante. Uma mesma notícia pode ter vários ao mesmo tempo, ou"
        " nenhum. O conjunto de indicadores é o CATÁLOGO, e é ele que define o"
        " escopo do sistema: tiroteio que não tem nenhuma dessas"
        " características está fora.",
    ),
    "fenomeno_circunstancia": (
        "Fenômeno observado × circunstância",
        "Duas coisas diferentes que é fácil confundir. O FENÔMENO é da"
        " ocorrência inteira ('houve ação policial neste caso'). A"
        " CIRCUNSTÂNCIA é de UMA vítima ('esta mulher foi atingida por bala"
        " perdida'). Numa mesma ocorrência, uma vítima pode ter bala perdida"
        " e outra não.",
    ),
    "corpus": (
        "Corpus",
        "O conjunto de notícias já classificadas por pessoas, que serve de"
        " material de estudo para o BERT. A palavra vem do latim e quer dizer"
        " 'corpo' — o corpo de textos. Sem corpus não há treino: o programa"
        " aprende imitando as decisões que estão ali.",
    ),

    # ----------------------------------------------- como o modelo decide
    "multirrotulo": (
        "Multirrótulo",
        "A pergunta que o sistema faz não é 'qual é O indicador desta"
        " notícia?' e sim 'QUAIS destes indicadores ela sustenta?'. A diferença"
        " é grande: na primeira, as opções competem e só uma vence; na segunda,"
        " cada indicador decide sozinho se entra, e a resposta pode ser"
        " nenhum, um ou todos.",
    ),
    "sigmoide": (
        "Sigmoide e softmax",
        "Duas formas de o programa transformar os seus cálculos internos em"
        " porcentagem. A SIGMOIDE trata cada indicador separadamente: cada um"
        " recebe a sua própria nota de 0 a 100%, e elas não precisam somar"
        " 100. O SOFTMAX faz as opções disputarem entre si, e aí elas somam"
        " 100% — serve para escolher UMA. Este sistema usa sigmoide, porque"
        " uma notícia pode ter vários indicadores de uma vez.",
    ),
    "limiar": (
        "Limiar",
        "A nota de corte de cada indicador. O programa não responde 'sim' ou"
        " 'não': ele dá uma nota de 0 a 100%. O limiar é o número a partir do"
        " qual essa nota vira um 'sim'. Cada indicador tem o seu, porque"
        " indicador raro costuma receber nota baixa mesmo quando está certo —"
        " com um corte único em 50% ele nunca apareceria.",
    ),
    "calibracao": (
        "Calibração do limiar",
        "O processo de descobrir a nota de corte de cada indicador. O sistema"
        " testa vários cortes no conjunto de VALIDAÇÃO e fica com o que dá o"
        " melhor resultado para aquele indicador. É feito na validação, e"
        " nunca no teste: escolher o corte olhando o teste seria como estudar"
        " com o gabarito na mão e depois dizer que a nota foi honesta.",
    ),
    "encoder": (
        "Encoder",
        "A parte do programa que transforma o texto em números antes de"
        " qualquer decisão. É ela que carrega o conhecimento de português. Na"
        " estratégia de uma rede só, esse encoder é COMPARTILHADO por todos os"
        " indicadores — o que ele aprende sobre a linguagem de notícia de"
        " tiroteio serve a todos de uma vez.",
    ),
    "checkpoint": (
        "Checkpoint",
        "O arquivo em disco com tudo o que o programa aprendeu, gravado ao fim"
        " do treino. É literalmente o modelo: sem ele, o aprendizado se perde"
        " quando o computador desliga. Cada treino grava o seu.",
    ),

    # ------------------------------------------------------- medir e avaliar
    "treino_validacao_teste": (
        "Treino, validação e teste",
        "O corpus é dividido em três montes. O de TREINO é o material de"
        " estudo. O de VALIDAÇÃO é o simulado: serve para ajustar as"
        " configurações e as notas de corte. O de TESTE fica guardado e só é"
        " usado no fim, uma vez, para dizer quanto o modelo realmente acertou."
        " Misturar os três é o erro clássico: o programa decora as respostas e"
        " a nota final deixa de significar alguma coisa.",
    ),
    "particao": (
        "Partição",
        "A qual dos três montes — treino, validação ou teste — cada notícia"
        " pertence. Fica gravada no banco e não muda mais. Se mudasse a cada"
        " treino, dois modelos teriam sido avaliados em provas diferentes e"
        " não daria para comparar.",
    ),
    "estratificacao": (
        "Estratificação",
        "O cuidado de distribuir os exemplos entre os três montes de forma"
        " equilibrada, em vez de sortear às cegas. Sem isso, um indicador raro"
        " pode cair inteiro no treino e ficar sem nenhum exemplo no teste — e"
        " aí ele sai com nota zero, que parece defeito do programa e é falta"
        " de exemplo.",
    ),
    "f1": (
        "F1",
        "A nota que resume o desempenho num número de 0 a 1. Ela combina duas"
        " coisas que puxam para lados opostos: não deixar passar e não marcar"
        " errado. Um programa que marca tudo acha todos os casos mas erra"
        " muito; um que quase nunca marca erra pouco mas deixa passar quase"
        " tudo. O F1 só é alto quando os dois estão bons ao mesmo tempo.",
    ),
    "precisao_recall": (
        "Precisão e recall",
        "As duas metades do F1. PRECISÃO responde: das vezes que o programa"
        " marcou este indicador, quantas estavam certas? RECALL responde: dos"
        " casos que realmente tinham este indicador, quantos ele encontrou?"
        " Dá para ter 100% de precisão marcando só o caso óbvio, e 100% de"
        " recall marcando tudo. É por isso que se olha os dois.",
    ),
    "acuracia": (
        "Acurácia",
        "A porcentagem de acertos no total. Engana quando os casos são raros:"
        " num conjunto em que só 5% das notícias têm certo indicador, um"
        " programa que NUNCA o marca já acerta 95%. Por isso ela aparece aqui"
        " ao lado do F1, e não sozinha.",
    ),
    "macro_micro": (
        "Macro e micro",
        "Duas formas de juntar as notas dos vários indicadores num número só."
        " A MACRO tira a média simples: cada indicador pesa igual, então um"
        " indicador raro que vai mal derruba o número. A MICRO soma todos os"
        " casos antes de calcular, então os indicadores frequentes dominam. A"
        " macro é a mais exigente, e é por ela que este sistema escolhe o"
        " melhor modelo.",
    ),
    "suporte": (
        "Suporte",
        "Quantos exemplos daquele indicador existiam no conjunto que está"
        " sendo medido. É o número que diz se a nota ao lado significa alguma"
        " coisa: um F1 de 1,00 com suporte 2 quer dizer que o programa acertou"
        " dois casos, o que não prova nada.",
    ),
    "matriz_confusao": (
        "Matriz de confusão",
        "Uma tabelinha de quatro casas que abre o resultado de um indicador:"
        " quantas vezes ele marcou e estava certo, quantas marcou e estava"
        " errado, quantas deixou passar e quantas acertou ao não marcar. É"
        " onde se vê o TIPO de erro, e não só a quantidade.",
    ),

    # ------------------------------------------------------- o treino em si
    "epoca": (
        "Época",
        "Uma passada completa por todo o material de estudo. Treinar por três"
        " épocas é ler o mesmo conjunto de notícias três vezes. Poucas épocas"
        " e o programa não aprende; muitas e ele decora, passando a ir bem no"
        " material de estudo e mal em notícia nova.",
    ),
    "hiperparametro": (
        "Hiperparâmetro",
        "As configurações do treino, decididas por quem treina e não"
        " aprendidas pelo programa: quantas épocas, de quanto em quanto ele"
        " ajusta o que aprendeu, quantas notícias olha por vez. Como não há"
        " como saber de antemão a melhor combinação, o sistema testa várias e"
        " compara.",
    ),
    "learning_rate": (
        "Taxa de aprendizado",
        "O tamanho do passo que o programa dá ao se corrigir depois de cada"
        " erro. Passo grande demais e ele oscila sem assentar; pequeno demais"
        " e o treino não sai do lugar. É o hiperparâmetro que mais muda o"
        " resultado.",
    ),
    "lote": (
        "Lote",
        "Quantas notícias o programa olha de uma vez antes de se corrigir."
        " Lote maior deixa o treino mais estável e mais rápido, mas exige mais"
        " memória da placa de vídeo.",
    ),
    "pos_weight": (
        "Peso da classe",
        "Um ajuste que torna mais caro errar nos indicadores raros. Com 20"
        " exemplos de um indicador contra 400 sem ele, um programa que NUNCA o"
        " marca já acerta quase tudo — e é exatamente nisso que ele se acomoda"
        " se ninguém intervier. O peso multiplica o custo desse erro e é o que"
        " impede o resultado de colapsar em 'não marca nada'.",
    ),
    "semente": (
        "Semente",
        "Um número que faz o sorteio do computador dar sempre o mesmo"
        " resultado. Sorteio de computador não é aleatório de verdade: parte"
        " de um número inicial. Fixando esse número, outra pessoa consegue"
        " repetir exatamente a mesma divisão e chegar ao mesmo resultado — que"
        " é o que torna o trabalho reproduzível.",
    ),
    "gpu_cpu": (
        "GPU e CPU",
        "A CPU é o processador comum do computador. A GPU é a placa de vídeo,"
        " que faz muitas contas simples ao mesmo tempo e por isso treina estes"
        " programas muito mais rápido. O sistema usa a placa quando ela existe"
        " e cai para o processador comum quando não — o resultado é o mesmo, o"
        " tempo é que muda.",
    ),
    "cuda_vram": (
        "CUDA e VRAM",
        "CUDA é a tecnologia que permite usar placas de vídeo NVIDIA para"
        " cálculo. VRAM é a memória dessa placa. É a VRAM que limita o tamanho"
        " do programa que cabe: por isso o sistema treina uma rede de cada vez"
        " e descarta antes da próxima.",
    ),

    # ------------------------------------------------- estratégias e triagem
    "estrategias": (
        "As três estratégias",
        "Três formas de chegar à mesma resposta. ÚNICO: uma rede só, com uma"
        " saída por indicador. POR INDICADOR: uma rede separada para cada"
        " indicador, cada uma respondendo sim ou não. CLUSTER: três redes por"
        " indicador, de programas-base diferentes, que votam. Qual é melhor"
        " não se decide discutindo — treina-se as três na mesma prova e"
        " compara-se.",
    ),
    "cluster_votacao": (
        "Votação do cluster",
        "Na estratégia de três redes por indicador, conta-se quantas marcaram."
        " Duas de três bastam para seguir. Uma só vira alerta, que é dúvida"
        " para uma pessoa resolver. Nenhuma significa que o indicador não"
        " está ali. Exigir as três (unanimidade) foi medido e descartado:"
        " custa casos encontrados sem ganho que compense.",
    ),
    "concordancia": (
        "Concordância",
        "Quantas das redes marcaram o mesmo indicador. É esse número que"
        " define para onde a notícia vai: seguir sozinha, parar para alguém"
        " conferir, ou ir para o arquivo.",
    ),
    "estados_fila": (
        "Os estados da fila",
        "Por onde uma notícia recebida pode estar. NA FILA: ninguém olhou"
        " ainda (ou não há modelo treinado). ARQUIVO: o classificador leu e"
        " não achou nenhum indicador — fica guardada, não vira pendência e"
        " não gasta chamada ao LLM. ALERTA: houve dúvida, uma pessoa precisa"
        " confirmar. COM FENÔMENO: pode virar ocorrência. SEM FENÔMENO: uma"
        " pessoa leu e concluiu que não há nada. VIROU OCORRÊNCIA: já virou"
        " registro.",
    ),
    "arquivo": (
        "Arquivo",
        "O acervo de notícias em que o classificador não encontrou nenhum"
        " indicador. Elas não são apagadas nem viram pendência: ficam ali e"
        " são RELIDAS quando houver um modelo novo ou um indicador novo. É o"
        " que permite dizer hoje 'assalto comum está fora do escopo' sem"
        " perder esses casos caso o escopo mude amanhã.",
    ),

    # --------------------------------------------------- custo e limpeza
    "token": (
        "Token",
        "O pedaço de texto que o LLM conta para cobrar e para medir o seu"
        " limite. Não é exatamente uma palavra: palavras comuns valem um"
        " token, palavras longas se partem em dois ou três. Neste conteúdo em"
        " português, a conta medida é de cerca de 3,5 caracteres por token.",
    ),
    "tpm": (
        "Teto por minuto (TPM)",
        "O limite de tokens que a conta pode enviar a cada minuto — aqui,"
        " 8.000. É o limite de VELOCIDADE, não de tamanho: passar dele faz o"
        " pedido ser recusado mesmo que o texto não seja grande. É esse teto"
        " que obriga o prompt a ser montado em pedaços.",
    ),
    "poda": (
        "Poda do texto",
        "A limpeza automática da notícia antes de ela ser classificada: saem"
        " assinatura do jornalista, data de publicação, 'Compartilhe no"
        " Whatsapp', legenda de foto repetida e botão de baixar aplicativo."
        " A regra é conservadora de propósito — uma linha só é removida se"
        " casar com um padrão de ruído conhecido E não contiver nada que a"
        " classificação use (nome de vítima, idade, arma, horário).",
    ),
    "impressao": (
        "Impressão (sha1)",
        "Uma espécie de impressão digital do texto: um código curto calculado"
        " a partir dele, em que textos iguais dão o mesmo código. Serve para"
        " perceber que a mesma notícia já está no sistema, mesmo colada de"
        " outro lugar. 'sha1' é o nome do cálculo usado.",
    ),

    # -------------------------------------------------- revisão e qualidade
    "aprovada": (
        "Rotulada × aprovada",
        "Duas perguntas diferentes sobre a mesma notícia do corpus. ROTULADA:"
        " alguém leu e decidiu quais indicadores se aplicam. APROVADA: alguém"
        " REVISOU essa decisão e assumiu que está certa. Só o aprovado treina"
        " o modelo, porque quem cadastra é o analista júnior e quem revisa é o"
        " sênior — e um rótulo errado não produz erro visível, produz um"
        " modelo pior errando onde ninguém está olhando.",
    ),
    "divergencia_proposta": (
        "Divergência e proposta",
        "Quando duas notícias da mesma ocorrência se contradizem, isso é uma"
        " DIVERGÊNCIA e fica registrada para o analista decidir. Quando o LLM"
        " discorda de uma correção que uma pessoa já fez — um ferido que uma"
        " notícia nova diz ter morrido —, a versão dele vira uma PROPOSTA à"
        " espera de aceitação, em vez de sobrescrever o que a pessoa escreveu.",
    ),
    "decidido_por": (
        "Decidido por: BERT, LLM ou regra",
        "De onde sai cada indicador. BERT: o classificador lê o texto e marca."
        " LLM: é um juízo de leitura que só o LLM faz, dentro do que o BERT já"
        " marcou. REGRA: o sistema CALCULA a partir da lista de vítimas, sem"
        " perguntar a ninguém — é o caso da chacina, que sai da contagem de"
        " mortos. Calcular é exato; perguntar de novo ao modelo criaria uma"
        " segunda resposta capaz de contradizer a conta.",
    ),

    # ------------------------------------------------- vocabulário do domínio
    "chacina": (
        "Chacina",
        "Três ou mais pessoas mortas por disparo na mesma ocorrência, contando"
        " civis e agentes de segurança que NÃO estavam em serviço. O agente em"
        " serviço fica fora da contagem porque, numa ação policial, ele está"
        " do outro lado do mesmo fato: se ele é baleado e mata três pessoas, a"
        " chacina é das três. O sistema calcula isso sozinho.",
    ),
    "vitimizacao_policial": (
        "Vitimização policial",
        "Agente de segurança que foi BALEADO — vítima, não autor. Vale estando"
        " de serviço ou de folga, e é diferente de 'ação policial', que exige"
        " o agente atuando numa operação no momento do disparo. Policial"
        " morto na folga é vitimização policial e não é ação policial.",
    ),
    "lgbtqiapn": (
        "LGBTQIAPN+",
        "Sigla que reúne lésbicas, gays, bissexuais, travestis e transexuais,"
        " queer, intersexo, assexuais, pansexuais e não-binárias, com o '+'"
        " para as identidades não listadas. No sistema é uma circunstância da"
        " vítima: registra-se quando a notícia AFIRMA a identidade, nunca"
        " quando ela seria deduzida de nome, profissão ou companhia.",
    ),
    "cis_trans": (
        "Cis, trans e travesti",
        "CIS é quem se identifica com o gênero atribuído no nascimento; TRANS"
        " é quem não se identifica. TRAVESTI é uma identidade feminina própria"
        " da América Latina, e o termo é reivindicado como tal — não é"
        " sinônimo de mulher trans, por isso aparece nomeado ao lado dela.",
    ),
    "eca": (
        "ECA",
        "Estatuto da Criança e do Adolescente, a lei brasileira de 1990. É"
        " dela que vem o corte de idade do sistema: criança vai até 11 anos"
        " incompletos e adolescente até 17, então 'criança ou adolescente"
        " baleado' é qualquer vítima com menos de 18.",
    ),
    "corporacoes": (
        "PM, PC, BPM, UPP, GAT",
        "PM é a Polícia Militar e PC a Polícia Civil. BPM é Batalhão de"
        " Polícia Militar, e o número identifica a unidade e a região — '41º"
        " BPM' é o batalhão de Irajá. UPP é Unidade de Polícia Pacificadora e"
        " GAT é Grupamento de Ações Táticas, equipes com funções específicas"
        " dentro da PM. Aparecem no campo 'unidade' da vítima agente.",
    ),
    "id_caso": (
        "ID_CASO e ID_VÍTIMA",
        "Os códigos que o sistema dá a cada ocorrência e a cada pessoa"
        " atingida, gerados sozinhos no momento em que o registro nasce."
        " Servem para citar um caso sem depender de nome ou endereço, que"
        " podem mudar ou estar errados.",
    ),

    # --------------------------------------------------------- informática
    "slug": (
        "Slug",
        "O nome técnico de uma opção, escrito sem acento, sem espaço e sem"
        " maiúscula — `mulheres_baleadas` em vez de 'Mulheres baleadas'. O"
        " slug é o que o programa usa internamente; o texto bonito é só o"
        " rótulo que aparece na tela. Separar os dois permite corrigir o"
        " rótulo sem quebrar nada.",
    ),
    "json": (
        "JSON",
        "Um formato de texto para guardar dados organizados, legível por"
        " pessoa e por programa. Aparece nas telas quando é preciso mostrar ou"
        " editar uma estrutura que não caberia em campos soltos — uma lista"
        " dentro de outra, por exemplo.",
    ),
    "xlsx": (
        "xlsx",
        "A extensão dos arquivos do Excel. Os botões de baixar geram esse"
        " formato para que a base possa ser aberta em planilha, com uma aba"
        " por tabela.",
    ),
    "crud": (
        "CRUD",
        "Sigla em inglês para as quatro operações básicas sobre uma lista:"
        " criar, ler, atualizar e apagar. Uma 'tela de CRUD' é simplesmente"
        " uma tela onde dá para incluir, ver, corrigir e remover itens.",
    ),
}


def termos(chaves) -> list[tuple[str, str]]:
    """Resolve uma lista de chaves do glossário, ignorando o que não existe.

    Ignorar em silêncio é deliberado: uma chave errada numa tela não pode
    derrubar a tela. A auditoria é que reclama — ver `conferir()`.
    """
    return [GLOSSARIO[c] for c in (chaves or []) if c in GLOSSARIO]


# --------------------------------------------------------------------------
# AS TELAS
# --------------------------------------------------------------------------
# A chave é o `endpoint` do Flask — o nome da função que responde por aquele
# endereço. É o mesmo nome que o menu usa em `base.html`, e é por isso que a
# ajuda consegue se achar sozinha sem cada template declarar nada.

TELAS: dict[str, dict] = {}


TELAS["fila_recebidas"] = {
    "titulo": "Recebidas",
    "resumo":
        "A porta de entrada. Toda notícia chega aqui antes de virar qualquer"
        " outra coisa, e é aqui que o classificador olha pela primeira vez.",
    "secoes": [
        ("Receber notícia",
         "Três campos: o link, a data da notícia e o texto colado. Só o texto"
         " é obrigatório. Ao salvar acontecem três coisas em sequência: o"
         " sistema confere se esse mesmo texto já entrou antes (e, se já,"
         " devolve o registro existente em vez de criar outro), limpa o texto"
         " tirando o ruído de portal, e deixa o classificador lê-lo. Nada"
         " disso depende de haver uma ocorrência aberta — o endereço só é"
         " pedido depois, se a notícia virar registro."),
        ("A faixa em que a notícia nasce",
         "Logo ao entrar, a notícia recebe um estado, que é a cor da linha na"
         " lista. Esse estado sai da CONCORDÂNCIA: quantos dos modelos"
         " marcaram cada indicador. Nenhum voto em nada manda para o arquivo;"
         " voto abaixo do corte vira alerta laranja, que é dúvida esperando"
         " uma pessoa; voto no corte ou acima libera a notícia para virar"
         " ocorrência. Sem modelo treinado não há voto nenhum, e tudo cai em"
         " 'na fila' para uma pessoa decidir."),
        ("Rotular uma notícia",
         "Marcar os indicadores que o texto sustenta. Marcar NENHUM é uma"
         " resposta válida e importante: 'li e não há fenômeno' é o exemplo"
         " que ensina o classificador a recusar o que está fora do escopo."
         " Essa decisão vai para o corpus de treino, mas entra lá SEM"
         " aprovação — quem cadastra não é quem autoriza o exemplo a treinar"
         " o modelo."),
        ("Arquivo e o botão de reavaliar",
         "O arquivo guarda o que o classificador leu e não reconheceu. Não é"
         " lixeira: é acervo. Um homem adulto baleado num assalto comum cai"
         " ali de propósito, porque assalto não está no catálogo de"
         " indicadores — e se um dia estiver, o botão 'Reavaliar o arquivo'"
         " relê tudo com o modelo novo e traz de volta o que passar a contar."),
        ("Busca e filtro por estado",
         "A lista pode ser filtrada por estado e por trecho de texto. O"
         " contador no menu conta só as PENDÊNCIAS — o que está na fila ou em"
         " alerta. Arquivo e 'sem fenômeno' não entram nessa conta, porque já"
         " têm uma decisão tomada e empilhá-los ali emperraria o trabalho com"
         " o caso mais comum que existe."),
    ],
    "construcao":
        "Esta tela não existia no começo: o sistema abria direto numa"
        " ocorrência, e isso obrigava a decidir que havia um caso ANTES de"
        " qualquer leitura. O resultado era que notícia fora de escopo gastava"
        " uma chamada ao LLM para receber 'nada se aplica' de volta, e notícia"
        " sem indicador nenhum não deixava rastro em lugar nenhum. Separar a"
        " fila da ocorrência resolveu os dois: o texto é gravado assim que"
        " chega, e o LLM só é chamado quando já se sabe que há algo ali."
        "\n\n"
        "O estado nasce da concordância, e não de a lista de indicadores estar"
        " vazia. Esse foi um defeito real e caro: uma notícia com dois votos"
        " de três em duas classes tinha lista vazia e era arquivada como se"
        " não tivesse nada — enterrando justamente a dúvida que precisava de"
        " uma pessoa.",
    "termos": ["bert", "indicador", "estados_fila", "arquivo", "concordancia",
               "cluster_votacao", "poda", "impressao", "corpus", "aprovada"],
}


TELAS["nova_ocorrencia"] = {
    "titulo": "Abertura da ocorrência",
    "resumo":
        "O formulário que transforma uma notícia em registro. É aqui, e só"
        " aqui, que se pede data, hora e endereço.",
    "secoes": [
        ("Data e hora do fato",
         "A data é a do FATO, não a da publicação da notícia — uma reportagem"
         " de hoje pode tratar de um tiroteio de ontem. A faixa horária"
         " (madrugada, manhã, tarde, noite) NÃO é digitada: o sistema a deriva"
         " da hora. Dois campos que podem se contradizer sempre acabam se"
         " contradizendo."),
        ("Endereço em listas encadeadas",
         "Estado, depois cidade, depois bairro — nessa ordem, e cada lista só"
         " carrega depois que a anterior é escolhida. A ordem não é capricho:"
         " 232 nomes de município se repetem entre estados no Brasil, então"
         " 'Santa Luzia' sem o estado não identifica lugar nenhum. A"
         " LOCALIDADE (sub-bairro, favela, conjunto) é texto livre e opcional,"
         " porque não existe lista fechada confiável nesse nível."),
        ("Quando vem da fila",
         "Se a ocorrência foi aberta a partir de uma notícia recebida, o texto"
         " dela já vem anexado e aparece na tela — não é preciso colar de"
         " novo. A classificação pelo LLM não roda automaticamente nesse"
         " momento: fica para o botão de reprocessar, já dentro da ocorrência."
         " O motivo é simples: seria gastar uma chamada antes de você conferir"
         " a data e o endereço que acabou de digitar."),
    ],
    "construcao":
        "O endereço ser pedido só aqui é consequência de a fila ter vindo"
        " antes. Enquanto a notícia está sendo triada, não se sabe se ela vai"
        " virar caso — e pedir cidade e bairro de algo que vai para o arquivo"
        " é trabalho jogado fora."
        "\n\n"
        "As listas fechadas de estado, cidade e bairro existem por causa da"
        " detecção de duplicidade e das estatísticas: 'Tijuca', 'tijuca' e"
        " 'Tijuça' digitados à mão viram três bairros diferentes nos números e"
        " impedem o sistema de perceber que dois registros são o mesmo caso.",
    "termos": ["id_caso", "estados_fila", "llm"],
}


TELAS["ocorrencia"] = {
    "titulo": "A ocorrência",
    "resumo":
        "A ficha completa de um caso: as notícias que o descrevem, os"
        " fenômenos observados e a lista de pessoas atingidas.",
    "secoes": [
        ("Classificação",
         "Os fenômenos observados do caso, com a ORIGEM de cada um ao lado —"
         " se veio do texto, de uma regra sobre a lista de vítimas, ou dos"
         " dois. Saber a origem é o que permite discordar da coisa certa: se"
         " o fenômeno veio de regra, discordar dele é corrigir uma vítima, não"
         " o modelo. Aparecem também o que o LLM tentou marcar sem"
         " autorização e o que ele não conseguiu sustentar."),
        ("Inclusão de notícia",
         "Cada notícia nova colada aqui passa a fazer parte do mesmo caso. Ao"
         " reprocessar, o LLM relê TODAS juntas, não só a última: é assim que"
         " o nome que só a terceira reportagem traz se junta à idade que só a"
         " primeira trazia. A ficha da vítima é a soma das fontes, e silêncio"
         " numa delas não apaga o que outra afirmou."),
        ("Verificação das vítimas",
         "A lista de pessoas atingidas, uma ficha por pessoa: nome, situação"
         " (ferida ou morta), gênero, grupo etário, idade, tipo (civil, agente"
         " de segurança, político), local e circunstâncias. Vítima é só quem"
         " foi ATINGIDO — não há vítima ilesa. Corrigir um campo aqui marca a"
         " vítima como editada à mão, e a partir daí o LLM não sobrescreve"
         " mais: se ele discordar, a versão dele vira uma proposta para você"
         " aceitar ou recusar."),
        ("Tabelas complementares",
         "Quando a vítima é marcada como agente de segurança, abrem-se os"
         " campos de corporação, cargo, situação de serviço e unidade. Quando"
         " é política, abrem-se partido, cargo e situação. Desmarcar o tipo"
         " apaga a tabela correspondente, para não sobrar uma patente"
         " pendurada em civil."),
        ("Circunstâncias calculadas",
         "A maioria das circunstâncias é marcada à mão ou pelo LLM, mas a"
         " CHACINA não: o sistema a calcula a partir da lista de vítimas e a"
         " mostra como somente leitura. Para mudá-la, corrige-se o que a"
         " gera — a situação da vítima ou a situação de serviço do agente."),
        ("Concluir e histórico",
         "Concluir grava o registro como não aprovado e o manda para a fila de"
         " aprovação. O histórico no fim da página lista toda alteração feita"
         " numa vítima: o valor anterior, o novo, quem mudou, quando e com que"
         " fonte. Sem isso, uma correção discutida semanas depois não teria"
         " como ser reconstruída."),
    ],
    "construcao":
        "A decisão que organiza esta tela é que CORREÇÃO DE PESSOA PREVALECE."
        " O LLM relê tudo a cada reprocessamento e poderia refazer a lista do"
        " zero; se fizesse, apagaria silenciosamente o trabalho do analista a"
        " cada notícia nova. Por isso a vítima editada à mão é preservada e a"
        " discordância do modelo vira proposta visível."
        "\n\n"
        "A contagem é de PESSOAS DISTINTAS, nunca a soma das notícias. Sem"
        " nome, o sistema compara idade, gênero e tipo, e um campo não"
        " informado nunca contradiz um informado. 'Um homem baleado' num"
        " jornal e 'dois homens baleados' em outro resultam em duas vítimas, e"
        " não em três — e esse cuidado é o que impede uma chacina que não"
        " houve de aparecer por soma errada.",
    "termos": ["llm", "fenomeno_circunstancia", "decidido_por", "chacina",
               "vitimizacao_policial", "divergencia_proposta", "cis_trans",
               "lgbtqiapn", "corporacoes", "eca", "id_caso"],
}


TELAS["inicio"] = {
    "titulo": "Ocorrências",
    "resumo":
        "A lista de todos os casos registrados, com busca por período e por"
        " situação.",
    "secoes": [
        ("A tabela",
         "Uma linha por ocorrência, com data do fato, local, quantas vítimas e"
         " quantas notícias estão vinculadas, e a situação do registro"
         " (rascunho, não aprovado, aprovado, unificada). Clicar na linha abre"
         " a ficha completa."),
        ("Busca por período",
         "O filtro de data é pela DATA DO FATO, não pela de cadastro. É a"
         " pergunta que se faz de verdade: 'o que aconteceu em setembro?', e"
         " não 'o que alguém digitou em setembro?'."),
        ("Baixar base",
         "Gera uma planilha com uma aba por tabela: ocorrências, vítimas,"
         " notícias, as tabelas complementares e o histórico. RESPEITA O"
         " FILTRO da tela — com status ou data marcados, o botão passa a"
         " dizer 'Baixar o filtrado' e todas as abas saem recortadas juntas,"
         " sem vítima órfã da ocorrência dela. Sem filtro nenhum, sai a base"
         " inteira, que é o caso de arquivar o estado da base num momento"
         " determinado."),
        ("Paginação",
         "Vinte por página. O filtro ativo é mantido ao mudar de página e ao"
         " baixar a planilha — o arquivo sai com o que está na tela, e não com"
         " a base toda, quando há busca."),
    ],
    "construcao":
        "Esta era a primeira tela do sistema e deixou de ser quando a fila de"
        " recebidas apareceu. Hoje ela é a tela de CONSULTA do que já virou"
        " registro, e não a porta de entrada — o trabalho começa em Recebidas.",
    "termos": ["id_caso", "xlsx", "estados_fila"],
}


TELAS["vitimas"] = {
    "titulo": "Vítimas",
    "resumo":
        "Todas as pessoas atingidas de todas as ocorrências, numa tabela só,"
        " com o local do caso ao lado de cada uma.",
    "secoes": [
        ("Por que uma tabela de pessoas",
         "A tabela de ocorrências responde 'quantos casos houve'. Esta"
         " responde 'quem foi atingido', que é uma pergunta diferente e é a"
         " que os indicadores medem. Uma ocorrência com cinco vítimas é um"
         " caso e cinco pessoas."),
        ("Os filtros",
         "Período, cidade, bairro, localidade (busca parcial), situação,"
         " gênero, cor/raça, grupo etário, tipo e circunstância. Cidade e"
         " bairro vêm do endereço da OCORRÊNCIA, não da vítima — é por isso"
         " que dá para buscar por local sem cruzar duas abas à mão."),
        ("Acompanhamento de feridos",
         "Um atalho de filtro, não uma tela separada: mostra os feridos dos"
         " últimos 90 dias. Existe porque a situação de uma vítima pode mudar"
         " depois — alguém ferido pode morrer semanas mais tarde, e isso"
         " altera a contagem e pode transformar o caso em chacina. O sistema"
         " recalcula os fenômenos sozinho quando a situação muda."),
        ("Edição rápida",
         "Dá para corrigir na própria linha a cor/raça, a situação e a data da"
         " morte, sem abrir a ocorrência. São justamente os campos que não vêm"
         " da notícia: a imprensa quase nunca informa cor/raça, e o"
         " acompanhamento de 90 dias existe para corrigir a situação de quem"
         " morreu depois. A descrição não está aqui — para mudá-la, abra a"
         " ocorrência."),
        ("Baixar planilha",
         "Gera a planilha só de vítimas, respeitando a busca que está na tela."
         " É a mesma aba 'vitima' do download da base completa, com cidade,"
         " bairro e localidade já ao lado de cada pessoa."),
    ],
    "construcao":
        "Cor/raça é editável aqui e NÃO é pedida ao LLM, de propósito. A"
        " imprensa quase nunca informa, e pedir ao modelo produziria palpite"
        " apresentado como dado. A coluna existe no banco e é preenchida à mão"
        " quando há fonte.",
    "termos": ["fenomeno_circunstancia", "chacina", "cis_trans", "lgbtqiapn",
               "id_caso", "xlsx", "vitimizacao_policial"],
}


TELAS["aprovacao"] = {
    "titulo": "Aprovação",
    "resumo":
        "A fila de registros concluídos esperando a revisão de um analista"
        " sênior. É aqui que o ciclo de aprendizado fecha.",
    "secoes": [
        ("O que está em jogo",
         "Aprovar faz duas coisas ao mesmo tempo, na mesma operação: libera o"
         " registro para a comunicação e AUTORIZA as notícias dele a treinar o"
         " classificador. As duas andam juntas de propósito — não pode existir"
         " registro aprovado sem o exemplo de treino correspondente, nem o"
         " contrário."),
        ("Servir de exemplo de treino",
         "A caixa vem marcada. Desmarcá-la grava as notícias no corpus SEM"
         " aprovação, em vez de não gravar. Serve ao caso em que o registro"
         " está certo mas a notícia é um exemplo ruim: texto truncado, duas"
         " ocorrências misturadas no mesmo texto, fonte duvidosa. Ela fica"
         " esperando revisão no Corpus, e não se perde."),
        ("Devolver ao analista",
         "Devolve o registro com um motivo escrito. O registro volta para"
         " 'não aprovado' e some desta fila até ser concluído de novo."),
        ("O bloqueio por duplicidade",
         "Se existir alerta de duplicidade aberto envolvendo o registro, a"
         " aprovação é recusada. Aprovar dois registros do mesmo fato"
         " contaria as vítimas duas vezes nos indicadores — e, pior, levaria"
         " o mesmo texto duas vezes ao corpus de treino."),
    ],
    "construcao":
        "Antes, aprovar só mudava o status: o trabalho de conferir a"
        " classificação não deixava nada atrás de si, e o corpus só crescia se"
        " alguém o alimentasse à mão. Hoje a conferência que o analista já fez"
        " É o rótulo, e é o rótulo de melhor qualidade do sistema, porque"
        " passou por duas leituras."
        "\n\n"
        "Limite conhecido e assumido: o rótulo é da OCORRÊNCIA e cada notícia"
        " dela recebe o conjunto inteiro. Numa ocorrência com várias fontes,"
        " uma notícia curta pode receber um indicador que ela sozinha não"
        " sustenta. Por isso elas entram EDITÁVEIS no Corpus, e não como"
        " verdade intocável."
        "\n\n"
        "Limite técnico: a aplicação não tem login nem papéis. A separação"
        " entre quem cadastra e quem aprova é garantida pelo FLUXO, não por"
        " permissão.",
    "termos": ["aprovada", "corpus", "indicador", "impressao", "id_caso"],
}


TELAS["alertas"] = {
    "titulo": "Alertas de duplicidade",
    "resumo":
        "Pares de registros que podem ser o mesmo fato, levantados pelo"
        " sistema para uma pessoa decidir.",
    "secoes": [
        ("Como o alerta aparece",
         "O sistema compara TRÊS campos, e só eles: a data do caso, a cidade e"
         " o bairro. Registros que batem nos três viram suspeita. O perfil das"
         " vítimas NÃO entra nessa comparação — é justamente o que a pessoa"
         " olha para decidir. Ele não une nada sozinho: juntar dois casos"
         " diferentes por engano é pior do que deixar dois registros do mesmo"
         " caso esperando alguém olhar."),
        ("Unificar",
         "Quando é o mesmo fato: as notícias dos dois registros passam a"
         " pertencer a um só, tudo é reprocessado pelo LLM com o conjunto"
         " completo de fontes, e o registro volta para a fila de aprovação. O"
         " registro absorvido fica marcado como unificado, e não é apagado."),
        ("Descartar o alerta",
         "Quando são fatos diferentes. Pede um motivo escrito, que fica"
         " registrado — um alerta descartado sem explicação volta a parecer"
         " pendência para quem olhar depois."),
        ("Desfazer unificação",
         "Se a união foi um engano, dá para separar os registros de novo. É o"
         " que torna a unificação uma decisão reversível em vez de"
         " definitiva."),
    ],
    "construcao":
        "A duplicidade é inevitável neste fluxo: a mesma ocorrência chega por"
        " jornais diferentes, em dias diferentes, e nem sempre quem cadastra a"
        " segunda sabe da primeira. O alerta existe para que isso seja"
        " percebido ANTES da aprovação — depois de aprovado, o caso duplicado"
        " já contaminou tanto os indicadores quanto o corpus de treino.",
    "termos": ["llm", "impressao", "id_caso", "divergencia_proposta"],
}


TELAS["corpus_repositorio"] = {
    "titulo": "Corpus de treino",
    "resumo":
        "O material de estudo do classificador: notícias já classificadas por"
        " pessoas, que é o que o programa imita para aprender.",
    "secoes": [
        ("Situação do corpus",
         "Seis números. TOTAL é tudo que está ali. ANOTADAS são as que já têm"
         " uma decisão. APROVADAS são as que foram revisadas e podem treinar —"
         " só estas contam. ESPERANDO REVISÃO é a diferença entre as duas, e é"
         " a fila de trabalho de quem revisa. SEM NENHUM INDICADOR são os"
         " exemplos negativos, que ensinam o programa a recusar o que está"
         " fora do escopo. NA FILA DE ANOTAÇÃO são as que ainda não têm"
         " decisão nenhuma."),
        ("Distribuição por indicador",
         "Quantos exemplos aprovados cada indicador tem, em barras. É o"
         " número que decide se o treino pode rodar: são precisos 30 positivos"
         " aprovados de CADA classe. Abaixo disso a medição daquela classe é"
         " ruído — com 4 exemplos no teste, acertar 3 ou 4 muda o resultado de"
         " 75% para 100%."),
        ("Divisão treino / validação / teste",
         "Mostra quantas notícias de cada indicador caíram em cada um dos três"
         " montes, e o botão revela o código exato que faz a divisão. Um"
         " indicador que não aparece na validação não pode ter a sua nota de"
         " corte calibrada; um que não aparece no teste sai com nota zero, que"
         " parece defeito do programa e é falta de exemplo."),
        ("Adicionar e importar",
         "Dá para colar uma notícia à mão ou importar uma planilha com as"
         " colunas 'texto' e 'indicadores'. A importação tem uma caixa de 'já"
         " aprovar', que só deve ser marcada se aquele corpus já foi revisado"
         " fora daqui."),
        ("Baixar o corpus",
         "A planilha sai com o texto, os indicadores, o anotador — e também"
         " com a APROVAÇÃO (se está aprovada, por quem e quando) e o CONJUNTO"
         " de cada notícia: treino, validação ou teste. São essas duas"
         " últimas que permitem reproduzir a medição fora daqui; sem elas, a"
         " planilha seria a lista das notícias e não o corpus que gerou os"
         " números."),
        ("A lista e a coluna de aprovação",
         "Cada linha tem um botão de Aprovar (ou Devolver, se já aprovada). O"
         " filtro 'Esperando revisão' isola exatamente o que falta revisar."
         " Mudar o rótulo de uma notícia derruba a aprovação dela: rótulo novo"
         " é afirmação nova, e quem assinou a anterior não assinou esta."),
        ("Reler por causa de um indicador novo",
         "Quando um indicador é criado depois de o corpus já existir, as"
         " notícias antigas nunca foram lidas com a pergunta nova — mas no"
         " treino elas contam como exemplo NEGATIVO dele. O aviso vermelho"
         " conta quantas são e o filtro as isola, para que a dívida seja"
         " visível em vez de virar um modelo que nunca marca a classe nova."),
    ],
    "construcao":
        "A separação entre ROTULADA e APROVADA é a decisão central desta tela,"
        " e vem da divisão de trabalho real: quem cadastra a ocorrência é o"
        " analista júnior, quem revisa é o sênior. Um rótulo errado não produz"
        " erro visível em lugar nenhum — produz um modelo pior, errando em"
        " notícia que ninguém está olhando. É defeito que não aparece em tela"
        " nenhuma, e por isso a trava está na consulta de treino e não na"
        " confiança."
        "\n\n"
        "Não se anota aqui nada que o LLM preencha: ficha de vítima,"
        " circunstância, tabela complementar. O rótulo é só a lista de"
        " indicadores, porque é só isso que o classificador aprende do texto.",
    "termos": ["corpus", "aprovada", "indicador", "treino_validacao_teste",
               "particao", "estratificacao", "suporte", "xlsx", "impressao",
               "bert"],
}


TELAS["corpus_noticia"] = {
    "titulo": "Uma notícia do corpus",
    "resumo":
        "A tela de anotação: marcar quais indicadores um texto sustenta, e"
        " aprovar esse rótulo para o treino.",
    "secoes": [
        ("Classes",
         "Uma caixa por indicador. Marcar várias é normal e esperado. Marcar"
         " NENHUMA é uma resposta válida: é o exemplo negativo, e sem ele o"
         " programa nunca aprende a recusar notícia fora do escopo. Salvar"
         " sem marcar nada tira a notícia da fila de anotação — 'nenhum se"
         " aplica' é uma decisão, não uma pendência."),
        ("A regra de anotação de cada indicador",
         "O bloco expansível mostra, para cada indicador, quando marcar e os"
         " exemplos cadastrados. É o manual de anotação, e está aqui — e não"
         " num documento à parte — para que duas pessoas anotando em semanas"
         " diferentes usem o mesmo critério."),
        ("Fonte",
         "O link da notícia. Guardado para que a origem de cada rótulo seja"
         " auditável: sem ele, nenhum resultado de treino se sustenta diante"
         " de uma banca."),
        ("Aprovação",
         "A caixa que autoriza este exemplo a treinar o modelo. Vem no mesmo"
         " envio que os rótulos, para que corrigir e assumir a correção sejam"
         " um clique só. Mudar os indicadores sem marcar a caixa tira a"
         " notícia do treino até alguém revisar."),
    ],
    "construcao":
        "O título da notícia não é um campo à parte: cola-se na primeira linha"
        " do texto. Um campo separado se desencontra do texto com o tempo, e o"
        " classificador lê o texto inteiro de qualquer forma."
        "\n\n"
        "A partição (treino, validação ou teste) JÁ ATRIBUÍDA é preservada ao"
        " corrigir o rótulo: mover um exemplo do teste para o treino"
        " invalidaria a comparação com as execuções anteriores. Desaprovar,"
        " sim, libera a vaga — exemplo que saiu do treino não guarda lugar.",
    "termos": ["corpus", "indicador", "aprovada", "particao",
               "treino_validacao_teste", "slug", "bert"],
}


TELAS["classificacoes"] = {
    "titulo": "Classificações do BERT",
    "resumo":
        "O registro de tudo o que o classificador leu e de tudo o que ele"
        " marcou, guardado antes de o LLM ver qualquer coisa.",
    "secoes": [
        ("Banco de entradas",
         "TODO texto que entra no sistema ganha uma linha aqui, antes de"
         " qualquer classificação e mesmo quando não existe modelo treinado."
         " É a garantia de que nada submetido se perde — nem o texto que não"
         " recebe indicador nenhum, nem o que chega antes do primeiro treino."),
        ("Situação",
         "Quantas marcações existem, quantas já foram conferidas por uma"
         " pessoa e quantas estão esperando. Conferir aqui não altera a"
         " ocorrência: ela já foi classificada. O que a conferência produz é"
         " exemplo de treino."),
        ("Onde o modelo erra",
         "Separa os dois tipos de erro, que pedem correções diferentes."
         " FALTOU MARCAR costuma ser falta de exemplo no corpus — o programa"
         " nunca viu casos suficientes daquele indicador. MARCOU À TOA costuma"
         " ser nota de corte baixa demais. Confundir os dois leva a anotar"
         " mais quando o problema era o corte, e vice-versa."),
        ("A confiança e o limiar lado a lado",
         "Cada classe aparece com a nota que recebeu e com a nota de corte que"
         " precisava vencer. É isso que torna a revisão legível: uma classe"
         " marcada com 0,42 não é erro de leitura se o corte calibrado dela é"
         " 0,30."),
        ("Enviar ao corpus",
         "O botão leva a marcação corrigida para o corpus já rotulada. É assim"
         " que o erro de hoje vira o acerto do próximo treino."),
    ],
    "construcao":
        "Esta tela existe por um motivo específico: sem ela, o erro do"
        " classificador DESAPARECE assim que o LLM o corrige. O registro final"
        " fica certo, o trabalho sai bem feito, e ninguém jamais fica sabendo"
        " que o classificador errou — nem quais classes precisam de mais"
        " exemplos. Guardar a marcação antes da correção é o que transforma o"
        " uso diário em dado de melhoria.",
    "termos": ["bert", "llm", "limiar", "calibracao", "corpus", "indicador",
               "aprovada", "impressao"],
}


TELAS["modelos"] = {
    "titulo": "Modelos e resultados",
    "resumo":
        "Onde o classificador é treinado e onde se compara, com número, qual"
        " configuração ficou melhor.",
    "secoes": [
        ("As três estratégias",
         "Três formas de chegar à mesma resposta. ÚNICO: uma rede só, com uma"
         " saída por indicador — o que ela aprende sobre a língua serve a"
         " todos os indicadores de uma vez. POR INDICADOR: uma rede separada"
         " para cada um, de modo que um indicador raro não disputa capacidade"
         " com os frequentes. CLUSTER: três redes por indicador, de"
         " programas-base diferentes, que votam entre si. Em todas, cada"
         " indicador tem a sua própria saída e decide sozinho — não há"
         " competição entre as classes."),
        ("Catálogo de modelos e grade",
         "Os três programas-base comparados, todos treinados em português por"
         " universidades, e as quatro configurações testadas em cada um. A"
         " comparação completa ajusta 12 redes na estratégia de rede única e"
         " 96 na de uma rede por indicador — por isso a comparação entre"
         " estratégias costuma rodar com a grade reduzida."),
        ("Ambiente de execução",
         "Qual placa de vídeo foi detectada, se o CUDA está disponível, qual"
         " dispositivo está em uso e qual versão do torch está instalada. A"
         " memória da placa não aparece aqui. Está nesta tela porque"
         " resultado de treino sem o ambiente em que rodou não é"
         " reproduzível."),
        ("Treinar",
         "Dispara o treino em segundo plano — dá para fechar a página. Antes"
         " de começar, o sistema distribui as notícias novas entre os três"
         " montes e confere se cada classe tem 30 positivos aprovados. Se"
         " alguma não tiver, o treino NÃO roda, e a tela diz exatamente quais"
         " faltam e quantos."),
        ("Placar e todas as execuções",
         "Duas tabelas, e a diferença entre elas é informação. O PLACAR reduz"
         " a uma linha por programa-base — a melhor configuração dele —"
         " ordenado pela nota na VALIDAÇÃO, que é o mesmo critério com que o"
         " treino decide quais pesos gravar. TODAS AS EXECUÇÕES fica abaixo,"
         " ordenada pela nota no TESTE. Quando a primeira linha de uma não é a"
         " da outra, isso quer dizer que a configuração com melhor número no"
         " teste não era a que a validação apontava — e adotá-la seria"
         " escolher olhando o gabarito."),
        ("Limiar de cada indicador em produção",
         "A nota de corte que cada indicador precisa vencer no modelo que está"
         " respondendo hoje. Serve para ler as outras telas: uma classe"
         " marcada com nota 0,42 não é erro se o corte dela é 0,30."),
        ("Aviso de modelo defasado",
         "Se alguém criou um indicador depois do último treino, aparece um"
         " aviso vermelho. O modelo lê a lista de classes do arquivo que ele"
         " mesmo gravou, e não do catálogo atual — então ele simplesmente"
         " nunca marca a classe nova, e notícia que só ela sustentaria vai"
         " para o arquivo em silêncio."),
    ],
    "construcao":
        "A regra que organiza esta tela: OS HIPERPARÂMETROS SÃO ESCOLHIDOS NA"
        " VALIDAÇÃO, NUNCA NO TESTE. Olhar o teste para escolher a"
        " configuração vaza informação e infla o resultado — a nota final"
        " deixa de dizer como o modelo se sai em notícia nova e passa a dizer"
        " como ele se sai na prova que ajudou a escolher."
        "\n\n"
        "Qual estratégia vence não se decide por argumento: as três são"
        " medidas na MESMA divisão de teste e aparecem como linhas da mesma"
        " tabela. Falhas também são registradas — um modelo que não cabe na"
        " placa de vídeo é resultado, não motivo para omitir a linha.",
    "termos": ["bert", "bertimbau_albertina", "estrategias", "cluster_votacao",
               "treino_validacao_teste", "f1", "macro_micro", "limiar",
               "calibracao", "hiperparametro", "epoca", "learning_rate",
               "lote", "gpu_cpu", "cuda_vram", "semente", "aprovada",
               "checkpoint", "encoder", "multirrotulo", "sigmoide"],
}


TELAS["modelo_execucao"] = {
    "titulo": "Uma execução de treino",
    "resumo":
        "O relatório completo de um treino: o que foi configurado, o que saiu"
        " e onde o modelo errou.",
    "secoes": [
        ("Hiperparâmetros",
         "As configurações usadas: quantas vezes o material foi relido, de"
         " quanto em quanto o programa se corrigiu, quantas notícias olhou por"
         " vez. Ficam registradas para que a execução possa ser repetida."),
        ("Métricas no conjunto de teste",
         "A nota geral. O F1 MACRO é o número principal: dá o mesmo peso a"
         " cada indicador, então um indicador raro que vai mal derruba o"
         " resultado — é a leitura mais exigente e a mais honesta quando as"
         " classes são desiguais. A acurácia aparece ao lado porque, sozinha,"
         " ela engana: num conjunto em que 5% têm certo indicador, nunca"
         " marcá-lo já dá 95%."),
        ("Quantos indicadores por notícia",
         "A média de indicadores que o modelo marcou por notícia, ao lado da"
         " média do gabarito. É o número que responde direto à pergunta 'ele"
         " está marcando só um?' — se o gabarito tem média 1,8 e o modelo tem"
         " 0,4, ele colapsou em marcar pouco, e isso não aparece no F1 geral."),
        ("Métricas por classe",
         "A nota de cada indicador separadamente, com o SUPORTE ao lado —"
         " quantos exemplos daquela classe havia no teste. O suporte é o que"
         " diz se a nota significa algo: F1 de 1,00 com suporte 2 quer dizer"
         " que acertou dois casos."),
        ("Como cada limiar foi escolhido",
         "A nota de corte de cada indicador e o motivo dela. Quando não havia"
         " exemplo daquela classe na validação, não há o que calibrar e o"
         " corte fica no valor neutro — o motivo registra isso, para que"
         " ninguém leia 0,5 como escolha."),
        ("Efeito da calibração",
         "O mesmo modelo medido duas vezes: com os cortes calibrados e com"
         " todos fixos em 0,5. A diferença entre as duas colunas é o efeito"
         " medido da calibração, e não uma afirmação de que ela ajuda."),
        ("Matrizes de confusão",
         "Uma tabelinha por indicador abrindo o resultado em quatro casas:"
         " marcou e acertou, marcou e errou, deixou passar, e acertou ao não"
         " marcar. É onde se vê o TIPO de erro. Marcar demais e deixar passar"
         " pedem correções opostas."),
        ("Aprendizado por época",
         "A curva do treino, leitura por leitura. Se o erro no material de"
         " estudo cai e o erro na validação sobe, o programa está decorando em"
         " vez de aprender, e treinar mais só piora."),
    ],
    "construcao":
        "Esta tela existe para que o resultado seja AUDITÁVEL, e não só"
        " anunciado. Cada número vem acompanhado do que permite julgá-lo: a"
        " nota vem com o suporte, o corte vem com o motivo, a calibração vem"
        " com o resultado sem ela. Um relatório que só mostra o número bom"
        " não serve nem para melhorar o sistema nem para defender o trabalho.",
    "termos": ["f1", "precisao_recall", "acuracia", "macro_micro", "suporte",
               "matriz_confusao", "limiar", "calibracao", "epoca",
               "hiperparametro", "learning_rate", "lote", "pos_weight",
               "treino_validacao_teste", "checkpoint", "gpu_cpu"],
}


TELAS["testar"] = {
    "titulo": "Testar o classificador",
    "resumo":
        "Um campo de texto para colar uma notícia e ver o que o classificador"
        " diz, antes de qualquer LLM. Não cria OCORRÊNCIA, mas grava sim um"
        " registro: a entrada fica salva com origem 'playground', e a"
        " classificação entra na auditoria — um erro visto aqui vale para o"
        " próximo treino tanto quanto um de produção.",
    "secoes": [
        ("Classificar",
         "Cola-se o texto e o modelo responde. Aparecem TODOS os indicadores,"
         " não só os marcados, com a nota de cada um e com a nota de corte que"
         " ele precisava vencer. Ver os não marcados é tão útil quanto ver os"
         " marcados: um indicador que ficou em 0,28 contra um corte de 0,30"
         " quase entrou, e isso é informação."),
        ("Limiar manual",
         "O campo de limiar troca todos os cortes por um número só, para ver o"
         " que apareceria ou desapareceria com um corte mais frouxo ou mais"
         " rígido. Não altera nada no modelo: é só a leitura daquela resposta."),
        ("O que a LLM receberia",
         "Mostra exatamente os blocos de prompt que entrariam na chamada com"
         " aquela marcação. É onde se vê, na prática, o efeito de o BERT"
         " estreitar o prompt: com duas classes marcadas viajam dois blocos,"
         " sem modelo treinado viajam todos."),
        ("O modelo errou?",
         "Leva o texto direto para o corpus, já com o link, para ser rotulado."
         " É o caminho mais curto entre perceber um erro e transformá-lo em"
         " exemplo de treino."),
    ],
    "construcao":
        "O valor desta tela é separar as duas camadas. No fluxo normal, o LLM"
        " corrige o classificador e o resultado final fica bom — o que torna"
        " impossível saber se o classificador está ajudando ou se o LLM está"
        " fazendo tudo sozinho. Aqui se vê só o classificador, cru.",
    "termos": ["bert", "llm", "limiar", "calibracao", "prompt", "skill",
               "indicador", "corpus", "token", "multirrotulo"],
}


TELAS["gestao"] = {
    "titulo": "Gestão de indicadores e skills",
    "resumo":
        "O catálogo de indicadores: onde se cria, edita e remove o que o"
        " sistema procura — sem mexer no código.",
    "secoes": [
        ("Indicadores cadastrados",
         "A lista do catálogo, com a definição, quem decide cada um, as"
         " circunstâncias que ele destrava e o prompt que viaja com ele. É"
         " esta lista que define as classes do classificador e o escopo do"
         " sistema."),
        ("Nome (slug) e rótulo",
         "O NOME é o identificador técnico, sem acento nem espaço"
         " (`mulheres_baleadas`). O RÓTULO é o texto que aparece nas telas"
         " ('Mulheres baleadas'). Separar os dois permite corrigir o texto"
         " bonito sem quebrar nada — e o nome é o que fica gravado junto com"
         " os pesos do modelo treinado."),
        ("Decidido por",
         "De onde o indicador sai. BERT: o classificador lê o texto e marca."
         " LLM: é um juízo de leitura que só o LLM faz. REGRA: o sistema"
         " calcula a partir da lista de vítimas. Esta escolha tem uma"
         " consequência que não é óbvia — indicador que só existe como REGRA"
         " não consegue controlar se o LLM é chamado, porque a regra roda"
         " depois do LLM e essa decisão acontece antes."),
        ("Skill",
         "O parágrafo de instrução que viaja no prompt quando este indicador"
         " é marcado. É aqui que se escreve o critério: o que caracteriza, o"
         " que NÃO caracteriza, e o que fazer na dúvida."),
        ("Circunstâncias que a classe destrava",
         "Algumas circunstâncias da vítima só podem ser avaliadas quando a"
         " classe correspondente foi marcada. Feminicídio só é avaliado se"
         " houver mulher baleada; sem a marcação, a instrução nem viaja no"
         " prompt e uma marcação que o LLM devolva assim mesmo é descartada."),
        ("Regra",
         "Para os indicadores calculados: a operação, o campo da vítima, o"
         " valor procurado e o mínimo. As exceções NÃO são editáveis aqui —"
         " elas moram no código. A chacina é o exemplo: a tela mostra que ela"
         " conta mortos a partir de três, mas a exceção do agente de"
         " segurança em serviço está em `indicadores.py`."),
        ("Saídas (tabelas complementares)",
         "O formato, em JSON, dos campos extras que este indicador exige do"
         " LLM: corporação e patente para o agente, partido e cargo para o"
         " político. JSON inválido ABORTA o salvamento, para que um erro de"
         " digitação não apague as tabelas."),
        ("Prompt enviado à LLM",
         "Mostra a SKILL crua deste indicador, como ela está gravada — e não o"
         " bloco montado que de fato viaja, que inclui as listas fechadas e as"
         " tabelas complementares. Serve para conferir o texto escrito aqui;"
         " para ver a chamada inteira, use a pré-visualização em /testar."),
    ],
    "construcao":
        "Criar um indicador aqui funciona, mas não basta. Duas coisas não"
        " acontecem sozinhas, e as duas eram silenciosas antes de serem"
        " tratadas: o modelo já treinado NÃO conhece a classe nova (ele lê a"
        " lista de classes do arquivo que ele mesmo gravou), e todo o corpus"
        " que já existe passa a contar como exemplo NEGATIVO dela, porque"
        " aquelas notícias nunca foram lidas com a pergunta nova. Hoje as duas"
        " geram aviso — em Modelos e no Corpus."
        "\n\n"
        "O formulário escreve SOBRE o indicador que já existe em vez de montar"
        " um dicionário novo. Era um defeito real: montar do zero apagava"
        " calado os campos que a tela não envia, e o indicador saía da lista"
        " de classes do classificador sem ninguém perceber.",
    "termos": ["indicador", "slug", "decidido_por", "skill", "prompt", "bert",
               "llm", "fenomeno_circunstancia", "chacina", "json", "corpus",
               "vitimizacao_policial"],
}


TELAS["locais_e_listas"] = {
    "titulo": "Locais e listas",
    "resumo":
        "Onde se mantêm as listas fechadas do sistema: estados, cidades,"
        " bairros, e as opções de agente de segurança e de político.",
    "secoes": [
        ("Estados, cidades e bairros",
         "Encadeados: escolher o estado carrega as cidades dele, escolher a"
         " cidade carrega os bairros. Vêm 27 estados e os 5.571 municípios"
         " brasileiros; os bairros são cadastrados conforme aparecem, porque"
         " não existe base pública nacional de bairros — o IBGE não publica"
         " essa camada."),
        ("Incluir e apagar",
         "A inclusão compara ignorando acento e maiúscula, então 'Tijuca' não"
         " entra duas vezes por causa de um til. Apagar uma cidade apaga os"
         " bairros dela junto, porque bairro sem cidade não identifica lugar"
         " nenhum."),
        ("Listas de agente e de político",
         "Sete listas editáveis: corporações, cargos de agente, situação de"
         " serviço, cargos de político, situação do político, e os contextos"
         " de ação policial e de crime organizado. São as opções que aparecem"
         " nos campos da vítima e que o LLM é obrigado a usar — ele não pode"
         " devolver valor fora delas."),
        ("O que fica gravado",
         "Só a DIFERENÇA em relação ao padrão do código. Uma lista que você"
         " não mexeu continua vindo do programa, e uma melhoria futura no"
         " padrão chega sem apagar o que você escreveu."),
    ],
    "construcao":
        "As listas fechadas não são burocracia: são o que torna a estatística"
        " possível. Texto livre no campo de bairro produz 'Tijuca', 'tijuca'"
        " e 'Tijuça' como três lugares diferentes, o que quebra tanto a"
        " contagem por bairro quanto a detecção de que dois registros são o"
        " mesmo caso."
        "\n\n"
        "Serem editáveis AQUI, e não no código, é a outra metade da decisão:"
        " uma corporação nova ou um cargo que mudou de nome não podem exigir"
        " um programador.",
    "termos": ["crud", "slug", "llm", "corporacoes", "vitimizacao_policial"],
}


TELAS["editar_prompts"] = {
    "titulo": "Prompts da LLM",
    "resumo":
        "O texto das instruções enviadas ao LLM, visível e editável fora do"
        " código.",
    "secoes": [
        ("Como a chamada é montada",
         "A chamada é a soma de cinco partes: as seções de papel e recorte"
         " (geradas de um arquivo), o prompt mestre, o formato de saída, o"
         " cabeçalho da tarefa de classes, e a skill de cada indicador"
         " marcado. As quatro primeiras vão sempre; a quinta só quando o"
         " classificador marcou aquele indicador."),
        ("Prompt mestre",
         "A instrução que manda o LLM preencher a ocorrência inteira: a lista"
         " de vítimas e, em cada uma, os campos e as listas fechadas que ele"
         " pode usar. É a maior peça e a que vai em toda chamada."),
        ("Formato de saída",
         "O desenho do JSON que o LLM tem de devolver. É cheio de chaves"
         " literais, e por isso os marcadores do sistema são trocados por"
         " substituição direta e não pelo mecanismo padrão do Python — que"
         " confundiria as chaves do JSON com marcadores."),
        ("Os três cabeçalhos",
         "Cobrem as três situações possíveis: o classificador marcou classes,"
         " o classificador leu e não marcou nenhuma, e não há modelo treinado."
         " A terceira é a chamada mais cara, porque nela viajam os blocos de"
         " todas as classes."),
        ("Restaurar padrão",
         "Cada bloco tem um botão que o devolve ao texto original. Prompt"
         " vazio não é salvo: deixaria o LLM sem a instrução, e o sistema"
         " recusa com uma mensagem em vez de gravar o vazio."),
    ],
    "construcao":
        "Nenhum texto que vai ao LLM está escondido no código. A razão é"
        " prática: instrução de modelo presa numa constante de programa é"
        " instrução que ninguém revisa, e a qualidade do resultado depende"
        " diretamente dela."
        "\n\n"
        "O arquivo de personalização guarda SÓ o que difere do padrão. Assim"
        " um bloco que você não editou continua vindo do código, e uma"
        " melhoria futura no padrão chega até você sem apagar o seu texto."
        "\n\n"
        "O tamanho importa de verdade: o teto da conta é de 8.000 tokens por"
        " minuto. É por isso que o prompt é montado em pedaços e só os"
        " pedaços necessários viajam — com o catálogo inteiro são cerca de"
        " 6.750 tokens só de instrução, e com duas classes marcadas, 4.620.",
    "termos": ["llm", "prompt", "skill", "token", "tpm", "json", "indicador",
               "bert", "slug"],
}


# --------------------------------------------------------------------------
# Resolução e conferência
# --------------------------------------------------------------------------

# Rotas que de propósito NÃO têm ajuda: nenhuma delas desenha uma tela. Três
# devolvem arquivo para baixar, duas devolvem dados para os campos encadeados
# de endereço, e duas são atalhos que redirecionam para outra tela — a ajuda
# que vale ali é a da tela de destino.
SEM_TELA = {
    "exportar", "exportar_vitimas", "corpus_exportar",
    "cidades_do_estado", "bairros_da_cidade",
    "acompanhamento", "corpus_anotar",
}


def para(endpoint: str | None) -> dict | None:
    """A ajuda da tela, com os termos do glossário já resolvidos.

    Devolve None quando não há ajuda para aquela rota, e aí o `base.html`
    simplesmente não desenha o botão. É melhor não ter botão do que ter um
    botão que abre um painel vazio.
    """
    tela = TELAS.get(endpoint or "")
    if not tela:
        return None
    return {**tela, "glossario": termos(tela.get("termos"))}


def conferir(endpoints) -> list[str]:
    """Os problemas do conteúdo de ajuda, para o script de auditoria.

    `endpoints` são as rotas que DESENHAM TELA — as de GET, sem a `static`.
    Passar as de POST junto encheria o relatório de falsos achados: `aprovar`,
    `excluir_vitima` e as outras trinta recebem um formulário e redirecionam,
    nunca mostram página, e ajuda para elas não teria onde aparecer.

    Confere três coisas que um texto escrito à mão erra com o tempo: rota sem
    ajuda, ajuda apontando para rota que não existe mais, e termo citado numa
    tela que não está no glossário. A terceira é a mais comum e a mais chata —
    ela faz a tela prometer uma explicação que não aparece.
    """
    achados = []
    rotas = set(endpoints)

    for nome in sorted(rotas - set(TELAS) - SEM_TELA):
        achados.append(f"a rota '{nome}' não tem ajuda em TELAS")
    for nome in sorted(set(TELAS) - rotas):
        achados.append(f"TELAS tem '{nome}', que não é rota")
    for nome in sorted(set(TELAS) & SEM_TELA):
        achados.append(f"'{nome}' está em TELAS e em SEM_TELA ao mesmo tempo")

    usados = set()
    for nome, tela in sorted(TELAS.items()):
        for campo in ("titulo", "resumo", "construcao"):
            if not (tela.get(campo) or "").strip():
                achados.append(f"ajuda de '{nome}' sem '{campo}'")
        if not tela.get("secoes"):
            achados.append(f"ajuda de '{nome}' sem nenhuma seção")
        for parte in tela.get("secoes") or []:
            if len(parte) != 2 or not parte[0].strip() or not parte[1].strip():
                achados.append(f"seção malformada em '{nome}': {parte[:1]}")
        for chave in tela.get("termos") or []:
            usados.add(chave)
            if chave not in GLOSSARIO:
                achados.append(
                    f"ajuda de '{nome}' cita o termo '{chave}',"
                    " que não está no GLOSSARIO"
                )

    # Termo no glossário que nenhuma tela usa: não é erro, é peso morto — a
    # pessoa nunca chega até ele, porque o painel só mostra os da tela aberta.
    for chave in sorted(set(GLOSSARIO) - usados):
        achados.append(f"o termo '{chave}' está no glossário e nenhuma tela o usa")

    return achados
