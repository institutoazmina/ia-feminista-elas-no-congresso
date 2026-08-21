# QuitérIA - A IA Feminista do Elas no Congresso

> **Análise feminista do Legislativo: uma IA open-source para classificar projetos de lei sobre gênero e direitos das mulheres.**

---

## 📌 Sumário

- [👁 Visão Geral](#-visão-geral)
- [⚙️ Como Funciona](#-como-funciona)
- [🔄 Dependência de Coleta de Dados](#-dependência-de-coleta-de-dados)
- [📊 Dados de Treinamento](#-dados-de-treinamento)
- [🧪 Etapas do Projeto](#-etapas-do-projeto)
- [📂 Modelo](#-modelo)
  - [Modelo de Classificação de Temas](#modelo-de-classificação-de-temas)
  - [Definição dos Temas](#definição-dos-temas)
  - [Avaliação dos Modelos por Tema](#avaliação)
- [Modelo de Avaliação das Posições dos Projetos](#modelo-de-avaliação-das-posições-dos-projetos)
- [📚 Trabalhos Prévios](#-trabalhos-prévios)
- [🎖️ Conheça Maria Quitéria, nossa homenageada](#-conheça-maria-quitéria-nossa-homenageada)
- [🛠 Versões](#-versões)
- [📄 Licença e Contribuições](#-licença-e-contribuições)

---

## 👁 Visão Geral

A **QuitérIA do Elas no Congresso** é um **projeto de inteligência artificial criado para analisar e rotular automaticamente proposições legislativas (PL)**, com foco em gênero e feminismo. A ferramenta classifica os temas e atribui uma nota que indica o quão desfavoráveis são os PLs aos direitos de gênero.

A solução avaliou a performance de diversos modelos de linguagem pré-treinados e ajustados para classificar os projetos. Este repositório concentra as informações sobre a pesquisa e desenvolvimento de soluções para aperfeiçoar o processo de rotulagem de dados das PL por meio de técnicas de PLN (Processamento de Linguagem Natural) e a coleta automatizada de dados no site da Câmara dos Deputados e do Senado Federal.

---

## ⚙️ Como Funciona

Como parte de seu monitoramento legislativo, a equipe do [Instituto AzMina](https://institutoazmina.org.br/) usa o [Bot do Elas no Congresso](https://github.com/institutoazmina/elasnocongressobot) para coletar as proposições de interesse por meio de palavras-chave. Mas até a edição de 2024, as propostas eram revisadas e rotuladas pela equipe e instituições parceiras com dois objetivos:

- **Tema**: definir o tema da proposta a partir do resumo (ementa) do PL, designando uma das categorias definidas pela contratante;
- **Avaliação**: definir se a proposta é favorável ou não ao direito das mulheres, a partir de uma análise da ementa e do inteiro teor do PL.

Devido ao alto número de propostas coletadas, a rotulagem completamente manual é custosa e demanda alto investimento de recursos humanos. Modelos de inteligência artificial supervisionados poderiam facilitar o processo de rotulagem inicial dos dados.

Assim, a QuitérIA surge com o objetivo de **facilitar a análise e o acompanhamento de projetos de lei** relacionados a temas de gênero, fornecendo **insights automatizados** que auxiliem na tomada de decisão, defesa de políticas públicas e divulgação de informações relevantes.

Implementada em 2025, a classificação automática apresentada aqui **passa por validação humana**, com apoio das mesmas organizações parceiras que já colaboravam no processo anterior. A diferença é que agora as equipes ganham mais tempo para atuar politicamente e produzir conteúdo no site do [Elas no Congresso](https://elasnocongresso.com.br).

---

## 🔄 Dependência de Coleta de Dados

Os dados utilizados para treinar e alimentar os modelos da QuitérIA são coletados automaticamente pelo robô do projeto [Elas no Congresso Bot](https://github.com/institutoazmina/elasnocongressobot).

Este scraper busca proposições legislativas nos sites da Câmara e do Senado, usando palavras-chave relacionadas a gênero, mulheres, direitos feministas e comunidades LGBTQIAP+. As informações extraídas (como título, autor, ementa, status etc.) são salvas em arquivos CSV e posteriormente utilizadas no treinamento e classificação dos modelos de IA.

O código-fonte do robô de coleta está disponível em:  
🔗 https://github.com/institutoazmina/elasnocongressobot

---

## 📊 Dados de Treinamento

Os dados utilizados para o treinamento dos modelos consistem em **resumos de proposições legislativas** em tramitação no Congresso Nacional. Esses dados foram **pré-processados e rotulados manualmente** durante quatro anos e meio, servindo de base para o fine-tuning dos classificadores.

Os dados rotulados estão disponíveis na API: [`api.elasnocongresso.com.br/api/v1/docs`](https://api.elasnocongresso.com.br/api/v1/docs)

---

## 🧪 Etapas do Projeto

### **Pré-processamento de dados**: scripts de limpeza e preparação das proposições legislativas;

#### Preparação

Para treinamento, consideramos apenas projetos distintos, deduplicando quando ha `id`s de projetos repetidos, utilizando apenas o principal autor do projeto.

#### Particionamento da base

A partir dos dados completos, realizados uma amostragem aleatória estratificada considerando a variável `fl_desfavorável`.

| Base | Quantidade de Projetos (%) |
|---|---:|
| Treino | 2.527 (80,94%) |
| Validação | 282 (09,03%) |
| Teste | 313 (10,02%) |


#### **Estratégia de Fine-tuning**

Para treinamento do modelo, consideramos o modelo `neuralmind/bert-base-portuguese-cased` para ambos os desafios de inferência. No lugar de ajustar apenas uma vez o modelo e avaliá-lo, foi realizada uma interação (loop) de treino com early-stop de 6 steps em caso de não melhoria da métrica objetivo. Assim, este melhor modelo é passado para a próxima iteração repetindo o processo.

3. **Avaliação**: desempenho dos modelos medido por métricas como:
   - Acurácia
   - F1-Score
   - Precisão
   - Recall

Vale considerar que a comparação dos resultados entre as difrentes versões do modelo entre os não é válida, uma vez que temos uma composição das bases diferente. Tanto por questão de novos dados (novos projetos de lei avaliados), como também, um novo sorteio aleatório utilizando extratificação.

Ainda assim, nossa expectativa é aumentar os valores obtidos anteriormente nas métricas de ajuste.

---

## 📂 Modelos

### Modelo de Classificação de Temas

Os temas utilizados para classificação dos projetos de lei são:

```json
{
  0: "Direitos_Sexuais_e_Reprodutivos",
  1: "Educação_e_Cultura",
  2: "Família_Parentalidade_e_Relações_Civis",
  3: "Igualdade_e_Antidiscriminação",
  4: "Infância_e_Adolescência",
  5: "LGBTQIAPN",
  6: "Participação_Política_e_Institucionalidade",
  7: "Saúde",
  8: "Trabalho_Economia_Cuidado_e_Proteção_Social",
  9: "Violências_de_Gênero"
}
```

#### Definição dos temas:

**Direitos Sexuais e Reprodutivos**: Abrange proposições sobre autonomia corporal, direitos reprodutivos e sexuais, acesso à contracepção, aborto, planejamento familiar, reprodução assistida e garantia de direitos ligados à sexualidade.

**Educação e Cultura**: Abrange projetos relacionados à educação formal e não formal, produção cultural, liberdade de ensino, conteúdos curriculares, memória e preservação histórica, homenagens e reconhecimento de personalidades ou marcos relevantes para a promoção da igualdade de gênero, da diversidade e dos direitos humanos. 

**Família, Parentalidade e Relações Civis**: Engloba proposições sobre organização familiar, casamento, união estável, divórcio, filiação, guarda, adoção, poder familiar, registro civil e demais direitos e deveres nas relações familiares.

**Igualdade e Antidiscriminação**: Contempla iniciativas destinadas a promover a igualdade de direitos e oportunidades e a combater discriminações baseadas em gênero, sexo, orientação sexual, identidade de gênero, raça, território e outras formas de desigualdade.

**Infância e Adolescência**: Reúne proposições relacionadas à promoção, proteção e garantia dos direitos de crianças e adolescentes, com atenção especial aos impactos sobre meninas e adolescentes em perspectiva de gênero. Inclui temas como proteção integral, convivência familiar e comunitária, desenvolvimento infantil, primeira infância, violência contra crianças e adolescentes, trabalho infantil, acolhimento institucional, medidas socioeducativas, ambiente digital, proteção de dados, uso de tecnologias, participação social, acesso a direitos e políticas públicas voltadas à infância e à adolescência.

**LGBTQIAPN+**: Reúne propostas que tratam dos direitos, da cidadania, da proteção e do reconhecimento das pessoas LGBTQIAPN+, incluindo identidade de gênero, orientação sexual, acesso a direitos e combate à violência e à discriminação.

**Participação Política e Institucionalidade**: Reúne iniciativas que tratam da participação de mulheres e da população LGBTQIAPN+ nos espaços de poder, representação política, cargos e funcionamento das instituições públicas, mecanismos de participação social e políticas públicas para promoção da igualdade de gênero.

**Saúde**: Inclui projetos voltados à promoção, prevenção e acesso à saúde, com atenção às necessidades específicas de mulheres, pessoas gestantes, pessoas LGBTQIAPN+ e outros grupos em situação de vulnerabilidade. Abrange temas como saúde materna, menstruação e dignidade menstrual, menopausa, climatério, cânceres relacionados ao aparelho reprodutor, saúde mental, acesso a medicamentos, prevenção de doenças e organização dos serviços de saúde.

**Trabalho, Economia, Cuidado e Proteção Social**: Inclui propostas relacionadas ao mercado de trabalho, economia, geração de renda, empreendedorismo, previdência, seguridade e assistência social, políticas de proteção econômica e social, maternidade, licença-maternidade e outros direitos trabalhistas, acesso a creches e políticas de cuidado, valorização do trabalho doméstico e de cuidados, regulamentação ou direitos relacionados à prostituição, bem como iniciativas que abordem os impactos das mudanças climáticas e das políticas ambientais sobre as condições de vida, o trabalho, o cuidado e a proteção social. 

**Violências de Gênero**: Reúne propostas relacionadas à prevenção, enfrentamento, responsabilização e reparação das diversas formas de violência baseadas em gênero, incluindo assédio, violência doméstica, sexual, política, institucional, obstétrica, digital e feminicídio, além de proposições específicas sobre acesso a armas ou que pretendem realizar alterações na Lei Maria da Penha.

#### Treinamento

**Modelo:** neuralmind/bert-base-portuguese-cased

Foi realizado um treinamento sequencial de 65 modelos com os mesmos hiperparâmetros. Após cada interação de treino, o modelo é salvo no `MLFlow` e utilizado na próxima iteração de treinamento com os mesmos hiperparâmetros e estratégia de treino.

Selecionamos o modelo campeão como aquele que obteve melhor métrica de F1 Macro na base de testes. Repare que não necessariamente o modelo campeão é o da última interação.

Na verdade, no nosso caso, o modelo campeão é o resultado da interação 35 no nosso caso.

No lugar de deixar o modelo ser treinado em muitas épocas independente da melhoria de performance, colocamos um critério de parada caso as métricas não melhorem em 6 passos. Assim, na próxima interação, aproveitamos os pesos encontrado para fazer um novo treinamento, com a expectativa de melhoria gradual do modelo.


#### Avaliação

O modelo campeão obteve as seguintes métricas na base de `teste`.

| Métrica | Valor |
|---|---|
| Acurácia| 0,82 |
| F1 Macro| 0,71|
| Precisão Macro| 0,73|
| Recall Macro| 0,72|
| F1 Direitos Sexuais e Reprodutivos| 0,72|
| F1 Educação e Cultura| 0,73|
| F1 Família Parentalidade e Relações Civis| 0,73|
| F1 Igualdade e Antidiscriminação| 0,67|
| F1 Infância e Adolescência| 0,31|
| F1 LGBTQIAPN| 0,76|
| F1 Participação Política e Institucionalidade| 0,59|
| F1 Saúde| 0,84|
| F1 Trabalho Economia Cuidado e Proteção Social| 0,81|
| F1 Violências de Gênero| 0,91 |

Nota-se a baixa performance em F1 para as categorias `Infância e Adolescência`, `Participação Política e Institucionalidade` e `Igualdade e Antidiscriminação`. Sendo estas as categorias com menor quantidade de amostras.

---

### Modelo de avaliação das posições dos projetos

O segundo modelo tem como objetivo identificar propostas desfavoráveis aos direitos das mulheres pela perspectiva feminista interseccional, a partir de um conjunto de dados previamente rotulado pela equipe AzMina e organizações parceiras. O modelo foi treinado a fim de otimizar a identificação de projetos da classe positiva (desfavoráveis aos direitos das mulheres)

Este modelo classifica PLs como:

- **Classe 0 (Favorável)**: Promovem direitos das mulheres, igualdade de gênero, garantias legais.
- **Classe 1 (Desfavorável)**: Representam retrocessos, ameaçam políticas públicas ou ampliam desigualdades.

#### Treinamento

**Modelo:** neuralmind/bert-base-portuguese-cased

A estratégia adotada de treinamento foi análoga ao modelo de temas. Realizamos um treinamento sequencial de 100 modelos com os mesmos hiperparâmetros. Após cada interação de treino, o modelo é salvo no `MLFlow` e utilizado na próxima iteração de treinamento com os mesmos hiperparâmetros e estratégia de treino.

Selecionamos o modelo campeão como aquele que obteve melhor métrica de F1 na base de testes.

#### Avaliação

O modelo campeão atual possui:
- F1-score de **0.92** para Classe 0
- F1-score de **0.67** para Classe 1

Importande destacar que os valores obtidos nas métricas de performance são calculado com um ponto de corte `cutoff` considerando a média da variável resposta, isto é, 0.2254. Assim, projetos com probabilidade maior que 0.2254 atribuida pelo menos, serão considerados **desfavoráveis**.

O modelo atingiu as seguintes métricas no dataset de `teste`:

|                | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Class 0        | 0.89      | 0.94   | 0.92     | 242     |
| Class 1        | 0.75      | 0.61   | 0.67     | 71      |
| Accuracy       |           |        | 0.87     | 313     |
| AUC Score      |           |        | 0.85     | 313     |


## 🎖️ Conheça Maria Quitéria, nossa homenageada
Em 1823, Maria Quitéria se vestiu de homem para lutar pela independência do Brasil, uma coragem que desafiou as expectativas de seu tempo. Dois séculos depois, QuitérIA nasce com o mesmo espírito revolucionário, ocupando os espaços digitais de poder para garantir que as leis brasileiras considerem leis sempre o recorte de gênero. 
Como sua inspiradora, que se disfarçou para acessar espaços de poder, QuitérIA penetra no intrincado sistema legislativo brasileiro, decodificando sua linguagem hermética e revelando impactos muitas vezes invisíveis ao debate público. Nossa arma contemporânea para mudar o futuro é a tecnologia, especialmente dados e algoritmos.
Num contexto onde a complexidade legislativa frequentemente funciona como barreira à participação cidadã, QuitérIA traduz, simplifica e empodera, transformando dados em ação.

## 📚 Trabalhos Prévios

A utilização de modelos de IA em documentos oficiais em português ainda é incipiente. Foi possível identificar trabalhos prévios que utilizam técnicas de processamento de linguagem natural (NLP) para textos legais, mas poucos guardam similaridade com as tarefas em questão. Não foi possível localizar nenhum trabalho prévio focado na classificação de textos legais entre favoráveis ou desfavoráveis aos direitos das mulheres, porém alguns trabalhos anteriores já utilizaram técnicas de NLP para classificação de textos legais em temas pré-determinados.

Os trabalhos listados propõem diferentes categorizações em documentos do âmbito jurídico ou legislativo. Apesar da similaridade da tarefa em questão, estes esforços não servem como um benchmark direto para o nosso projeto, uma vez que a taxonomia e os exemplos diferem daqueles usados no presente projeto. Dito isso, o trabalho que mais se aproxima ao nosso em termos comparativos é o [“A classification approach for estimating subjects of bills in the Brazilian Chamber of Deputies”](https://lume.ufrgs.br/handle/10183/267612), onde o autor experimentou classificadores para temas de proposições da Câmara dos Deputados do Brasil, comparando dois modelos BERT adaptados para o português, usando o texto da ementa. Os melhores resultados relatados foram com o modelo BERTimbau, alcançando 78,94% de pontuação F1 ponderada.

| Referência                                                                                                     | Acurácia | F1-score | Recall |
|---------------------------------------------------------------------------------------------------------------|----------|----------|--------|
| [A classification approach for estimating subjects of bills in the Brazilian Chamber of Deputies](https://lume.ufrgs.br/handle/10183/267612)               | 65%      | 73%      | 70%    |
| [ver BERT: Automating Brazilian Case Law Document Multi-label Categorization Using BERT](https://sol.sbc.org.br/index.php/stil/article/view/17803)                         | 38%      | 71%      | 66%    |
| [Classificação de documentos jurídicos utilizando a arquitetura Transformer: uma análise comparativa com algoritmos tradicionais de Machine Learning e ChatGPT](https://ojs.brazilianjournals.com.br/ojs/index.php/BRJD/article/view/60747) | 62%      | 62%      | 62%    |

## 🛠 Versões

- datasets==4.8.5
- evaluate==0.4.6
- ipython==9.14.0
- mlflow==3.12.0
- mlflow_skinny==3.12.0
- mlflow_tracing==3.12.0
- numpy==2.4.6
- openpyxl==3.1.5
- python-dotenv==1.2.2
- scikit_learn==1.9.0
- seaborn==0.13.2
- torch==2.12.0
- torchvision==0.27.0
- transformers[torch]==5.7.0

## 📄 Licença e Contribuições

Este repositório contém códigos, modelos próprios e integrações com APIs externas para execução de funcionalidades relacionadas à análise automatizada de proposições legislativas.

Isso significa que você pode:
**Compartilhar** — copiar e redistribuir o material em qualquer suporte ou formato.


**Adaptar** — remixar, transformar e criar a partir do material, para qualquer fim, mesmo que comercial.


Desde que:
**Atribuição** — dê o devido crédito, insira um link para a licença e indique se mudanças foram feitas.


**Compartilhar Igual** — se remixar, transformar ou criar a partir do material, deve distribuir suas contribuições sob a mesma licença.

Além disso, esse projeto segue as diretrizes da Responsible AI License (RAIL) para reforçar práticas éticas no desenvolvimento e uso de IA. Consulte RAIL License.
Ao utilizar este repositório, você **concorda em não empregar o código ou modelos para fins ilegais, discriminatórios, violentos ou que violem direitos humanos**.

**Em resumo:**
Código + modelos próprios → **CC BY-SA 4.0 + RAIL**


Modelos externos → **licença do provedor**, não redistribuídos aqui

### Contribua
Correções, melhorias ou ideias são muito bem-vindas!  
Você pode contribuir diretamente pelo GitHub abrindo uma [issue](https://github.com/institutoazmina/ia-feminista-elas-no-congresso/issues) ou enviando um [pull request](https://github.com/institutoazmina/ia-feminista-elas-no-congresso/pulls).


Ou então, fale com a gente: tecnologia@azmina.com.br 

Vamos construir uma tecnologia feminista e transformadora? 💜
