# Guia do Projeto para Iniciantes

## O que o projeto faz

Este projeto é um sistema de apoio à triagem de câncer de mama. Ele recebe medidas numéricas extraídas de exames e classifica o registro em uma das duas classes presentes no dataset:

- `B`: benigno.
- `M`: maligno.

O sistema não fornece diagnóstico médico. Ele encontra padrões estatísticos aprendidos em dados históricos e deve ser usado apenas como apoio à decisão de profissionais de saúde.

```text
Exames com diagnóstico conhecido
              ↓
Aprendizado de padrões pelo modelo
              ↓
Novo exame
              ↓
Estimativa de benigno ou maligno
```

## Motivação

O Tech Challenge pede uma solução de Machine Learning relacionada à saúde e segurança da mulher. O câncer de mama é um problema adequado para a etapa inicial porque permite trabalhar com uma base pública, dados estruturados, classificação supervisionada, métricas e explicabilidade.

O objetivo é apoiar a triagem. Em saúde, deixar de sinalizar um caso maligno pode atrasar a investigação clínica. Por isso, o projeto dá prioridade ao recall da classe maligna.

## Por que não usamos banco vetorial

Banco vetorial é uma tecnologia usada principalmente para procurar textos ou documentos parecidos por significado. Ele é útil, por exemplo, para buscar prontuários textuais semelhantes ou criar um chatbot que consulta documentos.

O problema atual é diferente: a entrada é uma tabela de números e a saída é uma classe. A arquitetura necessária é:

```text
CSV → pré-processamento → modelo de Machine Learning → resultado
```

Um banco vetorial seria uma extensão futura para análise de textos, mas não melhora a classificação estruturada desta fase.

## Dataset

O projeto usa o Breast Cancer Wisconsin Diagnostic Dataset, sugerido pelo enunciado. Ele possui 569 registros, 30 atributos numéricos e uma variável-alvo chamada `diagnosis`.

Os atributos representam medidas relacionadas a características celulares, como raio, textura, perímetro, área e concavidade. A classe `B` aparece 357 vezes e a classe `M`, 212 vezes.

## O que é Machine Learning

Machine Learning é uma forma de ensinar um sistema por meio de exemplos. Em vez de escrever manualmente todas as regras, mostramos registros antigos com a resposta correta:

```text
Medidas do exame → diagnóstico conhecido
```

O algoritmo identifica relações estatísticas e depois tenta prever a classe de um novo registro. Ele não entende medicina como uma pessoa e não deve tomar decisões clínicas autônomas.

## Arquitetura do projeto

O projeto usa Clean Architecture para separar as regras de negócio das ferramentas técnicas.

```text
presentation → application → domain
infrastructure → application → domain
```

### Domain

Contém os conceitos centrais do negócio, sem depender de bibliotecas externas:

- Diagnóstico benigno ou maligno.
- Dataset.
- Métricas.
- Matriz de confusão.
- Artefato de modelo.

### Application

Contém os casos de uso:

- Treinar modelos.
- Selecionar o melhor modelo.
- Analisar o dataset.
- Explicar o modelo selecionado.

### Infrastructure

Contém integrações com bibliotecas e arquivos:

- Leitura de CSV com `pandas`.
- Treino com `scikit-learn`.
- Persistência com `joblib`.
- Gráficos com `matplotlib` e `seaborn`.
- Explicabilidade com `SHAP`.

### Presentation

É a interface atual por comandos de terminal. Uma interface Streamlit pode ser incluída futuramente sem alterar o núcleo do sistema.

## Pré-processamento dos dados

Antes do treinamento, os dados precisam ser preparados.

### Identificadores

Colunas como `id` são removidas. Elas identificam registros, mas não possuem valor clínico para prever malignidade.

### Valores ausentes

Quando um valor numérico está ausente, o projeto usa a mediana da variável. A mediana é o valor central de uma lista ordenada e costuma ser menos afetada por valores extremos que a média.

### Padronização

A Regressão Logística usa padronização para colocar as variáveis em escalas comparáveis. Isso evita que atributos com números naturalmente maiores pareçam importantes apenas pela unidade usada.

### Vazamento de dados

Vazamento de dados acontece quando informações do conjunto de teste são usadas durante o treino. Seria parecido com estudar usando o gabarito da prova.

O projeto evita isso porque a imputação e a padronização são aprendidas apenas com os dados de treino, dentro de um `Pipeline` do scikit-learn.

## EDA: análise exploratória

EDA significa Exploratory Data Analysis, ou Análise Exploratória de Dados. É a etapa em que entendemos os dados antes de criar modelos.

O projeto gera:

- Estatísticas descritivas.
- Distribuição das classes.
- Histogramas dos atributos por diagnóstico.
- Matriz de correlação.
- Tabela completa de correlações.

Não foram encontrados valores ausentes ou linhas duplicadas na versão do dataset utilizada.

### Correlação

Correlação mostra se duas variáveis variam juntas. Ela vai de `-1` até `1`:

- `1`: relação positiva forte.
- `0`: ausência de relação linear clara.
- `-1`: relação negativa forte.

Por exemplo, `radius_mean` e `perimeter_mean` têm correlação aproximada de 0,998. Isso é esperado porque estruturas maiores tendem a ter perímetros maiores.

Correlação não prova causalidade.

## Treino, teste e validação cruzada

Os dados são divididos de forma estratificada:

```text
80% para treino
20% para teste
```

A estratificação preserva aproximadamente a proporção de casos benignos e malignos em cada grupo.

O conjunto de treino passa por validação cruzada estratificada de três dobras:

```text
Rodada 1: treina em A+B e valida em C
Rodada 2: treina em A+C e valida em B
Rodada 3: treina em B+C e valida em A
```

O modelo é escolhido pela média da validação cruzada. O conjunto de teste fica isolado e é utilizado uma única vez, na avaliação final.

```text
Treino → validação cruzada → escolha do modelo
Teste  → avaliação final única
```

## Fluxo completo de processamento e treinamento

Esta é a sequência completa executada pelo projeto. Ela é importante porque deixa claro onde cada decisão acontece e impede que o modelo seja avaliado de forma injusta.

```text
1. Arquivo CSV público
        ↓
2. Leitura e validação das colunas
        ↓
3. Remoção de identificadores e conversão para números
        ↓
4. EDA: estatísticas, gráficos e correlação
        ↓
5. Divisão estratificada em treino e teste
        ↓
6. Validação cruzada no treino
        ↓
7. Treino dos modelos candidatos
        ↓
8. Escolha pelo desempenho de validação
        ↓
9. Avaliação única no teste isolado
        ↓
10. Salvamento do modelo selecionado
        ↓
11. Matriz de confusão, importância por permutação e SHAP
```

### 1. Leitura do arquivo

O processo começa no arquivo `data/raw/breast_cancer.csv`. O adaptador `CsvDatasetRepository` abre esse CSV e verifica se a coluna `diagnosis` existe.

Essa coluna é obrigatória porque contém a resposta conhecida para cada registro:

```text
B = benigno
M = maligno
```

Se a coluna não existir, o sistema interrompe a execução em vez de treinar com dados incorretos.

### 2. Separação entre entrada e resposta

Depois da leitura, os dados são divididos em duas partes:

```text
Entradas ou features: raio, textura, área, concavidade e outras medidas
Resposta ou target: diagnosis
```

Em Machine Learning, as entradas são aquilo que o modelo observa. A resposta é aquilo que ele deve aprender a prever.

As colunas `id` e `Unnamed: 32` são ignoradas. Elas não descrevem o exame e poderiam induzir o modelo a aprender padrões sem valor clínico.

### 3. Preparação numérica

Cada medida é convertida para número. Caso exista uma célula vazia em uma feature, ela não é preenchida imediatamente usando todos os dados.

O preenchimento ocorre dentro do pipeline de cada modelo. Isso é essencial porque o pipeline aprende a mediana usando somente os dados de treino.

```text
Dados de treino → calcula mediana
Dados de teste  → recebe a mediana aprendida no treino
```

Assim, o teste não influencia nenhuma decisão do treinamento.

### 4. Análise exploratória antes do modelo

O comando `analyze` cria artefatos que ajudam a entender o dataset antes de treinar:

```bash
iadt-ml analyze --dataset data/raw/breast_cancer.csv
```

Ele produz:

- `descriptive_statistics.csv`: média, desvio padrão, mínimo, máximo e quartis.
- `class_distribution.png`: quantidade de benignos e malignos.
- `feature_distributions.png`: como cada medida se distribui por classe.
- `correlation_heatmap.png`: relações lineares entre as features.
- `feature_correlations.csv`: valores numéricos de correlação.

Essa etapa não treina nem escolhe modelos. Ela serve para compreender a qualidade, a escala, o balanceamento e possíveis relações dos dados.

### 5. Criação dos conjuntos de treino e teste

O `SklearnDatasetPartitioner` faz uma divisão estratificada com uma semente fixa (`42`). A semente é um número usado para tornar a divisão reproduzível: pessoas diferentes executando o mesmo projeto obtêm a mesma separação.

```text
569 registros
    ↓
455 aproximadamente para treino
114 aproximadamente para teste
```

A estratificação garante que as duas partes preservem a proporção de diagnósticos benignos e malignos.

### 6. Validação cruzada no conjunto de treino

O teste ainda não é usado. O conjunto de treino é dividido em três partes para a validação cruzada.

Para cada modelo, o processo acontece assim:

```text
Primeira dobra:
  treino interno = dobras 1 e 2
  validação interna = dobra 3

Segunda dobra:
  treino interno = dobras 1 e 3
  validação interna = dobra 2

Terceira dobra:
  treino interno = dobras 2 e 3
  validação interna = dobra 1
```

Em cada rodada, o modelo é criado do zero, recebe os dados do treino interno e gera previsões para a validação interna. No fim, o projeto calcula a média de accuracy, recall e F1-score das três rodadas.

Essa média é a nota usada para comparar os modelos candidatos.

### 7. Como a Regressão Logística é treinada

A Regressão Logística recebe um pipeline com três etapas:

```text
1. SimpleImputer
2. StandardScaler
3. LogisticRegression
```

O `SimpleImputer` substitui valores ausentes pela mediana calculada no treino interno. O `StandardScaler` transforma cada feature para uma escala comparável. Por fim, a Regressão Logística calcula uma combinação matemática dos atributos e retorna uma probabilidade de malignidade.

De forma simplificada:

```text
probabilidade = função(textura, área, concavidade, raio, ...)
```

Se essa probabilidade ultrapassar o limiar padrão do modelo, a previsão é `M`; caso contrário, é `B`.

### 8. Como a Random Forest é treinada

A Random Forest também recebe imputação pela mediana. Depois, ela cria centenas de árvores de decisão; no projeto, são 400 árvores.

Cada árvore recebe uma amostra diferente dos dados e considera subconjuntos de features. Isso reduz a dependência de uma única árvore e torna o conjunto mais robusto.

Uma árvore individual pode seguir uma lógica simplificada como:

```text
Se textura é alta e concavidade é alta, aumente a evidência de malignidade.
```

A Random Forest reúne os votos das árvores e retorna a classe com maior evidência coletiva.

### 9. Escolha do modelo

Depois da validação cruzada, os modelos são comparados usando somente os resultados do treino.

```text
Maior recall para malignidade
        ↓
Em empate, maior F1-score
        ↓
Em novo empate, maior accuracy
```

O projeto escolhe o melhor modelo com esse critério e o treina usando todos os dados do conjunto de treino. Isso dá ao modelo escolhido acesso a mais exemplos antes da avaliação final.

### 10. Avaliação final no teste

Somente depois de escolher o modelo, o conjunto de teste isolado é usado. O modelo produz previsões e probabilidades para os 20% de registros que não participaram do treino nem da escolha.

Nessa etapa são calculados:

```text
accuracy
recall
F1-score
ROC-AUC
matriz de confusão
```

Esse é o resultado mais importante para apresentar, porque representa o comportamento esperado do modelo em dados novos semelhantes aos da base.

### 11. Salvamento do modelo

O artefato selecionado é salvo em:

```text
models/best_model.joblib
```

Ele guarda o pipeline completo, não apenas o classificador. Isso significa que o mesmo tratamento de valores ausentes e escala é aplicado automaticamente antes de qualquer previsão futura.

### 12. Explicabilidade depois da avaliação

O comando `explain` utiliza o modelo salvo e o mesmo conjunto de teste reproduzível:

```bash
iadt-ml explain --dataset data/raw/breast_cancer.csv --model models/best_model.joblib
```

Ele não muda o modelo nem escolhe outro algoritmo. Apenas mede e visualiza como o modelo já escolhido chegou aos resultados.

São produzidos:

- `confusion_matrix.png`: quantidade de acertos e erros por classe.
- `feature_importance.csv`: importância quantitativa das features.
- `feature_importance.png`: gráfico da importância por permutação.
- `shap_summary.png`: impacto global das features no modelo.

## Fluxo dentro do código

```text
CLI
  ↓
Caso de uso da camada application
  ↓
Portas definidas no domain
  ↓
Adaptadores da infrastructure
  ↓
Pandas, scikit-learn, joblib, SHAP e arquivos gerados
```

Exemplo do treinamento:

```text
iadt-ml train
  ↓
TrainModels.execute
  ↓
CsvDatasetRepository.load
  ↓
SklearnDatasetPartitioner.split
  ↓
SklearnTrainingService.train
  ↓
SklearnEvaluationService.evaluate
  ↓
JoblibModelRepository.save
```

Essa separação permite trocar ferramentas sem alterar a regra principal. Por exemplo, o CSV poderia ser substituído por um banco de dados, ou a CLI por uma tela Streamlit, sem reescrever os casos de uso centrais.

## Modelos comparados

### Regressão Logística

É um algoritmo de classificação que calcula uma probabilidade. Exemplo:

```text
Probabilidade de malignidade: 0,87
Resultado: maligno
```

Ela é um bom modelo inicial porque é rápida, estável e relativamente simples de interpretar.

### Random Forest

É um conjunto de várias árvores de decisão. Cada árvore toma decisões com base nos atributos, e a floresta combina as respostas.

Ela consegue representar relações mais complexas e não lineares entre os atributos.

## Como o melhor modelo é escolhido

A prioridade é o recall da classe maligna. Em caso de empate, são usados F1-score e accuracy.

```text
1. Maior recall
2. Maior F1-score
3. Maior accuracy
```

Essa escolha é adequada para uma situação de triagem: é preferível investigar um caso adicional a deixar de sinalizar um possível caso maligno.

## Métricas

### Accuracy

É a proporção geral de acertos.

```text
accuracy = acertos / total de registros
```

### Recall

Mede quantos casos malignos reais foram identificados corretamente.

```text
recall = verdadeiros positivos / todos os malignos reais
```

É a métrica prioritária do projeto.

### F1-score

É uma métrica que equilibra recall e precisão. A precisão mede quantos casos classificados como malignos eram realmente malignos.

### ROC-AUC

Mede a capacidade geral de separar as duas classes. Quanto mais próximo de 1, melhor a separação; um valor próximo de 0,5 é parecido com um palpite aleatório.

## Resultado atual

A Regressão Logística foi selecionada pela validação cruzada. No conjunto de teste isolado, o resultado foi:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,965 |
| Recall para malignidade | 0,929 |
| F1-score | 0,951 |
| ROC-AUC | 0,996 |

A matriz de confusão apresentou:

| Classe real / classe prevista | Benigno | Maligno |
|---|---:|---:|
| Benigno | 71 | 1 |
| Maligno | 3 | 39 |

Isso significa que 39 casos malignos foram identificados corretamente, mas 3 não foram sinalizados. Esse resultado reforça que o modelo precisa de supervisão clínica.

## Explicabilidade

### Importância por permutação

Essa técnica embaralha uma variável por vez e mede quanto o desempenho piora. Se o recall cai bastante quando uma variável é embaralhada, aquela variável é importante para o modelo.

Na execução atual, atributos como `texture_worst`, `concavity_worst`, `symmetry_worst` e `concave points_worst` se destacaram.

### SHAP

SHAP mostra como os atributos contribuíram para a saída do modelo. Ele ajuda a responder quais variáveis puxaram uma previsão para maligno ou benigno.

SHAP explica o comportamento matemático do modelo. Não demonstra causalidade médica.

## Como executar

Crie e ative o ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Baixe o dataset e salve como `data/raw/breast_cancer.csv`. Depois execute:

```bash
iadt-ml analyze --dataset data/raw/breast_cancer.csv
iadt-ml train --dataset data/raw/breast_cancer.csv
iadt-ml explain --dataset data/raw/breast_cancer.csv --model models/best_model.joblib
```

Os gráficos e tabelas são salvos em `reports/figures/`; o modelo treinado é salvo em `models/best_model.joblib`.

## Glossário

| Termo | Definição |
|---|---|
| Dataset | Conjunto de dados usado no projeto. |
| Feature | Coluna de entrada usada pelo modelo. |
| Target | Resposta que o modelo tenta prever. |
| Classificação | Previsão de categorias, como benigno ou maligno. |
| Modelo | Algoritmo que aprende padrões nos dados. |
| Treino | Dados usados para ensinar o modelo. |
| Teste | Dados isolados para avaliação final. |
| Pipeline | Sequência de tratamento dos dados e modelo. |
| Imputação | Preenchimento de valores ausentes. |
| Padronização | Colocação de variáveis em escala comparável. |
| Validação cruzada | Avaliação repetida em partes diferentes do treino. |
| Accuracy | Percentual geral de acertos. |
| Recall | Percentual de malignos reais identificados. |
| F1-score | Equilíbrio entre recall e precisão. |
| ROC-AUC | Capacidade de separar as classes. |
| Falso positivo | Caso benigno classificado como maligno. |
| Falso negativo | Caso maligno classificado como benigno. |
| Matriz de confusão | Tabela de acertos e erros por classe. |
| SHAP | Técnica de explicabilidade de previsões. |
| Clean Architecture | Organização que separa regras de negócio de ferramentas. |
| Banco vetorial | Banco para busca por similaridade semântica. |
| LGPD | Lei Geral de Proteção de Dados. |

## Limitações e responsabilidade

Os resultados são acadêmicos e não comprovam uso hospitalar. A base é pública, histórica e limitada; ela não representa necessariamente a população de uma instituição específica nem contém todo o contexto clínico necessário.

Um uso real exigiria validação clínica externa, avaliação de viés, monitoramento contínuo, governança de dados, segurança, conformidade com a LGPD e decisão final de profissionais habilitados.
