# Relatório Técnico - Fase 1

## Problema e objetivo

O projeto classifica registros de exames de câncer de mama como benignos (`B`) ou malignos (`M`) para apoiar a triagem clínica. O resultado não é um diagnóstico e não substitui a decisão de profissionais de saúde.

O objetivo de modelagem é identificar adequadamente a classe maligna. Por isso, o recall de malignidade é a métrica prioritária na seleção; F1-score e accuracy servem como critérios de desempate.

## Dados

Foi utilizado o Breast Cancer Wisconsin Diagnostic Dataset, indicado no enunciado. A base contém 569 registros, 30 atributos numéricos derivados de imagens digitalizadas de aspiração por agulha fina e a variável-alvo `diagnosis`.

A análise exploratória identificou 357 registros benignos e 212 malignos. Não há valores ausentes nem linhas duplicadas na versão utilizada. Como todos os atributos preditores são numéricos, não existe variável categórica a ser codificada neste recorte.

Os arquivos gerados estão em `reports/figures/`:

- `descriptive_statistics.csv`
- `class_distribution.png`
- `feature_distributions.png`
- `correlation_heatmap.png`
- `feature_correlations.csv`

As dimensões relacionadas a raio, perímetro e área são fortemente correlacionadas. Por exemplo, `radius_mean` e `perimeter_mean` têm correlação aproximada de 0,998. Essa redundância é esperada pela natureza geométrica das medidas e deve ser considerada na interpretação de coeficientes, sem invalidar os modelos escolhidos.

## Pré-processamento

O carregamento remove campos identificadores (`id` e coluna vazia auxiliar) e converte atributos para formato numérico. O pré-processamento pertence ao `Pipeline` do scikit-learn e é ajustado apenas com os dados de treino.

A Regressão Logística utiliza imputação pela mediana e padronização. A Random Forest utiliza imputação pela mediana. Assim, valores ausentes futuros são tratados sem vazamento de informação do teste para o treino.

## Modelagem e validação

Foram comparados dois classificadores:

- Regressão Logística: baseline linear, estável e mais simples de interpretar.
- Random Forest: modelo não linear capaz de capturar interações entre atributos.

Os dados foram separados de forma estratificada em 80% para treino e 20% para teste. A escolha foi feita por validação cruzada estratificada de três dobras dentro dos dados de treino. O conjunto de teste foi utilizado uma única vez, apenas para a avaliação final do modelo selecionado.

| Modelo | Accuracy de validação | Recall de validação | F1 de validação |
|---|---:|---:|---:|
| Regressão Logística | 0,974 | 0,959 | 0,965 |
| Random Forest | 0,956 | 0,935 | 0,941 |

A Regressão Logística foi selecionada.

## Avaliação final

No teste isolado de 114 registros, a Regressão Logística obteve:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,965 |
| Recall para malignidade | 0,929 |
| F1-score para malignidade | 0,951 |
| ROC-AUC | 0,996 |

A matriz de confusão apresentou 71 verdadeiros negativos, 39 verdadeiros positivos, 1 falso positivo e 3 falsos negativos. O recall alto é adequado à triagem, mas os três falsos negativos reforçam que o modelo não deve tomar decisões autônomas.

## Explicabilidade

Foram gerados `feature_importance.csv`, `feature_importance.png` e `shap_summary.png`. A importância por permutação, calculada com foco no recall, destacou `texture_worst`, `concavity_worst`, `symmetry_worst` e `concave points_worst` entre as variáveis mais influentes na avaliação atual.

SHAP apresenta a contribuição global dos atributos para a saída do classificador. Os resultados devem ser usados para transparência e revisão clínica, não como relação causal entre uma característica e a doença.

## Discussão crítica

O desempenho medido é promissor para um exercício acadêmico, mas não comprova adequação ao ambiente hospitalar. A base é pública, histórica, limitada e não representa necessariamente a população atendida por uma instituição específica. Também não inclui contexto clínico completo, histórico familiar, exames complementares ou variáveis sociodemográficas.

Para qualquer uso real seriam necessários validação externa, avaliação de viés, monitoramento contínuo, controle de versão de dados e modelos, governança de acesso, conformidade com a LGPD e validação clínica formal. O médico ou médica mantém a palavra final em todo diagnóstico e conduta.
