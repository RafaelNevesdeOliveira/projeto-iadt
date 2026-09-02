# IADT - Saúde da Mulher com Machine Learning

Projeto da Fase 1 do Tech Challenge para apoio à triagem de risco de câncer de mama. A solução usa dados estruturados de exames para classificar registros como benignos ou malignos. O sistema é educacional e de suporte à decisão: não produz diagnóstico clínico e não substitui profissionais de saúde.

## Repositório

https://github.com/RafaelNevesdeOliveira/projeto-iadt

## Grupo

- Grupo 20
- Rafael Neves de Oliveira
- rafaelneves652@gmail.com
- RM371255

## Objetivo

Comparar modelos de classificação supervisionada, priorizando o recall da classe maligna. Um falso negativo pode atrasar uma investigação clínica; por isso, o modelo selecionado maximiza primeiro o recall, depois F1-score e accuracy.

## Status da entrega

| Item do requisito | Situação |
|---|---|
| Dataset público e classificação estruturada | Concluído |
| EDA, estatísticas, distribuições e correlação | Concluído |
| Pré-processamento e separação treino/teste | Concluído |
| Comparação de dois modelos | Concluído |
| Métricas e avaliação final | Concluído |
| Feature importance e SHAP | Concluído |
| Código, testes, README e relatório técnico | Concluído |
| Vídeo de demonstração | Pendente |

## Dataset

Use o [Breast Cancer Wisconsin Diagnostic Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). Baixe o arquivo `data.csv` e salve-o como `data/raw/breast_cancer.csv`.

O arquivo não é versionado no Git. Antes de usar qualquer fonte externa, confirme licença, procedência e ausência de dados pessoais identificáveis.

## Arquitetura

O projeto aplica Clean Architecture com dependências apontando para o domínio:

```text
presentation -> application -> domain
infrastructure -> application -> domain
```

- `domain`: entidades imutáveis e contratos de entrada/saída.
- `application`: casos de uso e regra de seleção do modelo.
- `infrastructure`: CSV, pandas, scikit-learn e joblib.
- `presentation`: CLI; uma interface Streamlit poderá ser adicionada depois sem alterar as regras centrais.

Não há necessidade de banco vetorial nesta fase. A entrada é tabular e o problema é classificação supervisionada. Uma busca vetorial seria uma extensão futura para prontuários textuais e não participaria do pipeline de diagnóstico estruturado.

## Metodologia

1. Carregamento e validação do CSV, removendo identificadores e convertendo atributos para valores numéricos.
2. Análise exploratória com estatísticas, distribuição das classes, histogramas e matriz de correlação.
3. Divisão estratificada em 80% para treino e 20% para teste.
4. Validação cruzada estratificada de três dobras apenas no treino.
5. Comparação entre Regressão Logística e Random Forest.
6. Seleção por recall da classe maligna, seguida de F1-score e accuracy como desempate.
7. Avaliação única e final no teste com accuracy, recall, F1-score, ROC-AUC e matriz de confusão.
8. Explicabilidade por importância de permutação e SHAP.

## Estrutura

```text
src/iadt_ml/
  domain/          entidades e contratos
  application/     casos de uso
  infrastructure/  adaptadores de dados, ML e persistência
  presentation/    CLI e futuras interfaces
data/raw/          dataset local
models/            artefatos serializados
reports/figures/   gráficos da análise e do relatório
tests/             testes unitários e de integração
```

## Pré-requisitos

- Python 3.11 ou superior
- `pip`
- Ambiente virtual recomendado

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Para a futura interface de demonstração:

```bash
python -m pip install -e '.[demo]'
```

## Execução

Com o dataset salvo no local indicado:

```bash
iadt-ml analyze --dataset data/raw/breast_cancer.csv
iadt-ml train --dataset data/raw/breast_cancer.csv
iadt-ml explain --dataset data/raw/breast_cancer.csv --model models/best_model.joblib
```

Os comandos executam, nesta ordem:

1. EDA: estatísticas descritivas, distribuição das classes, distribuições por atributo e correlação.
2. Treino: separação estratificada de treino e teste, validação cruzada de três dobras somente no treino, seleção de modelo e avaliação final única no teste.
3. Explicabilidade: matriz de confusão, importância por permutação e SHAP.

Os artefatos são gravados em `reports/figures/`; o melhor pipeline é salvo em `models/best_model.joblib`.

Para alterar o local do artefato:

```bash
iadt-ml train --dataset data/raw/breast_cancer.csv --output models/modelo_final.joblib
```

## Qualidade

```bash
pytest
ruff check .
```

## Resultados reproduzidos

Na execução atual, a Regressão Logística foi selecionada pela validação cruzada no treino. No conjunto de teste isolado, alcançou accuracy de `0,965`, recall de `0,929`, F1-score de `0,951` e ROC-AUC de `0,996`.

O relatório técnico em [reports/technical_report.md](reports/technical_report.md) descreve o método, os resultados, a análise crítica e os gráficos gerados.

### Artefatos gerados

| Artefato | Local |
|---|---|
| Relatório técnico (PDF) | [reports/relatorio_tecnico_fase1.pdf](reports/relatorio_tecnico_fase1.pdf) |
| Relatório técnico (Markdown) | [reports/technical_report.md](reports/technical_report.md) |
| Estatísticas descritivas | [reports/figures/descriptive_statistics.csv](reports/figures/descriptive_statistics.csv) |
| Distribuição das classes | [reports/figures/class_distribution.png](reports/figures/class_distribution.png) |
| Correlação | [reports/figures/correlation_heatmap.png](reports/figures/correlation_heatmap.png) |
| Matriz de confusão | [reports/figures/confusion_matrix.png](reports/figures/confusion_matrix.png) |
| Importância das variáveis | [reports/figures/feature_importance.png](reports/figures/feature_importance.png) |
| Explicabilidade SHAP | [reports/figures/shap_summary.png](reports/figures/shap_summary.png) |

## Limitações e responsabilidade

O dataset é público, histórico e limitado. Métricas em teste não comprovam desempenho em ambiente hospitalar. Um uso real exigiria validação clínica externa, governança de dados, LGPD, monitoramento de viés, segurança e decisão final de um profissional habilitado.
