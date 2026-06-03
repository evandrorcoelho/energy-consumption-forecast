# Energy Consumption Forecast
**TCC — Pós-Graduação Lato Sensu em Ciência de Dados e Big Data**
PUC Minas | Evandro Ribeiro Gomes Coelho

Evolução do TCC de Engenharia Elétrica (CEFET-MG, 2021) para um projeto completo
de Ciência de Dados, com análise exploratória aprofundada, feature engineering avançado
e comparação de múltiplos modelos de Machine Learning.

---

## Contexto

| | TCC1 (CEFET — Monografia) | TCC2 (CEFET — Artigo) | Este TCC (PUC) |
|---|---|---|---|
| Tipo | Revisão teórica | Artigo científico | Projeto de Ciência de Dados |
| Dados | TRT3 (mensal, ~30 registros/prédio) | Kaggle — horário, 17k registros | Kaggle — mesma base, tratamento expandido |
| Modelo | Nenhum (foco em metodologia) | MLP (1 camada oculta) | MLP + Random Forest + XGBoost + LSTM |
| Features | — | 6 | 20 |
| Validação | — | k-fold aleatório | TimeSeriesSplit |
| R² | — | 0,87 (SiteId 40) | Ver notebook 05 |

---

## Dataset

**Building Sites Power Consumption** — Möbius (2021), [Kaggle](https://www.kaggle.com/arashnic/building-sites-power-consumption-dataset)

| Arquivo | Descrição | Tamanho |
|---|---|---|
| `plfec-train-data.csv` | Medições de energia (treino) — SiteId, Timestamp, Value (kWh) | ~403 MB |
| `plfec-test-data.csv` | Medições de energia (teste) | ~80 MB |
| `plfec-weather.csv` | Temperatura por prédio e distância | ~209 MB |
| `plfec-holidays.csv` | Feriados por prédio | ~268 KB |
| `plfec-metadata.csv` | Metadados dos prédios | ~19 KB |

Os arquivos originais ficam em `data/raw/` (symlinks para o OneDrive local).

---

## Estrutura

```
energy-consumption-forecast/
├── data/
│   ├── raw/            # Symlinks para os CSVs originais (~2 GB)
│   └── processed/      # Dados derivados (gerados pelos notebooks)
│       ├── df_mlp.csv              (TCC2 — SiteId 40, pré-processado)
│       ├── df_site40_merged.csv    (gerado pelo nb01)
│       ├── df_features.csv         (gerado pelo nb02)
│       ├── results_baseline.csv    (gerado pelo nb03)
│       ├── results_advanced.csv    (gerado pelo nb04)
│       ├── predictions.csv         (gerado pelo nb04)
│       └── results_final.csv       (gerado pelo nb05)
├── notebooks/
│   ├── 01_eda.ipynb                # EDA multi-prédio, STL, correlações
│   ├── 02_feature_engineering.ipynb # Lag, rolling, encoding cíclico
│   ├── 03_baseline_models.ipynb    # MMS, MME, Regressão Linear, SARIMA
│   ├── 04_advanced_models.ipynb    # MLP, Random Forest, XGBoost, LSTM
│   ├── 05_results_comparison.ipynb # Dashboard, SHAP, conclusões
│   └── legacy/                     # Notebooks originais do TCC2 (CEFET)
│       ├── 1_preprocess.ipynb
│       ├── 2_final-df.ipynb
│       ├── 3_train-mlp.ipynb
│       ├── 4_train-mlp-neurons.ipynb
│       ├── 5-train-neurons-chart.ipynb
│       └── 6_final-mlp.ipynb
└── README.md
```

---

## Ordem de Execução dos Notebooks

Execute na sequência:

1. **`01_eda.ipynb`** — Carrega os dados brutos, explora distribuições, gera `df_site40_merged.csv`
2. **`02_feature_engineering.ipynb`** — Cria 20 features, gera `df_features.csv`
3. **`03_baseline_models.ipynb`** — Modelos clássicos como referência, gera `results_baseline.csv`
4. **`04_advanced_models.ipynb`** — ML avançado com GridSearch + TimeSeriesSplit, gera `results_advanced.csv` e `predictions.csv`
5. **`05_results_comparison.ipynb`** — Dashboard, SHAP, análise de erros, conclusões

---

## Dependências

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels pmdarima xgboost tensorflow shap
```

---

## Alinhamento com o Template PUC (.doc)

| Seção do TCC | Notebook |
|---|---|
| 2. Coleta de Dados | `01_eda.ipynb` |
| 3. Processamento/Tratamento | `02_feature_engineering.ipynb` |
| 4. Análise e Exploração | `01_eda.ipynb` |
| 5. Criação de Modelos de ML | `03_baseline_models.ipynb` + `04_advanced_models.ipynb` |
| 6. Interpretação dos Resultados | `04_advanced_models.ipynb` + `05_results_comparison.ipynb` |
| 7. Apresentação dos Resultados | `05_results_comparison.ipynb` |

---

## Referências

- Möbius (2021). *Building Sites Power Consumption*. Kaggle.
- Coelho, E. R. G. (2021). *Previsão do Consumo Energético de Prédios Públicos por Rede Neural Artificial MLP*. CEFET-MG (TCC2).
- Zhao, H.-x., & Magoulès, F. (2012). *A review on the prediction of building energy consumption*. Renewable and Sustainable Energy Reviews.
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.
