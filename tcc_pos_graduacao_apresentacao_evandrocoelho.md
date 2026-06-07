# Previsão do Consumo Energético de Edifícios com Modelos de Aprendizado de Máquina
## Um Estudo Comparativo com Engenharia de Recursos Expandida

**Evandro Ribeiro Gomes Coelho**
Especialização em Ciência de Dados e Big Data — PUC Minas · 2026

---

## Slide 1 — Introdução

**Por que prever o consumo energético de edifícios?**

- Cerca de **70% do consumo** em edificações decorre de iluminação e climatização (PROCEL, 2020)
- A previsão de curto prazo orienta a **gestão de faturas** e sinaliza **consumos ineficientes**
- Modelos computacionais "facilitam estimativas de retornos financeiros e ambientais sobre o investimento" (Amasyali & El-Gohary, 2017)

**Contexto deste trabalho**

- Evolução de dois TCCs anteriores (COELHO, 2021a; 2021b)
- O TCC de graduação atingiu R² = 0,868 com MLP e 6 recursos — mas com metodologia de validação inadequada

---

## Slide 2 — O Problema Proposto

**Questão central**

> Em que medida a expansão dos recursos preditores e a adoção de múltiplos algoritmos melhoram a previsão do consumo energético, em relação à abordagem anterior?

**Lacuna identificada**

| | TCC Graduação (2021b) | Este trabalho (2026) |
|---|---|---|
| Recursos de entrada | 6 variáveis | 20 variáveis |
| Validação cruzada | *k-fold* aleatório | `TimeSeriesSplit` (cronológico) |
| R² reportado | **0,868** | Reavaliado: **−0,108** |

> O resultado anterior foi inflacionado por **vazamento de dados** entre treino e teste

---

## Slide 3 — Objetivos

**Objetivo geral**
Desenvolver e comparar modelos de aprendizado de máquina para previsão de curto prazo do consumo energético de edifícios, com ênfase na contribuição da **engenharia de recursos**

**Objetivos específicos**
- Realizar análise exploratória e identificar padrões temporais
- Construir conjunto expandido de **20 recursos preditores**
- Implementar modelos de referência: MMS, MME, Regressão Linear e SARIMA
- Implementar modelos avançados: MLP, Random Forest, XGBoost e LSTM
- Comparar todos os modelos por MAE, RMSE, R² e MAPE
- Analisar importância de variáveis nos modelos baseados em árvore

---

## Slide 4 — Coleta de Dados

**Base:** *Building Sites Power Consumption* — Möbius (2021) / Kaggle

| Arquivo | Conteúdo |
|---|---|
| `plfec-train-data.csv` | 6,5 milhões de medições de energia (kWh) — 267 edifícios |
| `plfec-weather.csv` | Temperaturas horárias (°C) |
| `plfec-holidays.csv` | Feriados por edifício |
| `plfec-metadata.csv` | Área construída e dias de folga |

**Edifício selecionado: SiteId = 40**

- **17.352 medições horárias** · jul/2015 – nov/2017
- Cobertura de temperatura: **97,9%**
- Clima temperado · temperatura entre −8 °C e 38 °C
- Consumo médio: **5.476 kWh/hora** · desvio padrão: 3.950 kWh

---

## Slide 5 — Processamento de Dados

**Pipeline de preparação**

1. **Junção** das bases por `Timestamp` (*inner join*) → 16.995 registros
2. **Outliers**: 116 identificados (0,7%) — mantidos por representar comportamento legítimo
3. **Engenharia de recursos**: de 6 para **20 variáveis preditoras**

**Principais recursos criados**

| Grupo | Exemplos | Impacto |
|---|---|---|
| Atraso (lag) | `energy_lag1`, `energy_lag24` | r = 0,895 com o alvo |
| Médias móveis | `roll_mean_6h`, `roll_mean_24h` | r = 0,863 |
| Codificação cíclica | `hr_sin/cos`, `dow_sin/cos`, `month_sin/cos` | Elimina descontinuidade numérica |
| Calendário | `is_weekend`, `is_holiday` | Redução expressiva de consumo |
| Interação | `temp_x_hour`, `delta_energy` | Captura variação súbita |

> Base final: **16.632 registros · 20 preditores · divisão 75/25 cronológica**

---

## Slide 6 — Modelos

**Grupo 1 — Modelos de referência** *(baseline)*

| Modelo | Melhor R² |
|---|---|
| MMS (Média Móvel Simples) | 0,468 |
| MME (Média Móvel Exponencial) | 0,619 |
| Regressão Linear Múltipla | **0,707** |
| SARIMA (subconjunto 48h) | 0,360 |

**Grupo 2 — Modelos avançados** · 20 recursos · `TimeSeriesSplit` · `GridSearchCV`

| Modelo | Configuração | Validação CV |
|---|---|---|
| MLP TCC2 (controle) | 43 neurônios · 6 recursos | R² médio = −1,006 |
| MLP expandido | 100 neurônios · 3.000 iter. | R² CV = 0,574 |
| Random Forest | 200 árvores · profund. 20 | R² CV = 0,987 |
| XGBoost | 300 estimadores · lr 0,05 | R² CV = 0,989 |
| LSTM | 64 unidades · Dropout 20% | Parada na época 40 |

---

## Slide 7 — Resultados

**Ranking final — conjunto de teste (4.158 registros)**

| Rank | Modelo | MAE (kWh) | R² | MAPE (%) |
|---|---|---|---|---|
| 🥇 | MLP (recursos expandidos) | **3,36** | **0,99996** | **0,099** |
| 🥈 | Random Forest | 37,47 | 0,99928 | 0,988 |
| 🥉 | XGBoost | 95,83 | 0,99786 | 2,606 |
| 4 | LSTM | 104,26 | 0,99733 | 3,715 |
| 5 | Regressão Linear | 1.083,81 | 0,707 | 27,5 |
| 13 | MLP TCC2 (original reavaliado) | 2.435,50 | **−0,108** | 75,6 |

**Conclusões**

- A **engenharia de recursos é o fator determinante** — mesmo algoritmo, resultado oposto
- Redução de erro absoluto de **99,7%** entre Regressão Linear e MLP expandido
- `energy_lag1` e `delta_energy` concentram **>99% da importância** nos modelos de árvore
- **Random Forest** é o modelo recomendado para operação: R² = 0,999, interpretável e estável

---

## Encerramento

**Contribuição principal**
A expansão de 6 para 20 recursos transformou um modelo ineficaz (R² = −0,108) no melhor resultado do estudo (R² = 0,99996) — **sem trocar o algoritmo**

**Trabalhos futuros**
- Previsão de múltiplos passos à frente (horizonte de 24h)
- Aplicação em edifícios com histórico reduzido
- Sistema de detecção de anomalias baseado em resíduos

**Repositório:** github.com/evandrocoelho/energy-consumption-forecast

*Obrigado.*