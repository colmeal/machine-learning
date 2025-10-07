# Relatório do Projeto – KNN e K-Means

## 1. Exploração dos Dados
- Dataset: `fitness_dataset.csv` com atributos demográficos e de estilo de vida.
- Alvo: `is_fit` (1 = Fit, 0 = Not Fit).
- Tratamentos aplicados: substituição de `'?'`, preenchimento por moda, criação de `bmi`,
  conversão `smokes`→`smokes_bin`.

![Distribuições](img/eda_distribuicoes.png)

## 2. Pré-processamento
- Numéricos: **StandardScaler**.
- Categóricos: **OneHotEncoder**.
- Split estratificado 80/20 (treino/teste).

## 3. KNN (supervisionado)
- GridSearchCV (cv=5): `n_neighbors` 1–20, `weights` ['uniform','distance'], `p` [1,2].
- **Melhores hiperparâmetros**: {'clf__n_neighbors': 16, 'clf__p': 1, 'clf__weights': 'distance'}
- **Métricas (teste)**:
|          |    Valor |
|:---------|---------:|
| Acurácia | 0.74     |
| Precisão | 0.705882 |
| Recall   | 0.6      |
| F1       | 0.648649 |

![Curva KNN](img/knn_curva_acuracia.png)
![Matriz KNN](img/knn_matriz_confusao.png)

## 4. K-Means (não supervisionado)
- K=2, treino no conjunto de treino; avaliação no teste após mapear **cluster→classe** (maioria no treino).
- **Métricas (teste)**:
|          |   Valor |
|:---------|--------:|
| Acurácia |     0.6 |
| Precisão |     0   |
| Recall   |     0   |
| F1       |     0   |

![Matriz KMeans](img/kmeans_matriz_confusao.png)
![PCA KMeans](img/kmeans_pca_2d.png)

## 5. Comparação dos Resultados
|                              |   Acurácia |   Precisão |   Recall |     F1 |
|:-----------------------------|-----------:|-----------:|---------:|-------:|
| KNN (Supervisionado)         |       0.74 |     0.7059 |      0.6 | 0.6486 |
| K-Means (Não supervisionado) |       0.6  |     0      |      0   | 0      |

![Comparativo](img/comparativo_metricas.png)

## 6. Conclusões
- **KNN** apresentou melhor desempenho global (Acurácia/F1 maiores), como esperado para método **supervisionado**.
- **K-Means** foi útil para **segmentação exploratória**, aproximando parcialmente as classes reais.
- Próximos passos: balanceamento (SMOTE), ajustes de features e comparação com Logística/SVM/Random Forest.
