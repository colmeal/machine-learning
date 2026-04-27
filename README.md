#  Machine Learning — Mateus Colmeal

> Repositório de projetos práticos de Machine Learning desenvolvidos durante a graduação em Sistemas de Informação na ESPM (2025.4).
> Cada módulo documenta o pipeline completo: exploração de dados, pré-processamento, treinamento, avaliação e comparação de modelos.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![MkDocs](https://img.shields.io/badge/Docs-MkDocs-blueviolet?logo=readthedocs)](https://colmeal.github.io/machine-learning/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 📚 Documentação Completa

> 👉 **[colmeal.github.io/machine-learning](https://colmeal.github.io/machine-learning/)**

A documentação gerada com MkDocs detalha cada algoritmo com: relatório técnico, trechos de código comentados, visualizações e análise de resultados.

---

## 🗂️ Projetos

| Módulo | Algoritmo | Dataset | Melhor Acurácia | Destaque |
|---|---|---|---|---|
| [KNN](https://colmeal.github.io/machine-learning/KNN/main/) | K-Nearest Neighbors | `fitness_dataset.csv` | ~75% | GridSearchCV com k=18, weights='distance' |
| [Árvore de Decisão](https://colmeal.github.io/machine-learning/arvore-decisao/main/) | Decision Tree | `fitness_dataset.csv` | — | Visualização da árvore gerada |
| [Random Forest](https://colmeal.github.io/machine-learning/RandomForest/main/) | Random Forest | `fitness_dataset.csv` | ~82% | Feature importance: `activity_index` |
| [K-Means](https://colmeal.github.io/machine-learning/K-Mean/main/) | K-Means Clustering | — | — | Clusterização não supervisionada |
| [SVM](https://colmeal.github.io/machine-learning/SupportVectorMachine/main/) | Support Vector Machine | — | — | Fronteira de decisão com kernel RBF |
| [PageRank](https://colmeal.github.io/machine-learning/PageRank/main/) | PageRank | — | — | Análise de grafos e ranking de nós |
| [Métricas de Avaliação](https://colmeal.github.io/machine-learning/MetricasAvaliacao/main/) | — | — | — | Precision, Recall, F1, Curva ROC |

---

## 🔍 Destaque: KNN vs Random Forest

O mesmo dataset (`fitness_dataset.csv`, classificação binária `is_fit`) foi usado para comparar os dois modelos:

| Aspecto | KNN | Random Forest |
|---|---|---|
| Acurácia (teste) | ~75% | ~82% |
| Tempo de predição | Lento | Muito rápido |
| Requer normalização | Sim (StandardScaler) | Não |
| Feature importance | ❌ | ✅ `activity_index` = 35% |
| Hiperparâmetros | k=18, distance, Euclidiana | 200 árvores, max_features=sqrt |

**Conclusão:** Random Forest superou KNN em acurácia (+7pp) e forneceu interpretabilidade via feature importance.

---

## 🛠️ Stack

- **Linguagem:** Python 3.12  
- **ML:** Scikit-learn 
- **Dados:** Pandas, NumPy  
- **Visualização:** Matplotlib  
- **Documentação:** MkDocs + Material for MkDocs  
- **Versionamento:** Git + GitHub Actions (CI/CD automático para GitHub Pages)

---

## ⚙️ Como executar localmente

**1. Clone o repositório**
```bash
git clone https://github.com/colmeal/machine-learning.git
cd machine-learning
```

**2. Crie e ative o ambiente virtual**
```bash
python3 -m venv env
source ./env/bin/activate        # Linux/macOS
.\env\Scripts\activate           # Windows
```

**3. Instale as dependências**
```bash
pip install -r requirements.txt
```

**4. Execute a documentação localmente (opcional)**
```bash
mkdocs serve -o
```

---

## 📁 Estrutura do Repositório

```
machine-learning/
├── data/                    # Datasets utilizados nos projetos
├── docs/                    # Código-fonte da documentação MkDocs
│   ├── KNN/
│   ├── RandomForest/
│   ├── SupportVectorMachine/
│   ├── K-Mean/
│   ├── PageRank/
│   ├── PySpark/
│   ├── MetricasAvaliacao/
│   └── arvore-decisao/
├── .github/workflows/       # CI/CD — deploy automático no GitHub Pages
├── mkdocs.yml               # Configuração do MkDocs
└── requirements.txt         # Dependências do projeto
```

---

## 👤 Autor

**Mateus Carnevale Colmeal**  
Estudante de Sistemas de Informação — ESPM (2024–2027)  
Foco em Análise de Dados e Machine Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-mateus--colmeal-0077B5?logo=linkedin)](https://www.linkedin.com/in/mateus-colmeal)
[![GitHub](https://img.shields.io/badge/GitHub-colmeal-181717?logo=github)](https://github.com/colmeal)