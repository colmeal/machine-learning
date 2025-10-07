# 🧩 Projeto de Machine Learning – K-Means

---

## 1. Exploração dos Dados 

O dataset analisado, **`fitness_dataset.csv`**, contém **2.000 registros** e **11 variáveis**, relacionadas à **saúde e hábitos de vida** dos participantes.  
O objetivo da análise foi aplicar **K-Means** para identificar **agrupamentos naturais (clusters)** de indivíduos com características semelhantes, sem utilizar a variável-alvo `is_fit`.

### 🧠 Descrição das variáveis principais

| Variável | Descrição | Tipo |
|-----------|------------|------|
| `age` | Idade do indivíduo | Numérica |
| `height_cm` | Altura em centímetros | Numérica |
| `weight_kg` | Peso corporal (com outliers) | Numérica |
| `heart_rate` | Frequência cardíaca de repouso | Numérica |
| `blood_pressure` | Pressão arterial sistólica | Numérica |
| `sleep_hours` | Média de horas de sono | Numérica (com NaN) |
| `nutrition_quality` | Qualidade da alimentação (0–10) | Numérica |
| `activity_index` | Nível de atividade física (1–5) | Numérica |
| `smokes` | Hábito de fumar (0/1 ou sim/não) | Categórica |
| `gender` | Sexo biológico | Categórica |
| `is_fit` | Alvo binário (0 = não em forma, 1 = em forma) | Binária |

### 📊 Análise exploratória

- A variável `is_fit` apresenta **leve desbalanceamento** (~60% não fit e 40% fit).  
- Há correlações coerentes:  
  - `activity_index` e `nutrition_quality` correlacionam-se positivamente com o fitness;  
  - `weight_kg`, `heart_rate` e `blood_pressure` estão associadas negativamente.  
- O dataset possui **ruído e sobreposição entre classes**, simulando dados reais.

(Insira aqui figuras exploratórias geradas: distribuição, crosstab por gênero, estatísticas descritivas.)

---

## 2. Pré-processamento

O tratamento dos dados incluiu:

1. **Limpeza e imputação de valores ausentes**
   - Símbolos “?” convertidos em `NaN`.
   - Numéricas: imputadas pela **mediana**.
   - Categóricas: imputadas pela **moda**.

2. **Padronização e codificação**
   - `smokes` convertida em binária (`yes/1` → 1, `no/0` → 0).  
   - `gender` transformada em **OneHotEncoder**.

3. **Criação de variável derivada**
   - `bmi` (índice de massa corporal) = `weight_kg / (height_cm/100)**2`.

4. **Normalização**
   - Escalonamento com **StandardScaler** para evitar distorção entre variáveis.

Trecho ilustrativo:
```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_imputer = SimpleImputer(strategy="median")
X_num = num_imputer.fit_transform(X_num_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)
```

---

## 3. Divisão dos Dados 

Abordagem adotada:

- O K‑Means é não supervisionado, portanto o treino usa todos os registros processados.  
- Entretanto, para avaliação e relatórios, preservei a coluna `is_fit` e exportei atribuições por registro para validação externa.

Arquivos gerados:
- `data/kmeans_assignments.csv` — atribuição de cluster por registro com `is_fit`.
- `data/kmeans_cluster_profile.csv` — médias por cluster.
- `data/kmeans_cluster_fit_distribution.csv` — proporção de `is_fit` por cluster.

---

## 4. Treinamento e Execução do K-Means 

O modelo foi implementado conforme o pipeline:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
```

### 🔹 Projeção PCA 2D dos Clusters

![PCA 2D](../img/kmeans_pca_clusters.png)

A redução de dimensionalidade com PCA exibe três agrupamentos principais:
- Dois clusters com **alta sobreposição** (hábitos intermediários).
- Um cluster mais distinto, representando casos extremos (ex.: alto BMI ou baixa atividade).

---

## 5. Avaliação e Interpretação dos Resultados

### 🧩 Perfis médios por cluster

| Cluster | Perfil predominante | Características médias observadas |
|---|---|---|
| 0 – Sedentário | Maior peso, menor atividade e nutrição | `activity_index` baixo, `bmi` alto, `nutrition_quality` baixo |
| 1 – Intermediário | Hábitos medianos | Valores equilibrados |
| 2 – Ativo/Saudável | Melhor nutrição e atividade física | `activity_index` e `nutrition_quality` altos, `bmi` baixo |

### 📈 Cruzamento com `is_fit`

- **Cluster 2 (Ativo)** → maior proporção de `is_fit = 1`.  
- **Cluster 0 (Sedentário)** → predominância de `is_fit = 0`.  
- **Cluster 1 (Intermediário)** → mistura equilibrada entre ambos.

Essas relações confirmam que o modelo capturou padrões coerentes com condicionamento físico, mesmo sem usar `is_fit` no treino.

---

## 6. Relatório Final e Melhorias

### 🧾 Conclusões

- K-Means com **K = 3** apresentou boa interpretabilidade prática e capturou perfis coerentes de saúde.  
- Resultados mostram relação consistente entre atividade, nutrição e probabilidade de estar em forma.

### ⚠️ Limitações

- Outliers em `weight_kg` e `blood_pressure` afetam a formação dos clusters.

### 🚀 Sugestões de melhoria

1. Testar algoritmos alternativos (DBSCAN, Gaussian Mixture).  
2. Tratar/remover outliers em `weight_kg` e `bmi` antes do agrupamento.  
3. Aplicar PCA ou t-SNE antes do K‑Means para melhorar separabilidade.  
4. Enriquecer o conjunto de features (IMC categórico, qualidade do sono categorizada, interações).

---