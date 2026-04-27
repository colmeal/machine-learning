import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# 1. CARREGAR E PRÉ-PROCESSAR O DATASET
# ==========================================

# Caminho do seu arquivo
caminho = r"C:\Users\mateus.colmeal\Documents\GitHub\machine-learning\docs\data\fitness_dataset.csv"

df = pd.read_csv(caminho, sep=",")

# Tratar NaN em sleep_hours
df["sleep_hours"].fillna(df["sleep_hours"].median(), inplace=True)

# Codificar smokes e gender
df["smokes"] = df["smokes"].astype(str).map({"yes": 1, "no": 0, "1": 1, "0": 0})
df["gender"] = df["gender"].map({"F": 0, "M": 1})

# X: features, y: alvo (is_fit)
X = df.drop("is_fit", axis=1).values
y_orig = df["is_fit"].values

# SVM "vanilla" trabalha com rótulos -1 e +1
y = np.where(y_orig == 1, 1, -1)

# Normalizar (SVM com kernel usa distâncias -> importante escalar)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print("Shapes: ", X_train.shape, X_test.shape)


# ==========================================
# 2. DEFINIR KERNEL RBF E MATRIZ K
# ==========================================

def rbf_kernel(x1, x2, sigma=1.0):
    """Kernel RBF (gaussiano)."""
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))

def kernel_matrix(X, kernel, sigma=1.0):
    """Monta a matriz K_{ij} = K(x_i, x_j)."""
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(X[i], X[j], sigma)
    return K

sigma = 1.0
K = kernel_matrix(X_train, rbf_kernel, sigma)

# ==========================================
# 3. PROBLEMA DUAL: OBJETIVO + RESTRIÇÕES
# ==========================================

# P = y_i y_j K_ij
P = np.outer(y_train, y_train) * K

def objective(alpha):
    """
    Função objetivo da formulação dual:
    min  0.5 * alpha^T P alpha - sum(alpha)
    (Estamos minimizando a negativa da função de maximização)
    """
    return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)

def constraint_eq(alpha):
    """
    Restrição de igualdade: sum(alpha_i * y_i) = 0
    """
    return np.dot(alpha, y_train)

cons = {'type': 'eq', 'fun': constraint_eq}

# Hard-margin: alpha_i >= 0 (sem C explícito)
bounds = [(0, None) for _ in range(len(y_train))]

# Chute inicial
alpha0 = np.zeros(len(y_train))

# ==========================================
# 4. OTIMIZAÇÃO (resolver para alpha)
# ==========================================
print("Otimizando alpha (isso pode levar alguns segundos)...")
res = optimize.minimize(
    objective,
    alpha0,
    method='SLSQP',
    bounds=bounds,
    constraints=cons,
    options={'maxiter': 1000}
)

alpha = res.x

# Vetores de suporte: alpha_i > limiar
sv_threshold = 1e-5
sv_idx = alpha > sv_threshold

print(f"Número de vetores de suporte: {sv_idx.sum()} / {len(alpha)}")

# ==========================================
# 5. CÁLCULO DO VIÉS b
# ==========================================

# Usamos um dos vetores de suporte para calcular b
i = np.where(sv_idx)[0][0]
b = y_train[i] - np.dot(alpha * y_train, K[i, :])
print("b (viés) =", b)

# ==========================================
# 6. FUNÇÃO DE DECISÃO E PREVISÃO
# ==========================================

def decision_function(x):
    """
    f(x) = sum(alpha_i * y_i * K(x_i, x)) + b
    onde x_i são os pontos de treino.
    """
    kx = np.array([rbf_kernel(x, xi, sigma) for xi in X_train])
    return np.dot(alpha * y_train, kx) + b

def predict(X_new):
    """
    Aplica a função de decisão em vários pontos.
    Retorna rótulos -1/+1.
    """
    f_vals = np.array([decision_function(x) for x in X_new])
    y_pred = np.where(f_vals >= 0, 1, -1)
    return y_pred

# ==========================================
# 7. AVALIAÇÃO NO CONJUNTO DE TESTE
# ==========================================

y_pred = predict(X_test)

# Converter de -1/+1 para 0/1 para comparar com o original
y_test_01 = np.where(y_test == 1, 1, 0)
y_pred_01 = np.where(y_pred == 1, 1, 0)

acc = accuracy_score(y_test_01, y_pred_01)
cm = confusion_matrix(y_test_01, y_pred_01)

print("\nAcurácia SVM (do zero):", acc)
print("\nMatriz de confusão:\n", cm)
print("\nRelatório de classificação:")
print(classification_report(y_test_01, y_pred_01))


import os
import matplotlib.pyplot as plt

# Criar pasta de saída para imagens
output_dir = r"C:\Users\mateus.colmeal\Documents\GitHub\machine-learning\docs\SupportVectorMachine\images"
os.makedirs(output_dir, exist_ok=True)

# ================================
# 1. MATRIZ DE CONFUSÃO (IMAGEM)
# ================================
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Matriz de Confusão - SVM (Do Zero)")
plt.xlabel("Predito")
plt.ylabel("Real")

# Escrita dos valores dentro da matriz
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_svm.png"), dpi=300)
plt.close()

# ================================
# 2. DISTRIBUIÇÃO DAS PREVISÕES
# ================================
plt.figure(figsize=(7, 4))
plt.bar(["Classe 0", "Classe 1"], [sum(y_pred_01==0), sum(y_pred_01==1)],
        color=["skyblue", "salmon"])
plt.title("Distribuição das Previsões do SVM")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "predictions_distribution_svm.png"), dpi=300)
plt.close()

# ================================
# 3. COMPARAÇÃO REAL VS PREVISTO
# ================================
plt.figure(figsize=(7, 4))
plt.scatter(range(len(y_test_01)), y_test_01, label="Real", s=10)
plt.scatter(range(len(y_pred_01)), y_pred_01, label="Predito", s=10, alpha=0.6)
plt.title("Comparação entre Valores Reais e Preditos")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "real_vs_pred_svm.png"), dpi=300)
plt.close()

# ================================
# 4. HISTOGRAMA DOS SCORES (FUNÇÃO DECISÃO)
# ================================
scores = np.array([decision_function(x) for x in X_test])

plt.figure(figsize=(7,4))
plt.hist(scores, bins=30, color="purple", alpha=0.7)
plt.title("Distribuição da Função de Decisão (f(x)) - SVM")
plt.xlabel("Score f(x)")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "decision_function_histogram_svm.png"), dpi=300)
plt.close()

print("\nImagens geradas e salvas em:", output_dir)
