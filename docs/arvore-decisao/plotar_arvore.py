import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier, plot_tree

DATA_DIR = "data"
IMG_DIR  = "data/img"
os.makedirs(IMG_DIR, exist_ok=True)

x_train = pd.read_csv(f"{DATA_DIR}/dataset-x-train.csv")
y_train = pd.read_csv(f"{DATA_DIR}/dataset-y-train.csv")["is_fit"]

# Certifique-se de que todas as colunas são numéricas
categorical_cols = x_train.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    x_train = pd.get_dummies(x_train, columns=categorical_cols, drop_first=True)


clf_viz = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,  # ajuste para 3
    random_state=42
)
clf_viz.fit(x_train, y_train)

plt.figure(figsize=(12, 6))  # ajuste o tamanho para ficar mais compacto
plot_tree(
    clf_viz,
    feature_names=x_train.columns,
    class_names=["is fit(0)", "is not fit(1)"],
    filled=True, rounded=True, fontsize=10,
    max_depth=3  # mantenha igual ao treinamento
)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "tree_top_depth3.png"), dpi=200, bbox_inches="tight")
plt.show()