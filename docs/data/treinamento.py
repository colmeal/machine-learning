import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

data_dir = "data"
img_dir = "docs/img"

os.makedirs(img_dir, exist_ok=True)

 
x_train = pd.read_csv(f"data/dataset-x-train.csv")
x_test  = pd.read_csv(f"data/dataset-x-test.csv")

y_train = pd.read_csv(f"data/dataset-y-train.csv")["is_fit"]
y_test  = pd.read_csv(f"data/dataset-y-test.csv")["is_fit"]

# Certifique-se de que todas as colunas são numéricas
categorical_cols = x_train.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    x_train = pd.get_dummies(x_train, columns=categorical_cols, drop_first=True)
    x_test = pd.get_dummies(x_test, columns=categorical_cols, drop_first=True)
    # Alinhar colunas dos conjuntos de treino e teste
    x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)

clf = DecisionTreeClassifier(random_state=42) #criando um classificador
clf.fit(x_train, y_train) #treina com os dados do treino
y_pred = clf.predict(x_test) #gera previsões no cj de testes

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro", zero_division=0
)

print("\n== Classification Report (Baseline) ==")
print(classification_report(y_test, y_pred, target_names=["is fit(0)", "is not fit(1)"], zero_division=0))
print(f"Accuracy: {acc:.4f} | Precision_macro: {prec:.4f} | Recall_macro: {rec:.4f} | F1_macro: {f1:.4f}")


labels = [0,1]
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.imshow(cm)
plt.title("Matriz de confusão")
plt.xticks([0, 1], ["is fit(0)", "is notfit(1)"])
plt.yticks([0, 1], ["is fit(0)", "is  not fit(1)"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "cm_baseline.png"))
plt.show()
plt.clf()
