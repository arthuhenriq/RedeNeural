import pandas as pd  # pylint: disable=C0114
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split  # pylint: disable=E0401
from sklearn.preprocessing import StandardScaler  # pylint: disable=E0401
from sklearn.neural_network import MLPClassifier  # pylint: disable=E0401
from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    roc_auc_score,
)  # pylint: disable=E0401
from sklearn.preprocessing import LabelEncoder  # pylint: disable=E0401
from sklearn.metrics import r2_score  # pylint: disable=E0401

# Passo 1: Ler os dados do arquivo CSV
data = pd.read_csv("breast.csv")

##print(data)

# 2. Pré-processamento dos dados
X = data[
    [
        "id",
        "clump_thickness",
        "u_cell_size",
        "u_cell_shape",
        "marginal_adhesion",
        "s_epithelial_cell_size",
        "bare_nuclei",
        "bland_chromatin",
        "normal_nucleoli",
        "mitoses",
    ]
]
y = data["Class"]

scaler = StandardScaler()
scaler.fit(X)

X_norm = scaler.transform(X)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_norm_train, X_norm_test, y_norm_train, y_norm_test = train_test_split(
    X_norm, y, train_size=0.8, test_size=0.2, random_state=20
)


# 3. Modelagem e configuração da RNA
model = MLPClassifier(
    hidden_layer_sizes=(90),
    max_iter=2000,
    activation="tanh",
    solver="adam",
    alpha=0.003,
    random_state=10,
    learning_rate_init=0.1,
    learning_rate="constant",
    verbose=2,
)
model.fit(X_norm_train, y_norm_train)

# 4. Apresentação dos resultados
y_pred = model.predict(X_norm_test)
y_prob = model.predict_proba(X_norm_test)[:, 1]  # Probabilidade da classe positiva

# Accuracy
y_pred_r2 = r2_score(y_norm_test, y_pred)
print()
print("-----------------------------------------")
print("R2 MODEL RNA: ", y_pred_r2)
print("-----------------------------------------")
print()

# Prever o restante dos dados reservados para teste
y_pred_future = model.predict(X_norm_test)
X_test = scaler.inverse_transform(X_norm_test)

# Impressão do ID e do valor previsto pela rede para comparar com o valor real no txt com os dados
print("ID e Valor Previsto:")
for id_value, pred_value in zip(
    X_test[:, 0], y_pred_future
):  # Pegando o ID na primeira Coluna
    print("-------------------------------------------------------------")
    print("0 para BENIGNO e 1 para MALIGNO")
    print(f"ID: {id_value}, Valor Previsto: {pred_value}")
print("-------------------------------------------------------------")
print()

# Plotar a matriz de confusão
cm = confusion_matrix(y_norm_test, y_pred)
cm_normalized = (
    cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
)  # Normalized confusion matrix

plt.figure(figsize=(12, 6))

# Plotar a matriz de confusão
plt.subplot(1, 2, 1)
plt.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.get_cmap("Blues"))
plt.title("Matriz de Confusão")
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ["Benigno", "Maligno"])
plt.yticks(tick_marks, ["Benigno", "Maligno"])

# Adicionar valores numéricos à matriz de confusão
thresh = cm_normalized.max() / 2.0
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(
            j,
            i,
            format(cm_normalized[i, j], ".2f"),
            ha="center",
            va="center",
            color="white" if cm_normalized[i, j] > thresh else "black",
        )

plt.xlabel("Predito")
plt.ylabel("Verdadeiro")

# Plotar a curva ROC
plt.subplot(1, 2, 2)
fpr, tpr, thresholds = roc_curve(y_norm_test, y_prob)
roc_auc = roc_auc_score(y_norm_test, y_prob)
plt.plot(fpr, tpr, label="Curva ROC (área = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Taxa de Falso Positivo")
plt.ylabel("Taxa de Verdadeiro Positivo")
plt.title("Curva ROC")
plt.legend(loc="lower right")

# Adicionar valores numéricos à curva ROC
threshold_labels = [str(round(threshold, 2)) for threshold in thresholds]
for i, threshold_label in enumerate(threshold_labels):
    if i % 20 == 0:  # Display every 20th threshold label
        plt.text(
            fpr[i], tpr[i], threshold_label, fontsize=8, verticalalignment="center"
        )

plt.tight_layout()
plt.show()
