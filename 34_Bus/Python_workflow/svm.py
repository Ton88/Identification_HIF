from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import pandas as pd

# Carregar os dados
data = pd.read_csv('/data/Bus_13_mexh.csv', delimiter=';', engine='python')

# Separar os dados em recursos (X) e alvo (y)
X = data.iloc[:, :-1]  # Todas as colunas, exceto a última
y = data.iloc[:, -1]   # Apenas a última coluna

# Converter os dados para valores numéricos
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)  # Valores não numéricos são convertidos para 0
y = pd.to_numeric(y, errors='coerce').fillna(0)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=80
)

# Parâmetros da SVM
svm_type = 'epsilon'  # Tipo de SVM: 'epsilon' para regressão epsilon-SVR
C = 1.0               # Custo da regressão (Complexity Bound)
kernel = 'rbf'        # Kernel utilizado ('linear', 'poly', 'rbf', 'sigmoid')
epsilon = 0.1         # Tolerância numérica
max_iter = -1         # Limite de iterações (-1 para ilimitado)

# Criar e treinar o modelo SVM
svm = SVR(kernel=kernel, C=C, epsilon=epsilon, max_iter=max_iter)
svm.fit(X_train, y_train)

# Fazer previsões e avaliar o modelo
y_pred = svm.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred.round())
report = classification_report(y_test, y_pred.round())

print(f"Erro Quadrático Médio: {mse:.2f}")
print(f"Acurácia: {accuracy:.2f}")
print("Relatório de Classificação:")
print(report)

# Comentários sobre os parâmetros:
# - svm_type: Determina o tipo de SVM, aqui usamos SVR para regressão.
# - C: Controla o trade-off entre alcançar uma margem maior e reduzir o erro de classificação.
# - kernel: Define a função kernel a ser usada para transformar os dados.
# - epsilon: Especifica uma margem de tolerância no problema de regressão.
# - max_iter: Define o número máximo de iterações para a otimização.
