from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Carregar os dados
data = pd.read_csv('/.csv', delimiter=';', engine='python')

# Separar os dados em recursos (X) e alvo (y)
X = data.iloc[:, :-1]  # Todas as colunas, exceto a última
y = data.iloc[:, -1]   # Apenas a última coluna

# Converter os dados para valores numéricos
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)  # Valores não numéricos são convertidos para 0
y = pd.to_numeric(y, errors='coerce').fillna(0)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=80, stratify=y
)

# Parâmetros do modelo KNN
n_neighbors = 8  # Número de vizinhos
weights = 'distance'  # Pesos (uniform ou distance)
p = 2  # Distância de Minkowski (p=2 é equivalente à distância euclidiana)

# Criar e treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
knn.fit(X_train, y_train)

# Fazer previsões e avaliar o modelo
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["0", "1"])

print(f"Acurácia: {accuracy:.2f}")
print("Relatório de Classificação:")
print(report)
