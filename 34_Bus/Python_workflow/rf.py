from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Carregar os dados
data = pd.read_csv('/mnt/Bus_13_mexh.csv', delimiter=';', engine='python')

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

# Parâmetros do modelo Random Forest
n_estimators = 100  # Número de árvores na floresta
min_samples_split = 2  # Não dividir subconjuntos menores que este tamanho

# Criar e treinar o modelo Random Forest
rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=80)
rf.fit(X_train, y_train)

# Fazer previsões e avaliar o modelo
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Acurácia: {accuracy:.2f}")
print("Relatório de Classificação:")
print(report)
