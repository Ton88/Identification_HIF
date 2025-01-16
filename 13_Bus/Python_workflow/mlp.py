from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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

# Parâmetros do modelo MLP
hidden_layer_sizes = (50,)  # Neurônios nas camadas ocultas (ex.: (100,) para uma camada com 100 neurônios)
activation = 'tanh'          # Função de ativação (ex.: 'identity', 'logistic', 'tanh', 'relu')
solver = 'lbfgs'              # Otimizador (ex.: 'lbfgs', 'sgd', 'adam')
alpha = 0.6               # Taxa de regularização L2
max_iter = 100               # Número máximo de iterações

# Criar e treinar o modelo MLP
mlp = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    solver=solver,
    alpha=alpha,
    max_iter=max_iter,
    random_state=80
)
mlp.fit(X_train, y_train)

# Fazer previsões e avaliar o modelo
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["0", "1"])

print(f"Acurácia: {accuracy:.2f}")
print("Relatório de Classificação:")
print(report)

# Parâmetros ajustáveis estão comentados e podem ser modificados conforme necessário.
