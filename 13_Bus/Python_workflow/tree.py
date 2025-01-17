from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score
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
    X, y, test_size=0.3, random_state=80, stratify=y
)

# Configuração dos parâmetros da árvore de decisão
min_samples_leaf = 5  # Min. Number of Instances in Leaves
min_samples_split = 10  # Do not Split Subsets Smaller Than
max_depth = 10  # Limit the Maximal Tree Depth to
majority_stop = 95  # Stop When Majority Reaches [%]

# Criar e treinar o modelo da árvore de decisão
dt = DecisionTreeClassifier(
    criterion='gini',
    min_samples_leaf=min_samples_leaf,
    min_samples_split=min_samples_split,
    max_depth=max_depth,
    random_state=80
)
dt.fit(X_train, y_train)

# Fazer previsões e avaliar o modelo
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Exibir resultados
print(f"Acurácia: {accuracy:.2f}")
print("Relatório de Classificação:")
print(report)

# Exportar a árvore em formato de texto
tree_rules = export_text(dt, feature_names=list(X.columns))
print("Regras da Árvore de Decisão:")
print(tree_rules)

# Parâmetros ajustáveis no código:
# - min_samples_leaf: Número mínimo de instâncias em folhas.
# - min_samples_split: Não dividir subconjuntos menores que esse valor.
# - max_depth: Profundidade máxima da árvore.
# - majority_stop: Parada quando a maioria atinge [%].
