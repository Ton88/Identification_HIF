from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Load the data
data = pd.read_csv('/mnt/Bus_13_mexh.csv', delimiter=';', engine='python')

# Split the data into features (X) and target (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # Only the last column

# Convert data to numeric values
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)  # Non-numeric values are converted to 0
y = pd.to_numeric(y, errors='coerce').fillna(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=80, stratify=y
)

# MLP model parameters
hidden_layer_sizes = (50,)  # Neurons in hidden layers (e.g., (100,) for one layer with 100 neurons)
activation = 'tanh'          # Activation function (e.g., 'identity', 'logistic', 'tanh', 'relu')
solver = 'lbfgs'              # Optimizer (e.g., 'lbfgs', 'sgd', 'adam')
alpha = 0.6               # L2 regularization rate
max_iter = 100               # Maximum number of iterations

# Create and train the MLP model
mlp = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    solver=solver,
    alpha=alpha,
    max_iter=max_iter,
    random_state=80
)
mlp.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["0", "1"])

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

