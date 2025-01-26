from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Load the data
data = pd.read_csv('/data/Bus_34_mexh.csv', delimiter=';', engine='python')

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

# KNN model parameters
n_neighbors = 8  # Number of neighbors
weights = 'distance'  # Weights (uniform or distance)
p = 2  # Minkowski distance (p=2 is equivalent to Euclidean distance)

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
knn.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["0", "1"])

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
