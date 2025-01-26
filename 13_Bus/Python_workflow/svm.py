from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVR  # For ν-SVR
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import pandas as pd
import time
import matplotlib.pyplot as plt  # To plot the confusion matrix

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
    X, y, test_size=0.3, random_state=80
)

# ==============================
# Parameters for ν-SVR
# ==============================
C = 1.0               # Complexity Bound
nu = 0.2              # ν parameter (controls the number of support vectors and error tolerance)
kernel = 'rbf'        # Kernel used ('linear', 'poly', 'rbf', 'sigmoid')
max_iter = 100        # Iteration limit (-1 for unlimited)
tol = 1e-3            # Numerical tolerance for optimization

# Create and train the ν-SVR model
svm = NuSVR(kernel=kernel, C=C, nu=nu, max_iter=max_iter, tol=tol)
svm.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = svm.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred.round())
report = classification_report(y_test, y_pred.round())

print(f"Mean Squared Error: {mse:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
