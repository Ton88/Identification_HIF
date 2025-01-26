from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Random Forest model parameters
n_estimators = 100  # Number of trees in the forest
min_samples_split = 2  # Do not split subsets smaller than this size

# Create and train the Random Forest model
rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=80)
rf.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
