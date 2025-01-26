from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
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

# Decision tree parameter configuration
min_samples_leaf = 10  # Min. number of instances in leaves
min_samples_split = 10  # Do not split subsets smaller than this size
max_depth = 100  # Limit the maximal tree depth to this value
majority_stop = 99  # Stop when majority reaches [%]

# Create and train the decision tree model
dt = DecisionTreeClassifier(
    criterion='gini',
    min_samples_leaf=min_samples_leaf,
    min_samples_split=min_samples_split,
    max_depth=max_depth,
    random_state=80
)
dt.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Export the tree in text format
tree_rules = export_text(dt, feature_names=list(X.columns))
print("Decision Tree Rules:")
print(tree_rules)
