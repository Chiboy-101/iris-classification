import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


# Load the dataset
df = pd.read_csv("Iris.csv")

# Drop duplicates
print(df.isnull().sum())  # Check formissing values
df.drop_duplicates(inplace=True)

# Check for outliers
for col in ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]:
    df.boxplot(col)
    plt.show()

# Scale numeric columns
scaler = StandardScaler()
cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
df[cols] = scaler.fit_transform(df[cols])

# Encode the target variable
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])  # type: ignore

# Define features and target
X = df.drop(columns=["Id", "Species"])
y = df["Species"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train[cols] = scaler.fit_transform(X_train[cols])
X_test[cols] = scaler.transform(X_test[cols])

# Train the model
# Create a list of models to evaluate

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

# Train and evaluate each model
for model_name, model in models.items():
    # Train each model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Evaluate the model
    print(f"{model_name} Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix for {model_name}:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 50)

# Hyper parameter tuning for Logisitc Regression
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Logistic Regression
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"],
    "max_iter": [100, 1000, 2500],
}

# Create a GridSearchCV object
grid_search = GridSearchCV(
    LogisticRegression(random_state=42), param_grid, cv=5, scoring="accuracy"
)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters for Logistic Regression:", grid_search.best_params_)
print("Best Accuracy for Logistic Regression:", grid_search.best_score_)

# Train the best Logistic Regression model
best_logistic_model = grid_search.best_estimator_
best_logistic_model.fit(X_train, y_train)

# Make predictions with the best Logistic Regression model
y_pred_logistic = best_logistic_model.predict(X_test)

# Evaluate the best Logistic Regression model
print(
    f"Best Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_logistic):.3f}"
)
print("Classification Report for Best Logistic Regression:")
print(classification_report(y_test, y_pred_logistic))
print("Confusion Matrix for Best Logistic Regression:")
print(confusion_matrix(y_test, y_pred_logistic))

# Save the best model
with open("best_logistic_model.pkl", "wb") as f:
    pickle.dump(best_logistic_model, f)
    
# Can load the already saved model and predict without training
