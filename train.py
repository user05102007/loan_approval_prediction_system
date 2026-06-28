import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("loan_approval_dataset.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("\nColumns:")
print(df.columns.tolist())

# Remove loan_id if present
if "loan_id" in df.columns:
    df.drop("loan_id", axis=1, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Encode categorical columns
education_encoder = LabelEncoder()
self_emp_encoder = LabelEncoder()
loan_encoder = LabelEncoder()

df["education"] = education_encoder.fit_transform(df["education"])
df["self_employed"] = self_emp_encoder.fit_transform(df["self_employed"])
df["loan_status"] = loan_encoder.fit_transform(df["loan_status"])

# Save target encoder
joblib.dump(loan_encoder, "models/label_encoder.pkl")

# Features and target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

print("\nTraining Features:")
print(X.columns.tolist())

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

best_model = None
best_f1 = 0

print("\nMODEL RESULTS")
print("=" * 50)

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{name}")
    print("-" * 30)
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(pre, 4))
    print("Recall   :", round(rec, 4))
    print("F1 Score :", round(f1, 4))

    if f1 > best_f1:
        best_f1 = f1
        best_model = model

# Save best model
joblib.dump(best_model, "models/best_loan_approval_model.pkl")

print("\nBest Model Saved Successfully!")

# Feature Importance
if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop Features:")
    print(importance)

    plt.figure(figsize=(10, 5))
    plt.bar(importance["Feature"], importance["Importance"])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")

print("\nTraining Completed Successfully!"