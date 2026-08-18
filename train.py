from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "breast_cancer_wisconsin_diagnostic.csv"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"
TEST_FILE = BASE_DIR / "test_data.csv"

TARGET = "diagnosis"
RANDOM_STATE = 42

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

test_df = X_test.copy()
test_df[TARGET] = y_test.values
test_df.to_csv(TEST_FILE, index=False)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=4, random_state=RANDOM_STATE
    ),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Gaussian Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Support Vector Machine (Additional 6th)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ]),
}

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "Support Vector Machine (Additional 6th)": "svm.joblib",
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, prob),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })

    joblib.dump(model, MODEL_DIR / model_files[name])

metrics = pd.DataFrame(results)
metrics.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

print("\nModel comparison:")
print(metrics.round(4).to_string(index=False))
print(f"\nTest data saved to: {TEST_FILE.name}")
print("Models saved in model/")
print("Metrics saved in results/model_metrics.csv")
