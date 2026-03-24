import pandas as pd
import mlflow
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
def load_processed_data() : 
    X_train=pd.read_csv("data/processed/X_train.csv")
    X_test=pd.read_csv("data/processed/X_test.csv")
    y_train=pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test=pd.read_csv("data/processed/y_test.csv").values.ravel()
    return X_train,X_test,y_train,y_test;
def train_logistic_regression(X_train,X_test,y_train,y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
     # probabilities for ROC-AUC
    y_proba = model.predict_proba(X_test)[:, 1]

    # metrics
    roc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("ROC-AUC:", roc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Confusion Matrix:\n", cm)

    return model, roc
def train_random_forest(X_train,X_test,y_train,y_test):
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("ROC-AUC:", roc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Confusion Matrix:\n", cm)
    
    return model, roc
def train_xgboost(X_train,X_test,y_train,y_test):
    model = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("ROC-AUC:", roc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Confusion Matrix:\n", cm)

    return model, roc
def run_training_pipeline():
    
    X_train, X_test, y_train, y_test = load_processed_data()
    import json

    stats = {}

    for col in X_train.columns:
        stats[col] = {
            "mean": float(X_train[col].mean()),
            "std": float(X_train[col].std())
        }

    with open("models/train_stats.json", "w") as f:
        json.dump(stats, f)

    models = {
        "logistic": train_logistic_regression,
        "random_forest": train_random_forest,
        "xgboost": train_xgboost
    }

    best_score = 0
    best_model = None

    for name, train_func in models.items():
        print(f"\nTraining model: {name}")
        with mlflow.start_run(run_name=name):

            model, roc = train_func(X_train, X_test, y_train, y_test)

            mlflow.log_metric("roc_auc", roc)

            if roc > best_score:
                best_score = roc
                best_model = model

    joblib.dump(best_model, "models/best_model.pkl")

    print("Best model saved with ROC-AUC:", best_score)
if __name__ == "__main__" :
    run_training_pipeline()