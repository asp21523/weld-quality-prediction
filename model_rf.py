import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("weld_features.csv")

X = df.drop(columns=["Car Body", "Welding Spot", "Date", "Fault"])
y = df["Fault"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Random Forest: good for nonlinear relationships, no scaling needed
rf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced_subsample"
)

rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred, digits=3))

# Feature importance (risk drivers)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Risk Drivers (Feature Importance):")
print(importances.head(10))