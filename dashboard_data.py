import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("weld_features.csv")

X = df.drop(columns=["Car Body", "Welding Spot", "Date", "Fault"])
y = df["Fault"]

# Train RF on full dataset (for scoring)
rf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced_subsample"
)

rf.fit(X, y)

# Predict probability of defect
df["RiskScore"] = rf.predict_proba(X)[:,1]

# --- Risk by welding spot ---
spot_risk = (
    df.groupby("Welding Spot")
      .agg(FaultRate=("Fault","mean"),
           AvgRisk=("RiskScore","mean"),
           Count=("Fault","size"))
      .sort_values("AvgRisk", ascending=False)
)

spot_risk.to_excel("risk_by_spot.xlsx")

# --- Risk trend over time ---
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

trend = (
    df.groupby(df["Date"].dt.to_period("M"))
      .agg(FaultRate=("Fault","mean"),
           AvgRisk=("RiskScore","mean"))
)

trend.to_excel("risk_trend.xlsx")

# Save scored dataset
df.to_excel("weld_scored.xlsx", index=False)

print("Saved:")
print(" - weld_scored.xlsx")
print(" - risk_by_spot.xlsx")
print(" - risk_trend.xlsx")