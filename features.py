import pandas as pd

print("Loading merged dataset...")
df = pd.read_csv("merged_weld_data.csv")

print("Dataset shape:", df.shape)

# Helper function to find signal columns
def get_cols(prefix):
    return [c for c in df.columns if c.startswith(prefix + " T-")]

current_cols = get_cols("Current")
force_cols = get_cols("Force")
voltage_cols = get_cols("Voltage")

print("Current columns:", len(current_cols))
print("Force columns:", len(force_cols))
print("Voltage columns:", len(voltage_cols))

# Start new feature table with IDs + label
features = df[["Car Body", "Welding Spot", "Date", "Fault"]].copy()

# ---- CURRENT FEATURES ----
X = df[current_cols]
features["Current_mean"] = X.mean(axis=1)
features["Current_max"] = X.max(axis=1)
features["Current_min"] = X.min(axis=1)
features["Current_std"] = X.std(axis=1)
features["Current_range"] = features["Current_max"] - features["Current_min"]

# ---- FORCE FEATURES ----
X = df[force_cols]
features["Force_mean"] = X.mean(axis=1)
features["Force_max"] = X.max(axis=1)
features["Force_min"] = X.min(axis=1)
features["Force_std"] = X.std(axis=1)
features["Force_range"] = features["Force_max"] - features["Force_min"]

# ---- VOLTAGE FEATURES ----
X = df[voltage_cols]
features["Voltage_mean"] = X.mean(axis=1)
features["Voltage_max"] = X.max(axis=1)
features["Voltage_min"] = X.min(axis=1)
features["Voltage_std"] = X.std(axis=1)
features["Voltage_range"] = features["Voltage_max"] - features["Voltage_min"]

# Save feature dataset
features.to_csv("weld_features.csv", index=False)

print("\nSaved weld_features.csv")
print("Feature dataset shape:", features.shape)
print("\nFault distribution:")
print(features["Fault"].value_counts())
