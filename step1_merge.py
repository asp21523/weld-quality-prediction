import pandas as pd

print("Loading CSVs...")

current = pd.read_csv("current.csv")
force = pd.read_csv("force.csv")
voltage = pd.read_csv("voltage.csv")
labels = pd.read_csv("labels.csv")

print("Shapes before merge:")
print(" current:", current.shape)
print(" force  :", force.shape)
print(" voltage:", voltage.shape)
print(" labels :", labels.shape)

# These are the ID columns used to match rows
keys = ["Car Body", "Welding Spot", "Date"]

# Check if all key columns exist (prevents silent errors)
for name, df in [("current", current), ("force", force), ("voltage", voltage), ("labels", labels)]:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"{name}.csv is missing columns: {missing}")

print("\nMerging current + force...")
df = current.merge(force, on=keys, how="inner")
print(" after merge 1:", df.shape)

print("\nMerging + voltage...")
df = df.merge(voltage, on=keys, how="inner")
print(" after merge 2:", df.shape)

print("\nMerging + labels...")
df = df.merge(labels, on=keys, how="inner")
print(" after merge 3:", df.shape)

# Quick validation
print("\nFault distribution:")
print(df["Fault"].value_counts(dropna=False))

# Save merged dataset
df.to_csv("merged_weld_data.csv", index=False)
print("\n✅ Saved: merged_weld_data.csv")