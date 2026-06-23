# Weld Quality Prediction using Random Forest

## Overview

This project predicts the quality of resistance spot welds in automotive body manufacturing using machine learning. Process parameters such as **Current**, **Force**, and **Voltage** are analyzed to determine whether a weld is **OK** or **NOT OK**.

A Random Forest classifier is trained on engineered statistical features extracted from welding signals to identify defective welds and estimate their risk.

---

## Project Workflow

```
Raw Welding Data
(Current, Force, Voltage, Labels)
            │
            ▼
     step1_merge.py
            │
            ▼
   merged_weld_data.csv
            │
            ▼
    step2_features.py
            │
            ▼
    weld_features.csv
            │
            ├──────────────► model_rf.py
            │                     │
            │                     ▼
            │           Random Forest Model
            │
            ▼
    dashboard_data.py
            │
            ▼
    Risk Analysis & Dashboard Files
```

---

## Dataset

The project uses four input datasets:

| File | Description |
|----------|----------------------------|
| current.csv | Welding current measurements |
| force.csv | Welding force measurements |
| voltage.csv | Welding voltage measurements |
| labels.csv | Weld quality labels (OK / NOT OK) |

---

## Feature Engineering

For every weld, statistical features are extracted from the process signals.

### Current Features

- Mean
- Maximum
- Minimum
- Standard Deviation
- Range

### Force Features

- Mean
- Maximum
- Minimum
- Standard Deviation
- Range

### Voltage Features

- Mean
- Maximum
- Minimum
- Standard Deviation
- Range

Total numerical features: **15**

---

## Machine Learning Model

**Algorithm:** Random Forest Classifier

Model Parameters:

- n_estimators = 400
- random_state = 42
- class_weight = balanced_subsample

The model predicts:

- Weld Quality (OK / NOT OK)
- Risk Score (Probability of Defect)

---

## Project Files

```
step1_merge.py
```

Merges Current, Force, Voltage, and Label datasets into a single dataset.

```
step2_features.py
```

Performs feature engineering by extracting statistical parameters from the welding signals.

```
model_rf.py
```

Trains the Random Forest classifier using the engineered features.

```
step3_model_logreg.py
```

Alternative Logistic Regression implementation for comparison.

```
dashboard_data.py
```

Generates risk scores and dashboard reports.

---

## Output Files

### weld_scored.csv

Contains:

- Weld information
- Engineered features
- Predicted Risk Score

### risk_by_spot.csv

Provides:

- Fault Rate by Welding Spot
- Average Risk Score
- Number of Welds

Useful for identifying high-risk welding locations.

### risk_trend.csv

Provides:

- Fault Rate over time
- Average Risk over time

Useful for monitoring welding quality trends.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- OpenPyXL

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1

Merge raw data

```bash
python step1_merge.py
```

### Step 2

Generate features

```bash
python step2_features.py
```

### Step 3

Train Random Forest model

```bash
python model_rf.py
```

### Step 4

Generate dashboard reports

```bash
python dashboard_data.py
```

---

## Applications

- Automotive Manufacturing
- Resistance Spot Welding Quality Control
- Predictive Quality Analytics
- Manufacturing Process Monitoring
- Defect Detection using Machine Learning

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV
- Real-time prediction dashboard
- Feature importance visualization
- Model deployment using Flask/FastAPI
- Interactive dashboards using Power BI or Streamlit

---

## Author

**Adabala Sai Pranav**

Machine Learning | Data Analytics | Manufacturing Quality Engineering
