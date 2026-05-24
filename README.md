# Student Performance Prediction System
## IS 108 – Business Intelligence | Final Project SY 2025-2026
### Caraga State University – College of Computing and Information Sciences

## Files Included

| File                            | Description                               |
|---------------------------------|-------------------------------------------|
| `app.py`                        | Main Streamlit application (all 5 tabs)   |
| `StudentPerformanceFactors.csv` | Sample dataset                            |
| `requirements.txt`              | Python dependencies                       |
| `README.md`                     | This file                                 |

## Requirements

- Python 3.8 or higher
- pip

## How to Run

### Step 1 – Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 – Run the application
```bash
streamlit run app.py
```

### Step 3 – Open in browser
The app will automatically open at: **http://localhost:8501**

## Application Workflow

| Tab   | Step          | Description                                                           |
|-------|---------------|-----------------------------------------------------------------------|
| Tab 1 | Dataset       | Load CSV/Excel or use built-in sample. View data, stats, distribution |
| Tab 2 | Preprocessing | Handle missing values, encode categories, scale features, split data  |
| Tab 3 | Training      | Train KNN, SVM, and ANN with adjustable hyperparameters               |
| Tab 4 | Evaluation    | Compare accuracy, precision, recall, F1, confusion matrices           |
| Tab 5 | Prediction    | Enter new student data and predict performance category               |

## Algorithms

- **KNN** – K-Nearest Neighbor (k adjustable, default=5)
- **SVM** – Support Vector Machine (kernel + C adjustable)
- **ANN** – Artificial Neural Network / MLP (layers + iterations adjustable)

---

## Target Variable

`PerformanceCategory`: **Poor | Average | Good | Excellent**

---

## Evaluation Metrics

Accuracy · Precision · Recall · F1-Score · Confusion Matrix

All metrics use **weighted averaging** across all 4 classes.
