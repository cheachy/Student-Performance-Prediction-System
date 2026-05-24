# IS 108 - Business Intelligence| Final Project SY 2025-2026
# Student Performance Prediction System
# Algorithms: KNN, SVM, ANN

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Page setup
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="metric-container"] {
        background: #f7f7f7;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 12px;
    }
    button[data-baseweb="tab"] { font-size: 0.85rem; font-weight: 500; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("Student Performance Prediction System")
st.caption("IS 108 – Business Intelligence |  Final Project SY 2025-2026  |  Caraga State University")
st.divider()

# Chart style
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#eeeeee",
    "grid.linewidth":    0.8,
    "font.family":       "sans-serif",
    "axes.titleweight":  "bold",
    "axes.titlesize":    12,
})
COLORS = ["#222222", "#555555", "#999999"]

# Sidebar
st.sidebar.title("Model Settings")
st.sidebar.divider()
knn_k        = st.sidebar.slider("KNN — k (neighbors)", 1, 20, 5)
svm_c        = st.sidebar.selectbox("SVM — C (regularization)", [0.1, 1.0, 5.0, 10.0], index=1)
svm_kernel   = st.sidebar.selectbox("SVM — Kernel", ["rbf", "linear", "poly"])
ann_hidden   = st.sidebar.selectbox("ANN — Hidden Layers", [(64), (64, 32), (128, 64)], format_func=str)
ann_iter     = st.sidebar.slider("ANN — Max Iterations", 100, 500, 200, step=100)
test_percent = st.sidebar.slider("Test Size (%)", 10, 30, 20, step=5)

# Detect target column
def detect_target(df):
    """
    Automatically find the most likely target column.
    Checks known names first, then falls back to the last column.
    """
    known_targets = [
        "PerformanceCategory", "Exam_Score", "GPA", "Grade",
        "Performance", "Score", "Result", "Label", "Target", "Class"
    ]
    for name in known_targets:
        if name in df.columns:
            return name
    return df.columns[-1]

# Convert numeric target to categories
def bin_score_to_category(series):
    """
    If the target is numeric (e.g. exam score 0-100),
    convert it to Poor / Average / Good / Excellent.
    Bins are based on percentiles so every category always has enough samples.
    """
    p25 = series.quantile(0.25)
    p50 = series.quantile(0.50)
    p75 = series.quantile(0.75)
    mn  = series.min() - 1
    mx  = series.max() + 1

    bins   = [mn, p25, p50, p75, mx]
    labels = ["Poor", "Average", "Good", "Excellent"]
    return pd.cut(series, bins=bins, labels=labels).astype(str)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Dataset",
    "2. Preprocessing",
    "3. Train Models",
    "4. Evaluation",
    "5. Predict"
])

# TAB 1 – DATASET

with tab1:
    st.header("Step 1: Load Dataset")

    # Problem Identification ─
    with st.expander("Problem Identification", expanded=True):
        st.markdown("""
        Student academic performance is difficult to monitor proactively. By the time low grades
        are noticed, it is often too late to intervene effectively. This system predicts a student's
        performance category — Poor, Average, Good, or Excellent — early in the term, allowing
        educators to provide timely support to at-risk students.

        **Prediction Task:** Multi-class classification (4 categories)

        **Algorithms Used:** 
        - K-Nearest Neighbor (KNN) 
        - Support Vector Machine (SVM)
        - Artificial Neural Network (ANN)
        """)

    # Data Collection
    with st.expander("Data Collection", expanded=False):
        st.markdown("""
        **Dataset Source:** Kaggle — Student Performance Factors dataset (or uploaded CSV/Excel)

        **Collection Method:** Survey and academic records from educational institutions,
        covering student behavior, demographics, and academic history.

        **Format:** Structured tabular data (CSV), one row per student.

        Exam score converted to performance category:
        - Poor: bottom 25% of scores
        - Average: 25th to 50th percentile
        - Good: 50th to 75th percentile
        - Excellent: top 25% of scores
        """)
    st.divider()
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded:
        st.session_state.df = (pd.read_csv(uploaded)
                               if uploaded.name.endswith(".csv")
                               else pd.read_excel(uploaded))
        st.success("File uploaded successfully.")

    if "df" in st.session_state:
        df = st.session_state.df

        # Auto-detect and prepare target column
        target_col = detect_target(df)

        # If target is numeric, bin it into categories
        if pd.api.types.is_numeric_dtype(df[target_col]):
            df["PerformanceCategory"] = bin_score_to_category(df[target_col])
            df.drop(columns=[target_col], inplace=True)
            st.info(f"Column '{target_col}' was numeric — converted to categories: Poor / Average / Good / Excellent")
        elif target_col != "PerformanceCategory":
            df = df.rename(columns={target_col: "PerformanceCategory"})
            st.info(f"Using '{target_col}' as the target column.")

        st.session_state.df = df

        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Basic Information")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows",     df.shape[0])
        col2.metric("Total Columns",  df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)

        st.subheader("Target Variable Distribution")
        counts = df["PerformanceCategory"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(counts.index, counts.values, color="#222222", width=0.5)
        ax.set_xlabel("Performance Category")
        ax.set_ylabel("Number of Students")
        ax.set_title("Students per Performance Category")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# TAB 2 – PREPROCESSING

with tab2:
    st.header("Step 2: Data Preprocessing")

    if "df" not in st.session_state:
        st.warning("Please load a dataset in Tab 1 first.")
    else:
        st.markdown("""
        **Steps performed automatically:**
        1. Drop ID columns (StudentID, student_id, etc.) if present
        2. Fill missing values — median for numbers, mode for text
        3. Encode all text columns to numbers using Label Encoding
        4. Scale all features using StandardScaler
        5. Split into training and testing sets
        """)

        if st.button("Run Preprocessing", type="primary"):
            df = st.session_state.df.copy()

            # Step 1: Drop ID columns
            id_cols = [c for c in df.columns if c.lower() in ["studentid", "student_id", "id"]]
            df.drop(columns=id_cols, errors="ignore", inplace=True)

            # Step 2: Fill missing values
            # Check each column for nulls directly
            for col in df.columns:
                if col == "PerformanceCategory":
                    continue
                if df[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        fill_val = df[col].mode()[0] if df[col].notna().any() else "Unknown"
                        df[col] = df[col].fillna(fill_val)

            # Step 3: Encode all text/categorical columns
            # Check every column
            le_dict = {}
            for col in df.columns:
                if col == "PerformanceCategory":
                    continue
                # Encode if the column contains any string values
                if df[col].apply(lambda x: isinstance(x, str)).any():
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    le_dict[col] = le
            st.session_state.le_dict = le_dict

            # Step 4: Encode target label
            le_target = LabelEncoder()
            y = le_target.fit_transform(df["PerformanceCategory"])
            st.session_state.label_encoder = le_target

            # Step 5: Scale features
            feature_cols = [c for c in df.columns if c != "PerformanceCategory"]
            X = df[feature_cols].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            st.session_state.scaler       = scaler
            st.session_state.feature_cols = feature_cols

            # Step 6: Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_percent/100, random_state=42
            )
            st.session_state.X_train  = X_train
            st.session_state.X_test   = X_test
            st.session_state.y_train  = y_train
            st.session_state.y_test   = y_test
            st.session_state.df_clean = df
            st.success("Preprocessing complete.")

        if "df_clean" in st.session_state:
            st.subheader("Preprocessed Dataset")
            st.dataframe(st.session_state.df_clean.head(20), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Training Samples", st.session_state.X_train.shape[0])
            col2.metric("Test Samples",     st.session_state.X_test.shape[0])
            col3.metric("Features Used",    st.session_state.X_train.shape[1])

            # Feature Selection
            st.subheader("Feature Selection")
            st.markdown("""
            All available input features are used for training. The table below lists each feature,
            its type, and its role in predicting student performance.
            """)
            feature_cols = st.session_state.feature_cols
            le_dict      = st.session_state.get("le_dict", {})
            df_clean     = st.session_state.df_clean

            feat_rows = []
            for col in feature_cols:
                col_type = "Categorical (encoded)" if col in le_dict else "Numeric"
                feat_rows.append({"Feature": col.replace("_", " "), "Type": col_type, "Selected": "Yes"})
            st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)
            st.caption(f"Total features selected: {len(feature_cols)}")

# TAB 3 – TRAIN MODELS

with tab3:
    st.header("Step 3: Train KNN, SVM, and ANN")

    if "X_train" not in st.session_state:
        st.warning("Please complete preprocessing in Tab 2 first.")
    else:
        st.markdown(f"""
        **Current configurations:**
        - KNN — k = {knn_k} neighbors
        - SVM — kernel = {svm_kernel}, C = {svm_c}
        - ANN — hidden layers = {ann_hidden}, max iterations = {ann_iter}
        """)

        if st.button("Train All Models", type="primary"):
            X_train = st.session_state.X_train
            X_test  = st.session_state.X_test
            y_train = st.session_state.y_train
            y_test  = st.session_state.y_test

            results = {}

            # KNN
            with st.spinner("Training KNN..."):
                knn = KNeighborsClassifier(n_neighbors=knn_k)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                results["KNN"] = {
                    "model":     knn,
                    "y_pred":    y_pred,
                    "accuracy":  accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    "cm":        confusion_matrix(y_test, y_pred)
                }

            # SVM
            with st.spinner("Training SVM..."):
                svm = SVC(C=svm_c, kernel=svm_kernel, probability=True, random_state=42)
                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_test)
                results["SVM"] = {
                    "model":     svm,
                    "y_pred":    y_pred,
                    "accuracy":  accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    "cm":        confusion_matrix(y_test, y_pred)
                }

            # ANN
            with st.spinner("Training ANN..."):
                ann = MLPClassifier(hidden_layer_sizes=ann_hidden, max_iter=ann_iter,
                                    random_state=42, early_stopping=True)
                ann.fit(X_train, y_train)
                y_pred = ann.predict(X_test)
                results["ANN"] = {
                    "model":     ann,
                    "y_pred":    y_pred,
                    "accuracy":  accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    "cm":        confusion_matrix(y_test, y_pred)
                }

            st.session_state.results = results
            st.success("All 3 models trained successfully.")

        if "results" in st.session_state:
            r = st.session_state.results
            col1, col2, col3 = st.columns(3)
            col1.metric("KNN Accuracy", f"{r['KNN']['accuracy']*100:.2f}%")
            col2.metric("SVM Accuracy", f"{r['SVM']['accuracy']*100:.2f}%")
            col3.metric("ANN Accuracy", f"{r['ANN']['accuracy']*100:.2f}%")

# TAB 4 – EVALUATION

with tab4:
    st.header("Step 4: Model Evaluation and Comparison")

    if "results" not in st.session_state:
        st.warning("Please train the models in Tab 3 first.")
    else:
        r           = st.session_state.results
        le          = st.session_state.label_encoder
        class_names = le.classes_
        algos       = ["KNN", "SVM", "ANN"]

        # Metrics table
        st.subheader("Performance Metrics")
        summary = pd.DataFrame({
            "Algorithm":     algos,
            "Accuracy (%)":  [round(r[a]["accuracy"]  * 100, 2) for a in algos],
            "Precision (%)": [round(r[a]["precision"] * 100, 2) for a in algos],
            "Recall (%)":    [round(r[a]["recall"]    * 100, 2) for a in algos],
            "F1-Score (%)":  [round(r[a]["f1"]        * 100, 2) for a in algos],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        best = summary.loc[summary["Accuracy (%)"].idxmax(), "Algorithm"]
        st.info(f"Best model: {best} — {summary.loc[summary['Algorithm']==best, 'Accuracy (%)'].values[0]}% accuracy")

        # Accuracy bar chart
        st.subheader("Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(algos, [r[a]["accuracy"]*100 for a in algos], color=COLORS, width=0.4)
        for bar, val in zip(bars, [r[a]["accuracy"]*100 for a in algos]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.2f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy — KNN vs SVM vs ANN")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # All metrics grouped bar chart
        st.subheader("All Metrics Comparison")
        metrics       = ["accuracy", "precision", "recall", "f1"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
        x     = np.arange(len(metric_labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(9, 4))
        for i, (algo, color) in enumerate(zip(algos, COLORS)):
            vals = [r[algo][m]*100 for m in metrics]
            ax.bar(x + i*width, vals, width, label=algo, color=color)
        ax.set_xticks(x + width)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("Score (%)")
        ax.set_title("All Metrics — KNN vs SVM vs ANN")
        ax.legend()
        ax.set_ylim(0, 115)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Confusion matrices
        st.subheader("Confusion Matrices")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, algo in enumerate(algos):
            sns.heatmap(r[algo]["cm"], annot=True, fmt="d", cmap="Greys",
                        xticklabels=class_names, yticklabels=class_names,
                        ax=axes[i], linewidths=0.5, linecolor="#cccccc")
            axes[i].set_title(algo)
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")
        plt.suptitle("Confusion Matrices", fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# TAB 5 – PREDICT

with tab5:
    st.header("Step 5: Predict a New Student")

    if "results" not in st.session_state:
        st.warning("Please train the models in Tab 3 first.")
    else:
        feature_cols = st.session_state.feature_cols
        le_dict      = st.session_state.get("le_dict", {})

        st.markdown("Enter the student's details to predict their performance category.")
        st.divider()

        # Dynamically build input fields from the actual feature columns
        input_values = {}
        df_clean = st.session_state.df_clean

        cols = st.columns(2)
        for i, col_name in enumerate(feature_cols):
            col = cols[i % 2]
            original_col = df_clean[col_name]

            # If the column was originally categorical (stored in le_dict), show a selectbox
            if col_name in le_dict:
                le = le_dict[col_name]
                # Filter out "nan" in case any slipped through from missing values
                valid_options = [c for c in le.classes_ if str(c).lower() != "nan"]
                choice = col.selectbox(col_name.replace("_", " "), valid_options)
                input_values[col_name] = le.transform([choice])[0]
            else:
                # Numeric column — show a slider
                col_min  = float(df_clean[col_name].min())
                col_max  = float(df_clean[col_name].max())
                col_mean = float(df_clean[col_name].mean())
                # Use integer step if the column has no decimals
                step = 1.0 if df_clean[col_name].dtype in [np.int64, np.int32] else 0.5
                input_values[col_name] = col.slider(
                    col_name.replace("_", " "), col_min, col_max, col_mean, step
                )

        st.divider()
        algo_pick = st.selectbox("Algorithm to use", ["KNN", "SVM", "ANN"])

        if st.button("Predict", type="primary"):
            new_data = np.array([[input_values[c] for c in feature_cols]])
            new_data_scaled = st.session_state.scaler.transform(new_data)

            model      = st.session_state.results[algo_pick]["model"]
            pred_label = st.session_state.label_encoder.inverse_transform(
                             model.predict(new_data_scaled)
                         )[0]
            
            st.caption(f"Predicted using {algo_pick}")
            st.success(f"Predicted Performance Category: **{pred_label}**")

            tips = {
                "Excellent": "Outstanding performance. Maintain current habits.",
                "Good":      "Above average. Stay consistent and aim higher.",
                "Average":   "Satisfactory. Consider increasing study hours and attendance.",
                "Poor":      "Needs improvement. Seek tutoring, improve attendance, and study regularly."
            }
            st.info(tips.get(pred_label, "Prediction complete."))
