
import streamlit as st

import pandas as pd

import numpy as np

import joblib

import json

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (

    accuracy_score,

    roc_auc_score,

    precision_score,

    recall_score,

    f1_score,

    matthews_corrcoef,

    confusion_matrix,

    classification_report

)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(

    page_title="Dry Bean Classification",

    page_icon="🌱",

    layout="wide"

)

st.title("Dry Bean Classification Dashboard")

st.write(

    "Machine Learning Classification Model Evaluation"

)

st.write(

    "Compare five machine learning algorithms "

    "using six evaluation metrics."

)

# ==========================================
# LOAD METADATA
# ==========================================

with open("metadata.json", "r") as file:

    metadata = json.load(file)

target = metadata["target_column"]

features = metadata["feature_columns"]

# ==========================================
# MODEL FILES
# ==========================================

model_files = {

    "Logistic Regression":

    "model/logistic_regression.joblib",

    "Decision Tree":

    "model/decision_tree.joblib",

    "KNN":

    "model/knn.joblib",

    "Naive Bayes":

    "model/naive_bayes.joblib",

    "Support Vector Machine":

    "model/support_vector_machine.joblib"

}

# ==========================================
# MODEL SELECTION
# ==========================================

selected_model = st.selectbox(

    "Select Machine Learning Model",

    list(model_files.keys())

)

# ==========================================
# DATASET UPLOAD
# ==========================================

uploaded_file = st.file_uploader(

    "Upload Test Dataset (CSV)",

    type=["csv"]

)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(df.head())

    # Check required columns

    required_columns = features + [target]

    missing_columns = [

        col for col in required_columns

        if col not in df.columns

    ]

    if missing_columns:

        st.error(

            "Missing required columns: "

            + str(missing_columns)

        )

        st.stop()

    X_test = df[features]

    y_test = df[target]

    if y_test.isnull().any():

        st.error("Target column contains missing values.")

        st.stop()

    # ======================================
    # LOAD TRAINED MODEL
    # ======================================

    model = joblib.load(

        model_files[selected_model]

    )

    # ======================================
    # GENERATE PREDICTIONS
    # ======================================

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)

    # ======================================
    # CALCULATE METRICS
    # ======================================

    accuracy = accuracy_score(

        y_test,

        y_pred

    )

    auc = roc_auc_score(

        y_test,

        y_prob,

        labels=model.classes_,

        multi_class="ovr",

        average="weighted"

    )

    precision = precision_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )

    recall = recall_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )

    f1 = f1_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )

    mcc = matthews_corrcoef(

        y_test,

        y_pred

    )

    # ======================================
    # DISPLAY METRICS
    # ======================================

    st.subheader("Model Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy, 4))

    col2.metric("AUC", round(auc, 4))

    col3.metric("Precision", round(precision, 4))

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", round(recall, 4))

    col5.metric("F1 Score", round(f1, 4))

    col6.metric("MCC", round(mcc, 4))

    # ======================================
    # CONFUSION MATRIX
    # ======================================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(

        y_test,

        y_pred,

        labels=model.classes_

    )

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(cm)

    ax.set_xticks(range(len(model.classes_)))

    ax.set_yticks(range(len(model.classes_)))

    ax.set_xticklabels(model.classes_, rotation=45)

    ax.set_yticklabels(model.classes_)

    ax.set_xlabel("Predicted Class")

    ax.set_ylabel("Actual Class")

    ax.set_title("Confusion Matrix")

    fig.colorbar(image, ax=ax)

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(

                j,

                i,

                str(cm[i, j]),

                ha="center",

                va="center"

            )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # ======================================
    # CLASSIFICATION REPORT
    # ======================================

    st.subheader("Classification Report")

    report = classification_report(

        y_test,

        y_pred,

        output_dict=True,

        zero_division=0

    )

    st.dataframe(

        pd.DataFrame(report).transpose()

    )

    # ======================================
    # PREDICTION RESULTS
    # ======================================

    st.subheader("Prediction Results")

    output = df.copy()

    output["Predicted_Class"] = y_pred

    output["Correct_Prediction"] = (

        output[target].astype(str)

        == pd.Series(y_pred).astype(str).values

    )

    st.dataframe(output)

    # ======================================
    # DOWNLOAD PREDICTIONS
    # ======================================

    csv = output.to_csv(index=False)

    st.download_button(

        "Download Prediction Results",

        csv,

        file_name="predictions.csv",

        mime="text/csv"

    )

else:

    st.info(

        "Please upload your labelled test CSV to begin."

    )
