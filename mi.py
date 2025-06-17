import streamlit as st
import pandas as pd
from eda_module import DataExplorer, StrokeModelTrainer, ColumnEditor

st.set_page_config(page_title="Stroke Analysis Dashboard", layout="wide")
st.title("Stroke Risk Data Analysis Platform")

# Sidebar - Upload or load file
st.sidebar.header("Dataset Loader")
uploaded_file = st.sidebar.file_uploader("Upload Stroke Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Or use the default 'stroke_dataset.csv'")
    try:
        df = pd.read_csv("stroke_dataset.csv")
    except FileNotFoundError:
        df = None

# Sidebar - Navigation Menu
st.sidebar.header(" Dashboard Navigation")
option = st.sidebar.selectbox(
    "Choose a Functionality",
    [
        "Home",
        "Data Overview",
        "Handle Missing Values",
        "Class Distribution",
        "Balance Data",
        "Train ALL ML Models",
        "Column Manipulation",
        "Outlier Detection",
        "Evaluate Model with Confusion Matrix",
        "Visualize Data", 
        "Compute Risk Score",
        "Save Dataset"
    ]
)

if df is not None:
    explorer = DataExplorer(df)
    trainer = StrokeModelTrainer(df)
    editor = ColumnEditor(df)

    if option == "Home":
        st.markdown("""
        ## 🧠 Stroke Risk Data Analysis Platform

        Welcome to the interactive dashboard for exploring stroke prediction data.

        This platform helps you:
        - Perform Exploratory Data Analysis (EDA)
        - Handle missing values
        - Balance classes using SMOTE
        - Train machine learning models
        - Visualize class distributions and outliers
        - Compute individual risk scores

        Use the sidebar to navigate and select different functionalities.
        """)

    elif option == "Data Overview":
        st.subheader("📊 Data Preview")
        st.dataframe(df)
        st.subheader("📌 Statistical Summary")
        st.dataframe(explorer.statisticalSummary())

    elif option == "Handle Missing Values":
        st.subheader("🧹 Handle Missing Values")
        df_handled = explorer.handleMissingValues()
        st.success("Missing values handled.")
        st.dataframe(df_handled)

    elif option == "Class Distribution":
        target = st.selectbox("Select Target Column", df.columns)
        st.subheader(f"🎯 Class Distribution of '{target}'")
        explorer.classDistribution(target)

    elif option == "Balance Data":
        target = st.selectbox("Select Target Column to Balance", df.columns)
        if st.button("Balance Data using SMOTE"):
            df_balanced = explorer.balanceData(target)
            st.success(f"Data balanced on target '{target}'.")
            st.dataframe(df_balanced)

    elif option == "Train ALL ML Models":
        st.subheader("🧠 Train Models for All Targets")
        if st.button("Train Models"):
            trainer.trainAllTargets()

    elif option == "Column Manipulation":
        st.subheader("🧰 Feature Engineering")

        col_action = st.radio("Select action", ["Add Column", "Remove Column", "Rename Column"])

        if col_action == "Add Column":
            new_col = st.text_input("New column name")
            default_val = st.text_input("Default value (optional)")
            if st.button("Add Column"):
                msg = editor.addColumn(new_col, default_val if default_val != "" else None)
                st.success(msg)
                st.dataframe(editor.getDataframe())

        elif col_action == "Remove Column":
            rem_col = st.selectbox("Select column to remove", df.columns)
            if st.button("Remove Column"):
                msg = editor.removeColumn(rem_col)
                st.success(msg)
                st.dataframe(editor.getDataframe())

        elif col_action == "Rename Column":
            old_name = st.selectbox("Select column to rename", df.columns)
            new_name = st.text_input("New column name")
            if st.button("Rename Column"):
                msg = editor.renameColumn(old_name, new_name)
                st.success(msg)
                st.dataframe(editor.getDataframe())

    elif option == "Outlier Detection":
        col = st.selectbox("Select column to detect outliers", df.select_dtypes(include='number').columns)
        if st.button("Show Outlier Boxplot"):
            explorer.detect_outliers(col)

    elif option == "Evaluate Model with Confusion Matrix":
        target = st.selectbox("Select Target Column for Evaluation", df.columns)
        if st.button("Evaluate Models"):
            trainer.evaluateEachTarget(target)

    elif option == "Visualize Data":
        explorer.visualize()

    elif option == "Compute Risk Score":
        if st.button("Compute and Save Risk Score"):
            explorer.compute_risk_score()
            explorer.saveToFile()
            st.success("Risk score computed and dataset saved.")
            # Show some relevant columns + risk score
            cols_to_show = ['age', 'average_glucose_level', 'hypertension', 'heart_disease', 'computed_risk_score']
            available_cols = [c for c in cols_to_show if c in df.columns]
            st.dataframe(explorer.df[available_cols].head())

    elif option == "Save Dataset":
        if st.button("Save Current Dataset"):
            explorer.saveToFile()
            st.success("Dataset saved successfully.")

else:
    st.warning("Please upload a dataset CSV or ensure 'stroke_dataset.csv' is in the working directory.")
