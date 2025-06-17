import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, roc_curve, auc
)

class DataExplorer:
    def __init__(self, df):
        self.df = df

    def handleMissingValues(self):
        self.df.fillna(self.df.mean(numeric_only=True), inplace=True)
        self.df.fillna(self.df.mode().iloc[0], inplace=True)
        missingcounts = self.df.isnull().sum()
        missingcounts = missingcounts[missingcounts > 0]

        if missingcounts.empty:
            st.success("✅ No missing values found.")
        else:
            st.warning("⚠️ Missing values were found and handled:")
            st.write(missingcounts)

        return self.df

    def statisticalSummary(self):
        return self.df.describe(include='all')

    def classDistribution(self, target):
        st.write(f"📊 Distribution of '{target}'")
        fig, ax = plt.subplots()
        sns.countplot(x=target, data=self.df, ax=ax)
        ax.set_title(f'Distribution of {target}')
        st.pyplot(fig)

    def balanceData(self, target):
        X = self.df.drop(target, axis=1)
        y = self.df[target]
        X = pd.get_dummies(X, drop_first=True)
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        balanced_df = pd.concat([
            pd.DataFrame(X_resampled, columns=X.columns),
            pd.DataFrame(y_resampled, columns=[target])
        ], axis=1)
        st.success("✅ Data balanced using SMOTE.")
        return balanced_df

    def detect_outliers(self, column):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=self.df, x=column, ax=ax)
        ax.set_title(f"Outlier Detection for {column}")
        st.pyplot(fig)

    def compute_risk_score(self):
        self.df.fillna(0, inplace=True)
        self.df['rawScore'] = (
            0.4 * self.df['Age'] +
            0.3 * self.df['Average Glucose Level'] +
            0.2 * self.df['Hypertension'] +
            0.1 * self.df['Heart Disease']
        )

        min_score = self.df['rawScore'].min()
        max_score = self.df['rawScore'].max()
        self.df['computed_risk_score'] = (
            (self.df['rawScore'] - min_score) / (max_score - min_score)
        ) * 100

        self.df['computed_risk_score'] = self.df['computed_risk_score'].round(2)
        self.saveToFile()
        st.success("✅ Risk score computed and dataset saved to file.")

    def saveToFile(self, filename="updated_stroke_data.csv"):
        try:
            self.df.to_csv(filename, index=False)
            st.success(f"💾 Dataset saved to `{filename}` successfully.")
        except Exception as e:
            st.error(f"❌ Error saving file: {e}")
    def visualize(self):
        st.subheader(" Column Visualization")

        if self.df is None or self.df.empty:
            st.warning("DataFrame is empty.")
            return

        column = st.selectbox("Select column to visualize", self.df.columns)
        plotType = st.radio(
            "Choose plot type:",
            ["Bar Plot", "Pie Chart", "Box Plot", "Scatter Plot"]
        )

        fig, ax = plt.subplots()

        if plotType == "Bar Plot":
            self.df[column].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f"Bar Plot of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        elif plotType == "Pie Chart":
            self.df[column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_ylabel("")
            ax.set_title(f"Pie Chart of {column}")
            st.pyplot(fig)

        elif plotType == "Box Plot":
            sns.boxplot(y=self.df[column], ax=ax)
            ax.set_title(f"Box Plot of {column}")
            st.pyplot(fig)

        elif plotType == "Scatter Plot":
            numeric_cols = self.df.select_dtypes(include='number').columns
            if column not in numeric_cols:
                st.warning(f"'{column}' is not numeric. Scatter plot requires numeric columns.")
                return
            y_col = st.selectbox("Select Y-axis column", numeric_cols)
            fig2, ax2 = plt.subplots()
            sns.scatterplot(x=self.df[column], y=self.df[y_col], ax=ax2)
            ax2.set_title(f"Scatter Plot: {column} vs {y_col}")
            ax2.set_xlabel(column)
            ax2.set_ylabel(y_col)
            st.pyplot(fig2)


class ColumnEditor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def addColumn(self, columnname, defaultValue=None):
        if columnname in self.df.columns:
            return f"⚠️ Column '{columnname}' already exists."
        self.df[columnname] = defaultValue
        return f"✅ Column '{columnname}' added."

    def removeColumn(self, columnname):
        if columnname in self.df.columns:
            self.df.drop(columns=[columnname], inplace=True)
            return f"✅ Column '{columnname}' removed."
        return f"⚠️ Column '{columnname}' not found."

    def renameColumn(self, oldname, newname):
        if oldname not in self.df.columns:
            return f"⚠️ Column '{oldname}' not found."
        if newname in self.df.columns:
            return f"⚠️ Column '{newname}' already exists."
        self.df.rename(columns={oldname: newname}, inplace=True)
        return f"✅ Column '{oldname}' renamed to '{newname}'."

    def getDataframe(self):
        return self.df

    def saveToFile(self, filename="updated_stroke_data.csv"):
        try:
            self.df.to_csv(filename, index=False)
            st.success(f"💾 Dataset saved to '{filename}' successfully.")
        except Exception as e:
            st.error(f"❌ Error saving file: {e}")

class StrokeModelTrainer:
    def __init__(self, df):
        self.df = df

    def trainModel(self, X, y, model, modelname, target_name="", plot_conf_matrix=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader(f"{modelname} Results for '{target_name}'")
        st.text(classification_report(y_test, y_pred))

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        if plot_conf_matrix:
            self.plotConfusionMatrix(y_test, y_pred, model_name=f"{modelname} - {target_name}")

            if len(np.unique(y)) == 2 and hasattr(model, "predict_proba"):
                self.plotRocCurve(model, X_test, y_test, model_name=f"{modelname} - {target_name}")

        return {
            'Model': modelname,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'Target': target_name
        }

    def trainAllTargets(self):
        targets = ['Chronic Stress', 'Income Level', 'Physical Activity', 'Stroke Occurrence']
        classifiers = {
            'Random Forest': RandomForestClassifier(),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier()
        }

        all_results = []

        for target in targets:
            if target not in self.df.columns:
                st.warning(f"⚠️ Target '{target}' not found in dataset.")
                continue

            X = self.df.drop(columns=[target])
            y = self.df[target]
            X = pd.get_dummies(X, drop_first=True)

            for modelname, model in classifiers.items():
                metrics = self.trainModel(X, y, model, modelname, target_name=target)
                all_results.append(metrics)

        results_df = pd.DataFrame(all_results)
        st.subheader("📈 Model Performance Summary")
        st.dataframe(results_df)
        self.plotMetrics(results_df)

    def evaluateEachTarget(self, target):
        if target not in self.df.columns:
            st.error("❌ Invalid target column name.")
            return

        X = self.df.drop(columns=[target])
        y = self.df[target]
        X = pd.get_dummies(X, drop_first=True)

        classifiers = {
            'Random Forest': RandomForestClassifier(),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier()
        }

        for name, model in classifiers.items():
            self.trainModel(X, y, model, name, target_name=target, plot_conf_matrix=True)

    def plotMetrics(self, df):
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=df, x='Target', y=metric, hue='Model', ax=ax)
            ax.set_title(f'Model Comparison by {metric}')
            ax.set_ylim(0, 1)
            st.pyplot(fig)

    @staticmethod
    def plotConfusionMatrix(y_test, y_pred, model_name="Model"):
        cm = confusion_matrix(y_test, y_pred)
        labels = np.unique(np.concatenate((y_test, y_pred)))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"{model_name} - Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    @staticmethod
    def plotRocCurve(model, X_test, y_test, model_name="Model"):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_title(f"{model_name} - ROC Curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)
