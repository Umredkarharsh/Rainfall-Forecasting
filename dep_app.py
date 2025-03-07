import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("Ridge Regression with SHAP Interpretability")
st.write("Upload a dataset to visualize predictions and SHAP-based feature importance.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower().str.strip()

    required_cols = ["year", "total", "jan", "feb", "mar", "april", "may", "june", "july", "aug", "sept", "oct", "nov", "dec"]

    if all(col in df.columns for col in required_cols):
        X = df[['jan', 'feb', 'mar', 'april', 'may', 'june', 'july', 'aug', 'sept', 'oct', 'nov', 'dec']]
        y = df['total']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Ridge()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        def evaluate_model(true, predicted):
            mae = mean_absolute_error(true, predicted)
            mse = mean_squared_error(true, predicted)
            rmse = np.sqrt(mse)
            r2 = r2_score(true, predicted)
            return mae, rmse, r2

        train_mae, train_rmse, train_r2 = evaluate_model(y_train, y_train_pred)
        test_mae, test_rmse, test_r2 = evaluate_model(y_test, y_test_pred)

        st.write("### Model Performance")
        st.write(f"**Train R² Score:** {train_r2:.4f}")
        st.write(f"**Test R² Score:** {test_r2:.4f}")
        st.write(f"**Train MAE Score:** {train_mae:.4f}")
        st.write(f"**Test MAE Score:** {test_mae:.4f}")

        # Actual vs. Predicted Plot
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(y_test, y_test_pred, alpha=0.5, label="Predicted", color="blue")
        ax1.scatter(y_test, y_test, alpha=0.5, label="Actual", color="orange")
        ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Ideal Fit")
        ax1.set_xlabel("Actual Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title("Actual vs. Predicted Values")
        ax1.legend()
        st.pyplot(fig1)

        # Residuals Plot
        residuals = y_test - y_test_pred
        #Create Residuals Plot
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        # Scatter plot for residuals (y_test_pred vs residuals)
        ax2.scatter(y_test_pred, residuals, color="yellow", alpha=0.6, label="Residuals")
        # Scatter plot for predicted values (y_test_pred alone)
        ax2.scatter(y_test_pred, [0] * len(y_test_pred), color="blue", alpha=0.5, label="Predicted Values")
        # Add reference line
        ax2.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="Zero Error Line")

        ax2.set_xlabel("Predicted Values")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residuals vs. Predicted Values (Model Performance Check)")

        ax2.legend()
        st.pyplot(fig2)

        # Histogram of Residuals
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.hist(residuals, bins=20, color="purple", edgecolor="black", alpha=0.7)
        ax3.set_xlabel("Residuals")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Histogram of Residuals")
        st.pyplot(fig3)

        # Q-Q Plot (Normality Check)
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title("Q-Q Plot for Residuals")
        st.pyplot(fig4)

        # SHAP Analysis
        st.write("## SHAP (Feature Importance) Analysis")

        explainer = shap.KernelExplainer(model.predict, X_train)
        shap_values = explainer.shap_values(X_test)

        # SHAP Summary Plot
        st.write("### SHAP Summary Plot")
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig5)

        #  SHAP Decision Plot
        st.write("### SHAP Decision Plot")
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        shap.decision_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], show=False)
        st.pyplot(fig7)

    else:
        st.error("Dataset must contain 'year', 'total', and monthly columns ('jan' to 'dec').")
