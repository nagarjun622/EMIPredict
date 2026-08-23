import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
DATA = ROOT / "data"

st.set_page_config(page_title="EMIPredict AI", page_icon="💳", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(ARTIFACTS / "emi_predict_models.joblib")


@st.cache_data
def load_data():
    return pd.read_csv(DATA / "dashboard_sample.csv")


@st.cache_data
def load_summary():
    return json.loads((RESULTS / "dashboard_summary.json").read_text(encoding="utf-8"))


def engineer_features(frame):
    result = frame.copy()
    expenses = ["monthly_rent", "school_fees", "college_fees", "travel_expenses", "groceries_utilities", "other_monthly_expenses"]
    result["total_monthly_expenses"] = result[expenses].sum(axis=1)
    income = result["monthly_salary"].replace(0, np.nan)
    result["debt_to_income"] = result["current_emi_amount"] / income
    result["expense_to_income"] = result["total_monthly_expenses"] / income
    result["income_after_expenses"] = result["monthly_salary"] - result["total_monthly_expenses"] - result["current_emi_amount"]
    result["financial_cushion"] = result["bank_balance"] + result["emergency_fund"]
    result["requested_payment_burden"] = result["requested_amount"] / result["requested_tenure"].replace(0, np.nan)
    result["loan_to_income"] = result["requested_amount"] / income
    result["employment_stability"] = result["years_of_employment"] / result["age"].replace(0, np.nan)
    result["salary_credit_interaction"] = result["monthly_salary"] * result["credit_score"]
    result["salary_employment_interaction"] = result["monthly_salary"] * result["years_of_employment"]
    result["balance_emergency_interaction"] = result["bank_balance"] * result["emergency_fund"]
    return result


bundle = load_bundle()
summary = load_summary()
st.title("EMIPredict AI — Financial Risk Assessment")
st.caption("Real-time EMI eligibility and maximum safe EMI estimates")

page = st.sidebar.radio("Navigate", ["Dashboard", "Predict", "Data Explorer", "Model Performance"])

if page == "Dashboard":
    data = load_data()
    c1, c2, c3 = st.columns(3)
    c1.metric("Sample profiles", f"{len(data):,}")
    c2.metric("Best classifier", summary["best_classification_model"])
    c3.metric("Best regressor", summary["best_regression_model"])
    st.subheader("Eligibility distribution")
    st.bar_chart(data["emi_eligibility"].value_counts())
    st.subheader("Financial snapshot")
    st.dataframe(data[["monthly_salary", "credit_score", "requested_amount", "max_monthly_emi", "emi_eligibility"]].head(20), use_container_width=True)

elif page == "Predict":
    st.subheader("Applicant details")
    sample = load_data().iloc[0]
    with st.form("prediction_form"):
        numeric = {}
        left, right = st.columns(2)
        fields = ["age", "monthly_salary", "years_of_employment", "monthly_rent", "family_size", "dependents", "school_fees", "college_fees", "travel_expenses", "groceries_utilities", "other_monthly_expenses", "current_emi_amount", "credit_score", "bank_balance", "emergency_fund", "requested_amount", "requested_tenure"]
        for index, field in enumerate(fields):
            container = left if index % 2 == 0 else right
            numeric[field] = container.number_input(field.replace("_", " ").title(), value=float(sample[field]), min_value=0.0)
        categories = {}
        categorical = ["gender", "marital_status", "education", "employment_type", "company_type", "house_type", "existing_loans", "emi_scenario"]
        for field in categorical:
            options = sorted(load_data()[field].dropna().unique().tolist())
            categories[field] = st.selectbox(field.replace("_", " ").title(), options, index=options.index(sample[field]) if sample[field] in options else 0)
        submitted = st.form_submit_button("Assess EMI risk", type="primary")
    if submitted:
        input_frame = pd.DataFrame([{**numeric, **categories}])
        features = engineer_features(input_frame)[bundle["feature_columns"]]
        processed = bundle["preprocessor"].transform(features)
        class_id = bundle["classifier"].predict(processed)[0]
        eligibility = bundle["label_encoder"].inverse_transform([class_id])[0]
        probability = bundle["classifier"].predict_proba(processed).max() * 100
        max_emi = bundle["regressor"].predict(processed)[0]
        a, b, c = st.columns(3)
        a.metric("Eligibility", eligibility)
        b.metric("Prediction confidence", f"{probability:.1f}%")
        c.metric("Maximum safe monthly EMI", f"₹{max_emi:,.0f}")

elif page == "Data Explorer":
    data = load_data()
    scenario = st.multiselect("EMI scenario", sorted(data["emi_scenario"].unique()), default=sorted(data["emi_scenario"].unique()))
    filtered = data[data["emi_scenario"].isin(scenario)]
    st.dataframe(filtered, use_container_width=True, height=500)
    st.download_button("Download filtered sample", filtered.to_csv(index=False), "emi_dashboard_sample.csv", "text/csv")

else:
    st.subheader("Validation and test performance")
    class_metrics = pd.read_csv(RESULTS / "classification_test_metrics.csv")
    reg_metrics = pd.read_csv(RESULTS / "regression_test_metrics.csv")
    st.dataframe(class_metrics, use_container_width=True)
    st.dataframe(reg_metrics, use_container_width=True)
    st.subheader("Model comparison")
    st.dataframe(pd.read_csv(RESULTS / "classification_model_comparison.csv"), use_container_width=True)
    st.dataframe(pd.read_csv(RESULTS / "regression_model_comparison.csv"), use_container_width=True)
