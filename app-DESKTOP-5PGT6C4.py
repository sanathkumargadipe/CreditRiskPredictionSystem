import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("credit_model.pkl")

# Page config
st.set_page_config(page_title="Credit Risk System", layout="wide")


# HEADER

st.title("Credit Risk System")

# MAIN IMAGE
st.image("https://images.unsplash.com/photo-1569025690938-a00729c9e1d1", use_column_width=True)

st.markdown("---")

# SIDEBAR INPUT (UPDATED WITH LABELS)

st.sidebar.header("Input Features")

# Credit History
credit_history_label = st.sidebar.selectbox("Credit History", ["Bad", "Good"])
credit_history = 0 if credit_history_label == "Bad" else 1

# Loan Amount (no upper limit)
amount = st.sidebar.number_input("Loan Amount", min_value=100, step=100)

# Duration
duration = st.sidebar.slider("Duration (months)", 1, 72)

# Age
age = st.sidebar.slider("Age", 18, 75)

# Employment Duration
employment_label = st.sidebar.selectbox("Employment Duration", [
    "Unemployed",
    "< 1 year",
    "1–4 years",
    "4–7 years",
    "7+ years"
])

employment_mapping = {
    "Unemployed": 0,
    "< 1 year": 1,
    "1–4 years": 2,
    "4–7 years": 3,
    "7+ years": 4
}
employment_duration = employment_mapping[employment_label]

# Savings
savings_label = st.sidebar.selectbox("Savings", [
    "No savings",
    "< 1000",
    "1000–5000",
    "> 5000"
])

savings_mapping = {
    "No savings": 0,
    "< 1000": 1,
    "1000–5000": 2,
    "> 5000": 3
}
savings = savings_mapping[savings_label]

# Purpose
purpose_label = st.sidebar.selectbox("Purpose", [
    "Car",
    "Education",
    "Business",
    "Personal"
])

purpose_mapping = {
    "Car": 0,
    "Education": 1,
    "Business": 2,
    "Personal": 3
}
purpose = purpose_mapping[purpose_label]

# Other Debtors
other_debtors_label = st.sidebar.selectbox("Other Debtors", ["No", "Yes"])
other_debtors = 0 if other_debtors_label == "No" else 1

# INPUT DATAFRAME

input_data = pd.DataFrame([[
    credit_history,
    amount,
    duration,
    age,
    employment_duration,
    savings,
    purpose,
    other_debtors
]], columns=[
    "credit_history",
    "amount",
    "duration",
    "age",
    "employment_duration",
    "savings",
    "purpose",
    "other_debtors"
])

# PREDICTION

st.header("Prediction")

if st.sidebar.button("Predict"):

    prediction = model.predict(input_data)[0]

    prob_good = model.predict_proba(input_data)[0][1]
    prob_bad = model.predict_proba(input_data)[0][0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Low Risk", round(prob_good, 2))
    col2.metric("High Risk", round(prob_bad, 2))

    if prediction == 1:
        col3.success("Low Risk")
    else:
        col3.error("High Risk")

    if prob_bad > 0.7:
        decision = "Reject Loan"
    elif prob_bad > 0.4:
        decision = "Approve with Conditions"
    else:
        decision = "Approve Loan"

    st.subheader("Decision")
    st.info(decision)

    st.subheader("Input Data")
    st.write(input_data)

st.markdown("---")

# DATA VISUALS

st.header("Data Insights")

data = pd.read_csv("credit_data.csv")

col1, col2 = st.columns(2)

# Amount distribution
fig1, ax1 = plt.subplots()
sns.histplot(data["amount"], ax=ax1)
ax1.set_title("Loan Amount")
col1.pyplot(fig1)

# Age distribution
fig2, ax2 = plt.subplots()
sns.histplot(data["age"], ax=ax2)
ax2.set_title("Age")
col2.pyplot(fig2)