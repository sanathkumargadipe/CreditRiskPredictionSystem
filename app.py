import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and encoders
model = joblib.load("credit_model.pkl")
encoders = joblib.load("encoders.pkl")

# Page config
st.set_page_config(page_title="Credit Risk System", layout="wide")

# HEADER
st.title("Credit Risk System")

# MAIN IMAGE
st.image("https://images.unsplash.com/photo-1569025690938-a00729c9e1d1", width=700)

st.markdown("---")

# SIDEBAR INPUT
st.sidebar.header("Input Features")

# Credit History
credit_history = st.sidebar.selectbox("Credit History", ["good", "bad"])

# Loan Amount
amount = st.sidebar.number_input("Loan Amount", min_value=100, step=100)

# Duration
duration = st.sidebar.slider("Duration (months)", 1, 72)

# Age
age = st.sidebar.slider("Age", 18, 75)

# Employment Duration
employment_duration = st.sidebar.selectbox(
    "Employment Duration",
    ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"]
)

# Savings
savings = st.sidebar.selectbox(
    "Savings",
    ["no savings", "<1000", "1000<=X<5000", ">=5000"]
)

# Purpose
purpose = st.sidebar.selectbox(
    "Purpose",
    ["car", "education", "business", "radio/tv", "furniture", "others"]
)

# Other Debtors
other_debtors = st.sidebar.selectbox("Other Debtors", ["none", "yes"])

# Housing
housing = st.sidebar.selectbox("Housing", ["own", "rent", "free"])

# Job
job = st.sidebar.selectbox(
    "Job",
    ["unemployed", "unskilled", "skilled", "highly skilled"]
)

# Installment Rate
installment_rate = st.sidebar.slider("Installment Rate", 1, 5)

# Property
property = st.sidebar.selectbox(
    "Property",
    ["real estate", "savings", "car", "unknown"]
)

# CREATE INPUT DATAFRAME
input_data = pd.DataFrame([[
    credit_history,
    amount,
    duration,
    age,
    employment_duration,
    savings,
    purpose,
    other_debtors,
    housing,
    job,
    installment_rate,
    property
]], columns=[
    "credit_history",
    "amount",
    "duration",
    "age",
    "employment_duration",
    "savings",
    "purpose",
    "other_debtors",
    "housing",
    "job",
    "installment_rate",
    "property"
])

# APPLY ENCODING (VERY IMPORTANT)
for col in input_data.columns:
    if col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

# PREDICTION
st.header("Prediction")

if st.sidebar.button("Predict"):

    prediction = model.predict(input_data)[0]

    prob_good = model.predict_proba(input_data)[0][1]
    prob_bad = model.predict_proba(input_data)[0][0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Low Risk", f"{prob_good:.2f}")
    col2.metric("High Risk", f"{prob_bad:.2f}")

    if prediction == 1:
        col3.success("Low Risk")
    else:
        col3.error("High Risk")

    # Decision logic
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

data = pd.read_csv("data/credit_data.csv")

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