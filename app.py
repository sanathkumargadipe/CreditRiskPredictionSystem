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

st.title("Credit Risk System")

st.image("image.png", use_column_width=True)

st.markdown("---")

# SIDEBAR INPUT
st.sidebar.header("Input Features")

# IMPORTANT → use SAME values as training dataset

credit_history = st.sidebar.selectbox(
    "Credit History",
    list(encoders["credit_history"].classes_)
)

amount = st.sidebar.number_input("Loan Amount", min_value=100, step=100)

duration = st.sidebar.slider("Duration (months)", 1, 72)

age = st.sidebar.slider("Age", 18, 75)

employment_duration = st.sidebar.selectbox(
    "Employment Duration",
    list(encoders["employment_duration"].classes_)
)

savings = st.sidebar.selectbox(
    "Savings",
    list(encoders["savings"].classes_)
)

purpose = st.sidebar.selectbox(
    "Purpose",
    list(encoders["purpose"].classes_)
)

other_debtors = st.sidebar.selectbox(
    "Other Debtors",
    list(encoders["other_debtors"].classes_)
)

housing = st.sidebar.selectbox(
    "Housing",
    list(encoders["housing"].classes_)
)

job = st.sidebar.selectbox(
    "Job",
    list(encoders["job"].classes_)
)

property_val = st.sidebar.selectbox(
    "Property",
    list(encoders["property"].classes_)
)

installment_rate = st.sidebar.slider("Installment Rate (%)", 1, 4)

# CREATE INPUT DATAFRAME (ALL 12 FEATURES)
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
    property_val
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

# APPLY ENCODING
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

# VISUALS
st.header("Data Insights")

data = pd.read_csv("data/credit_data.csv")

col1, col2 = st.columns(2)

fig1, ax1 = plt.subplots()
sns.histplot(data["amount"], ax=ax1)
ax1.set_title("Loan Amount")
col1.pyplot(fig1)

fig2, ax2 = plt.subplots()
sns.histplot(data["age"], ax=ax2)
ax2.set_title("Age")
col2.pyplot(fig2)