import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from recommendation import get_recommendation

# load model
model = joblib.load("credit_model.pkl")

st.set_page_config(page_title="AI Credit Risk System", layout="wide")

st.title("AI Credit Risk Decision Support System")

st.sidebar.header("Customer Information")

credit_history = st.sidebar.selectbox(
    "Credit History",
    [0,1,2,3,4]
)

amount = st.sidebar.number_input("Loan Amount",100,20000)

duration = st.sidebar.slider("Loan Duration (months)",1,72)

age = st.sidebar.slider("Age",18,75)

employment_duration = st.sidebar.selectbox(
    "Employment Duration",
    [0,1,2,3,4]
)

savings = st.sidebar.selectbox(
    "Savings Level",
    [0,1,2,3,4]
)

purpose = st.sidebar.selectbox(
    "Loan Purpose",
    [0,1,2,3,4,5]
)

other_debtors = st.sidebar.selectbox(
    "Other Debtors",
    [0,1,2]
)

input_data = pd.DataFrame([[
    credit_history,
    amount,
    duration,
    age,
    employment_duration,
    savings,
    purpose,
    other_debtors
]],columns=[
    "credit_history",
    "amount",
    "duration",
    "age",
    "employment_duration",
    "savings",
    "purpose",
    "other_debtors"
])

if st.sidebar.button("Predict Credit Risk"):

    prob = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    col1,col2,col3 = st.columns(3)

    col1.metric("Risk Probability",round(prob,2))

    if prediction == 1:
        col2.success("Low Risk")
    else:
        col2.error("High Risk")

    recommendation = get_recommendation(prob)

    col3.info(recommendation)

    st.subheader("Customer Input Summary")
    st.write(input_data)

st.subheader("Dataset Insights")

# load dataset
data = pd.read_csv("credit_data.csv")

col1,col2 = st.columns(2)

fig1,ax1 = plt.subplots()
sns.histplot(data["amount"],ax=ax1)
ax1.set_title("Loan Amount Distribution")
col1.pyplot(fig1)

fig2,ax2 = plt.subplots()
sns.histplot(data["age"],ax=ax2)
ax2.set_title("Age Distribution")
col2.pyplot(fig2)

st.subheader("Correlation Heatmap")

fig3,ax3 = plt.subplots()
sns.heatmap(data.corr(),ax=ax3)

st.pyplot(fig3)
