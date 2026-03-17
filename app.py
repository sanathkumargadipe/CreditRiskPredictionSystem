import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("credit_model.pkl")

# Page config
st.set_page_config(page_title="Credit Risk System", layout="wide")

# -----------------------------------------------------
# HEADER
# -----------------------------------------------------

st.title("Credit Risk System")

st.image(
    "https://images.unsplash.com/photo-1554224155-6726b3ff858f",
    use_column_width=True
)

st.markdown("---")

# -----------------------------------------------------
# SIDEBAR INPUT
# -----------------------------------------------------

st.sidebar.header("Input Features")

credit_history = st.sidebar.selectbox("Credit History", [0,1,2,3,4])
amount = st.sidebar.number_input("Loan Amount", 100, 20000)
duration = st.sidebar.slider("Duration (months)", 1, 72)
age = st.sidebar.slider("Age", 18, 75)
employment_duration = st.sidebar.selectbox("Employment Duration", [0,1,2,3,4])
savings = st.sidebar.selectbox("Savings", [0,1,2,3,4])
purpose = st.sidebar.selectbox("Purpose", [0,1,2,3,4,5])
other_debtors = st.sidebar.selectbox("Other Debtors", [0,1,2])

# -----------------------------------------------------
# INPUT DATAFRAME
# -----------------------------------------------------

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

# -----------------------------------------------------
# PREDICTION
# -----------------------------------------------------

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

# -----------------------------------------------------
# IMAGE SECTION
# -----------------------------------------------------

st.image(
    "https://images.unsplash.com/photo-1569025690938-a00729c9e1d1",
    use_column_width=True
)

st.markdown("---")

# -----------------------------------------------------
# DATA VISUALS
# -----------------------------------------------------

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

# Correlation heatmap
fig3, ax3 = plt.subplots()
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)
