import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load("credit_model.pkl")

# Page configuration
st.set_page_config(page_title="Credit Risk System", layout="wide")

# -----------------------------------------------------
# HEADER
# -----------------------------------------------------

st.title("AI Credit Risk Decision Support System")

st.markdown("""
### Intelligent Loan Risk Assessment Platform

This system uses **Machine Learning** to analyze borrower financial data and
predict whether a customer is **Low Risk** or **High Risk** for loan approval.

The goal of this system is to assist financial institutions in making **data-driven
credit decisions**.
""")

st.image(
"https://images.unsplash.com/photo-1554224155-6726b3ff858f",
caption="AI-driven financial risk analysis"
)

st.markdown("---")

# -----------------------------------------------------
# SIDEBAR INPUT
# -----------------------------------------------------

st.sidebar.header("Customer Financial Information")

credit_history = st.sidebar.selectbox("Credit History", [0,1,2,3,4])

amount = st.sidebar.number_input(
"Loan Amount",
100,
20000
)

duration = st.sidebar.slider(
"Loan Duration (months)",
1,
72
)

age = st.sidebar.slider(
"Customer Age",
18,
75
)

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

# -----------------------------------------------------
# CREATE INPUT DATAFRAME
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
# PREDICTION SECTION
# -----------------------------------------------------

st.header("Credit Risk Prediction")

st.markdown("""
Click the **Predict Credit Risk** button to evaluate the applicant’s
loan repayment risk using the trained machine learning model.
""")

if st.sidebar.button("Predict Credit Risk"):

    prediction = model.predict(input_data)[0]

    prob_good = model.predict_proba(input_data)[0][1]
    prob_bad = model.predict_proba(input_data)[0][0]

    col1,col2,col3 = st.columns(3)

    col1.metric("Low Risk Probability", round(prob_good,2))
    col2.metric("High Risk Probability", round(prob_bad,2))

    if prediction == 1:
        col3.success("Prediction: Low Risk Borrower")
    else:
        col3.error("Prediction: High Risk Borrower")

    # Recommendation
    if prob_bad > 0.7:
        recommendation = "Loan should be rejected due to high credit risk."

    elif prob_bad > 0.4:
        recommendation = "Loan may be approved with a reduced amount or stricter conditions."

    else:
        recommendation = "Loan can be approved safely."

    st.subheader("Credit Decision Recommendation")
    st.info(recommendation)

    st.subheader("Customer Input Summary")
    st.write(input_data)

st.markdown("---")

# -----------------------------------------------------
# SYSTEM EXPLANATION
# -----------------------------------------------------

st.header("How the System Works")

st.markdown("""
The AI Credit Risk System follows these steps:

1. **Data Collection** – Customer financial and demographic data is gathered.
2. **Feature Analysis** – Important risk indicators such as credit history and savings are analyzed.
3. **Machine Learning Prediction** – The trained model evaluates repayment probability.
4. **Risk Classification** – The borrower is classified as Low Risk or High Risk.
5. **Decision Recommendation** – The system suggests whether the loan should be approved.
""")

st.image(
"https://images.unsplash.com/photo-1569025690938-a00729c9e1d1",
caption="AI systems assisting financial decision making"
)

st.markdown("---")

# -----------------------------------------------------
# DATASET INSIGHTS
# -----------------------------------------------------

st.header("Dataset Insights")

st.markdown("""
Below are some visual insights from the dataset used to train the credit risk model.
These charts help understand patterns in loan applications.
""")

data = pd.read_csv("data/credit_data.csv")

col1,col2 = st.columns(2)

# Loan Amount Distribution
fig1,ax1 = plt.subplots()
sns.histplot(data["amount"],ax=ax1)
ax1.set_title("Loan Amount Distribution")
col1.pyplot(fig1)

# Age Distribution
fig2,ax2 = plt.subplots()
sns.histplot(data["age"],ax=ax2)
ax2.set_title("Customer Age Distribution")
col2.pyplot(fig2)

st.subheader("Feature Correlation Heatmap")

fig3,ax3 = plt.subplots()

sns.heatmap(
data.corr(),
annot=True,
cmap="coolwarm",
ax=ax3
)

st.pyplot(fig3)

st.markdown("---")

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------

st.markdown("""
### Project Summary

This application demonstrates how **Artificial Intelligence**
can support financial institutions in **automating credit risk assessment**.

**Key Technologies Used**
- Python
- Machine Learning
- Data Visualization
- Interactive Web Dashboard

Developed as an academic project for **AI-based financial decision systems**.
""")
