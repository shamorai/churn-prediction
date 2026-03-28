import streamlit as st
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Churn Dashboard", layout="wide")


st.title("📊 Customer Churn Prediction System")

# st.markdown("### 🔎 Enter Customer Details")
st.markdown("""

### Predict customer behavior using Machine Learning

👉 Enter customer details to check churn risk and understand key factors affecting the decision.
""")
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure", 0, 72, 12)

with col2:
    monthly = st.slider("Monthly Charges", 0, 150, 50)

with col3:
    total = st.slider("Total Charges", 0, 10000, 1000)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

# Encode
c1, c2 = 0, 0
if contract == "One year":
    c1 = 1
elif contract == "Two year":
    c2 = 1

# Input
input_data = np.array([[tenure, monthly, total, c1, c2]])
input_scaled = scaler.transform(input_data)

st.subheader("🔍 Prediction")



if st.button("Predict Churn"):

    # 👉 prediction yahi banta hai
    pred = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    # 👉 yahi use karna hai
    if pred[0] == 1:
        st.markdown(f"""
        <div style="background-color:#ff4b4b;padding:15px;border-radius:10px">
            <h3 style="color:white;">❌ High Churn Risk</h3>
            <p style="color:white;">Probability: {round(prob[0][1]*100,2)}%</p>
        </div>
        """, unsafe_allow_html=True)

        # ✅ explanation bhi yahi andar
        st.warning("⚠️ This customer is likely to leave. Consider retention strategies.")

    else:
        st.markdown(f"""
        <div style="background-color:#4CAF50;padding:15px;border-radius:10px">
            <h3 style="color:white;">✅ Customer Safe</h3>
            <p style="color:white;">Probability: {round(prob[0][0]*100,2)}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.info("👍 This customer is stable. Maintain service quality.")

    # 👉 progress bar bhi andar
    st.progress(int(prob[0][1]*100))

 
  # =========================
# SHAP (SAFE FINAL FIX)
# =========================
# st.subheader("🧠 Feature Impact (SHAP)")
st.subheader("🧠 Why this prediction?")
st.write("Below chart shows which features influenced the prediction.")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_scaled)

feature_names = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract_One year",
    "Contract_Two year"
]

# 🔥 HANDLE ALL CASES PROPERLY
try:
    # Case 1: list output
    values = shap_values[1][0]
except:
    try:
        # Case 2: numpy array (5,2)
        values = shap_values[0][:, 1]
    except:
        # Case 3: already correct
        values = shap_values[0]

# Convert to 1D array
values = np.array(values).flatten()

# Plot
fig, ax = plt.subplots()
ax.barh(feature_names, values)
ax.set_title("Feature Impact on Prediction")

st.pyplot(fig) 

st.markdown("---")
st.write("Developed by Dev Upadhyay 🚀")