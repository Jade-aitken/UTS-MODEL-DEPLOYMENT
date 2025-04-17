import streamlit as st
import joblib
import numpy as np

# Load model dan encoder
model = joblib.load('best_xgb_model.pkl')  
gender_enc = joblib.load('gender_encode.pkl') 
education_enc = joblib.load('education_encode.pkl')  
home_own_enc = joblib.load('home_own_encode.pkl') 
loan_intent_enc = joblib.load('loan_intent_encode.pkl')  
prev_defaults_enc = joblib.load('prev_defaults_encode.pkl')  

# Encode categorical values
def apply_encoding(user_input):
    user_input[1] = gender_enc.get(user_input[1], 0)
    user_input[2] = education_enc.get(user_input[2], 0)
    user_input[5] = home_own_enc.get(user_input[5], 0)
    user_input[7] = loan_intent_enc.get(user_input[7], 0)
    user_input[12] = prev_defaults_enc.get(user_input[12], 0)

    # Convert to float numpy array and reshape for prediction
    return np.array(user_input, dtype=float).reshape(1, -1)



def make_prediction(features):
    input_array = apply_encoding(features)
    prediction = model.predict(input_array)
    return prediction[0]

# Streamlit UI
st.title("Loan Approval Prediction")
st.write("Masukkan data pelanggan untuk memprediksi status pinjaman.")

# User input 
person_age = st.number_input("Age", min_value=18, max_value=90, value=30, step=1)  
person_gender = st.selectbox("Gender", ["male", "female"])
person_education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
person_income = st.number_input("Income (Annual)", min_value=1000, max_value=500000, value=50000, step=1000) 
person_emp_exp = st.slider("Employment Experience (Years)", min_value=0, max_value=60, value=10)  
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
loan_amnt = st.number_input("Loan Amount", min_value=1000, max_value=100000, value=10000, step=500)  
loan_intent = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_int_rate = st.slider("Loan Interest Rate (%)", min_value=0, max_value=20, value=5)  
loan_percent_income = st.slider("Loan as Percentage of Income", min_value=0, max_value=100, value=20) 
loan_percent_income = loan_percent_income / 100 # karena dalam dataset decimal
cb_person_cred_hist_length = st.slider("Credit History Length (Years)", min_value=0, max_value=40, value=10) 
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)  
previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["Yes", "No"])

if st.button("Predict Loan Status"):
    user_input = [
        person_age, person_gender, person_education, person_income, person_emp_exp,
        person_home_ownership, loan_amnt, loan_intent, loan_int_rate, loan_percent_income,
        cb_person_cred_hist_length, credit_score, previous_loan_defaults_on_file
    ]
    
    prediction = make_prediction(user_input)
    
    if prediction == 1:
        st.success("✅ Pinjaman Disetujui.")
    else:
        st.error("⚠️ Pinjaman Ditolak!")

