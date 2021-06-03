import pandas as pd
import streamlit as st
from pycaret.classification import *
    
@st.cache(allow_output_mutation=True)
def get_data():
    return []

age = st.number_input("Enter Age")
sex = st.slider("Enter 1 for Male and 0 for female", 0, 1)
exng = st.slider("1 for yes, 0 for no", 0, 1)
caa = st.slider("number of major vessels", 0, 3)
cp = st.slider("chest pain type", 1, 4)
trtbps = st.number_input("Resting BP in mmHg")
chol = st.number_input("Cholestrol level")
fbs = st.slider("Is Fasting Sugar Level greater than 120 : 1 for Yes and 0 for no", 0,1)
restecg = st.slider("ECG", 0, 2)
thall = st.number_input("Enter Thall")
oldpeak = st.number_input("Enter Old Peak Data between 0 - 6")
thalachh = st.number_input("Enter between 71 - 202")
slp = st.slider("Slope", 0, 2)


if st.button("Add Data"):
    get_data().append({"age" : age, "sex" : sex, "exng" : exng, "caa" : caa, "cp" : cp,
        "trtbps" : trtbps, "chol" : chol, "fbs" : fbs, "restecg" : restecg, "thall" : thall,
        "oldpeak" : oldpeak, "thalachh" : thalachh, "slp" : slp})

    input_data = pd.DataFrame(get_data())
        
saved_final_lr = load_model('Heart Attack Prediction Model') 
new_prediction = predict_model(saved_final_lr, data=input_data)       
final = pd.DataFrame(new_prediction)
st.write(final["Score"].iloc[-1])

if(final["Score"].iloc[-1] > (.50)):
    st.write("The chnaces of heart are high")
else:
    st.write("Chances of heart attack are low")