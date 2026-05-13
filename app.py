import streamlit as st
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


# Load the trained model and preprocessing objects
model = load_model("my_model.h5")


# load the encoder and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender =pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo= pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler= pickle.load(file)

#Steamlit app
st.title('Customer Churn Prediction')

# Use input
geography=st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender=st.selectbox('Gender', label_encoder_gender.classes_)
age=st.slider('Age', 18, 92)
balance= st.number_input('Balance', min_value=0.0)
credit_score=st.number_input('Credit Score', min_value=0.0)
tenure=st.slider('Tenure', 0, 10)
num_of_products=st.slider('Number of Products', 1,4)
has_cr_card=st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)

# Preprocess the input data (same column layout as experiments / prediction notebooks)
input_df = pd.DataFrame([{
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,
}])

geo_encoded = one_hot_encoder_geo.transform(input_df[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=one_hot_encoder_geo.get_feature_names_out(['Geography']),
)
input_df = pd.concat([input_df.drop(columns=['Geography']), geo_encoded_df], axis=1)
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

if hasattr(scaler, 'feature_names_in_'):
    input_df = input_df[list(scaler.feature_names_in_)]

input_scaled = scaler.transform(input_df)

#Predict churn 
prediction = model.predict(input_scaled)
prediction_prob= prediction[0][0]

if prediction_prob > 0.5:
    st.write("The customer is likely to churn.")
    st.write(f"Churn Probability: {prediction_prob:.2f}")
else:
    st.write("The customer is not likely to churn.")
    st.write(f"Churn Probability: {prediction_prob:.2f}")

