import streamlit as st
import pandas as pd
import joblib

# Load the trained model from disk
model = joblib.load('titanic_mlp_model.pkl')

# Define the input form for the user to enter passenger details
st.title("Titanic Survival Prediction")
st.write("Enter the passenger details to predict survival")

# Input fields
pclass = st.selectbox("Passenger Class (pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (sibsp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=30.0)
embarked = st.selectbox("Port of Embarkation (embarked)", ["C", "Q", "S"])

# Create a dictionary from the input
data = {
    'pclass': [pclass],
    'sex': [sex],
    'age': [age],
    'sibsp': [sibsp],
    'parch': [parch],
    'fare': [fare],
    'embarked': [embarked]
}

# Convert the dictionary to a DataFrame
input_df = pd.DataFrame(data)

# Make predictions when the user clicks the button
if st.button("Predict"):
    prediction = model.predict(input_df)
    result = "Survived" if prediction[0] == 1 else "Did not survive"
    st.write(f"The prediction is: **{result}**")

# Add information about the model
st.write("""
### About this model:
- This model is a neural network classifier (MLPClassifier) trained to predict the survival of passengers on the Titanic based on their details.
- The model was trained on the Titanic dataset and includes preprocessing steps for handling missing values, scaling, and one-hot encoding of categorical variables.
""")
