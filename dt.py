import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load model (replace with the path to your trained model)
@st.cache
def load_model():
    model = joblib.load('decision_tree_model.pkl')  # Load a pre-trained model
    return model

# Load data
def load_data():
    data = pd.read_csv('sample_data.csv')  # Replace with your dataset
    return data

# Streamlit app
st.title("Decision Tree Classifier")

# Load the model and data
model = load_model()
data = load_data()

# Display data
st.subheader('Dataset')
st.write(data.head())

# Input features for prediction
st.sidebar.header('Input Features')
feature1 = st.sidebar.slider('Feature 1', min_value=0, max_value=100, value=50)
feature2 = st.sidebar.slider('Feature 2', min_value=0, max_value=100, value=50)
# Add more features as per your dataset
input_features = [feature1, feature2]

# Perform prediction
if st.sidebar.button('Predict'):
    prediction = model.predict([input_features])
    st.write(f'Prediction: {prediction[0]}')

# Display model parameters
st.subheader('Model Parameters')
st.write(model.get_params())