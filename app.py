import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Dashboard title
st.title("Credit Card Fraud Detection Dashboard")

# File upload
uploaded_file = st.file_uploader(r"C:\Users\DELL\OneDrive\Desktop\Credit Card\creditcard.csv", type=["csv"])

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    
    st.write("### First 10 rows of the dataset:")
    st.write(data.head(10))
    
    # Check for missing values
    if data.isnull().sum().sum() > 0:
        st.warning("Dataset contains missing values. They will be dropped.")
        data.dropna(inplace=True)

    # Separate features and target
    X = data.drop(['Class'], axis=1)
    Y = data['Class']

    # Split dataset into train and test
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier()
    model.fit(xTrain, yTrain)
    
    # Evaluate the model
    yPred = model.predict(xTest)
    acc = accuracy_score(yTest, yPred)
    cm = confusion_matrix(yTest, yPred)

    # Display metrics
    st.write("### Model Evaluation Metrics:")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write("#### Confusion Matrix:")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Valid', 'Fraud'], yticklabels=['Valid', 'Fraud'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    
    # Prediction Section
    st.write("### Predict Fraudulent Transactions")
    
    # Allow the user to input transaction details
    st.write("Enter the transaction details:")
    transaction_input = {col: st.number_input(col, value=float(data[col].mean())) for col in X.columns}

    # Convert user input to DataFrame
    input_df = pd.DataFrame([transaction_input])
    
    # Predict fraud or not
    prediction = model.predict(input_df)
    prediction_label = "Fraud" if prediction[0] == 1 else "Valid"
    
    st.write(f"**Prediction:** The transaction is **{prediction_label}**")
