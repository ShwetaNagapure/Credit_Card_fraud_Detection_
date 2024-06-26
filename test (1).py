import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load and prepare the dataset
df = pd.read_csv("/content/drive/MyDrive/creditcard.csv")
true = df[df.Class == 0]
fraud = df[df.Class == 1]

true_sample = true.sample(n=len(fraud), random_state=2)
df = pd.concat([true_sample, fraud], axis=0)

X = df.drop(columns='Class', axis=1)
Y = df['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)
ypred = model.predict(X_test)

train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc = accuracy_score(Y_test, ypred)

# Streamlit app
st.title("Credit Card Fraud Detection")

time = st.number_input('Enter the time')
amount = st.number_input('Enter the amount')
input_df = st.text_input('Enter the features Values (comma-separated)')
input_df_split = [x.strip() for x in input_df.split(',')]

submit = st.button('Submit')

# Initialize prediction variable
prediction = None

if submit:
    try:
        features = np.array(input_df_split, dtype=float)
        prediction = model.predict(features.reshape(1, -1))

        if prediction is not None and prediction[0] == 0:
            st.write('The transaction is legitimate.')
        elif prediction is not None and prediction[0] == 1:
            st.write('The transaction is fraudulent.')
    except ValueError as e:
        st.write(f"Error in input: {e}")
