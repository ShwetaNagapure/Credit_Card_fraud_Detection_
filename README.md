# Credit Card Fraud Detection

This project involves building a machine learning model to detect fraudulent credit card transactions. The dataset used, the model implemented, and the deployment of the model using Streamlit are detailed below.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)
- [Evaluation](#evaluation)
- [Deployment](#deployment)


## Introduction

Credit card fraud detection is a significant challenge in the financial industry. The goal of this project is to build a logistic regression model to distinguish between legitimate and fraudulent transactions. The model is trained on a balanced dataset and deployed using Streamlit to provide an interactive web interface.

## Dataset

The dataset used in this project is the Credit Card Fraud Detection dataset from Kaggle. It contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred over two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly imbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

- [Credit Card Fraud Detection Dataset](https://drive.google.com/file/d/1f0zpRTjU-9ic7hdKkxIBEpW1muxX4SuE/view?usp=sharing)

## Model

The model used for this project is a Logistic Regression classifier. Logistic Regression is a simple yet effective algorithm for binary classification tasks.

## Evaluation

The model's performance is evaluated using various metrics:

- **Accuracy**: 0.9137055837563451
- **Precision**: 0.9764705882352941
- **Recall**: 0.8469387755102041
- **F1 Score**: 0.907103825136612
- **ROC AUC Score**: 0.913368377654092

## Deployment

The model is deployed using Streamlit, a powerful and easy-to-use framework for deploying machine learning models as web applications.

### Streamlit App

You can interact with the deployed model through this Streamlit app: [Credit Card Fraud Detection App](https://short-parents-listen.loca.lt/)

![Application Interface](application interface)
