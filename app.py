pip install shap

import streamlit as st
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('model.pkl')

# Load the test data
X_test = pd.read_csv('Customer Churn.csv')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

predictions = model.predict(X_test)
print(predictions)

# Initialize SHAP Explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Streamlit App
st.title("SHAP Streamlit App")

#Summary Plot:
st.title("SHAP Summary Plot")
st.write("This plot shows the overall feature importance and their impact on predictions.")
shap.summary_plot(shap_values, X_test)
st.pyplot(bbox_inches='tight')

#Force Plot:
st.title("SHAP Force Plot")
selected_index = st.selectbox("Select a data point for Force Plot analysis", X_test.index)
st.write(f"Force plot for data point: {selected_index}")
shap.force_plot(explainer.expected_value[1], shap_values[1][selected_index, :], X_test.iloc[selected_index, :])

#Decision Plot:
st.title("SHAP Decision Plot")
st.write("This plot visualizes the decision path of features for a given instance.")
shap.decision_plot(explainer.expected_value[1], shap_values[1][selected_index, :], X_test.iloc[selected_index, :])

#Interective Elements:
st.title("Interactive Feature Adjustment")
feature_to_adjust = st.selectbox("Select a feature to adjust", X_test.columns)
new_value = st.slider(f"Adjust {feature_to_adjust}", float(X_test[feature_to_adjust].min()), float(X_test[feature_to_adjust].max()), float(X_test[feature_to_adjust].mean()))
X_test_copy = X_test.copy()
X_test_copy.loc[selected_index, feature_to_adjust] = new_value
new_shap_values = explainer.shap_values(X_test_copy)
shap.force_plot(explainer.expected_value[1], new_shap_values[1][selected_index, :], X_test_copy.iloc[selected_index, :])
