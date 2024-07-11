#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import joblib
import numpy as np

model = joblib.load(r"/Users/mac/Desktop/ipynb files/aptechfinalproject1_model.pkl")
st.title("Breast Cancer Prediction")
mean_radius = st.number_input("mean_radius", min_value=0.1, max_value=1000.0, value=10.0)
worst_concavity = st.number_input("worst_concavity", min_value=0.1, max_value=1000.0, value=10.0)
worst_compactness = st.number_input("worst_compactness", min_value=0.1, max_value=1000.0, value=10.0)
worst_radius = st.number_input("worst_radius", min_value=0.1, max_value=1000.0, value=10.0)
worst_concave_points = st.number_input("worst_concave_points", min_value=0.1, max_value=1000.0, value=10.0)
mean_concavity = st.number_input("mean_concavity", min_value=0.1, max_value=1000.0, value=10.0)
perimeter_se = st.number_input("perimeter_se", min_value=0.1, max_value=1000.0, value=10.0)
mean_compactness = st.number_input("mean_compactness", min_value=0.1, max_value=1000.0, value=10.0)
mean_concave_points = st.number_input("mean_concave_points", min_value=0.1, max_value=1000.0, value=10.0)
worst_perimeter = st.number_input("worst_perimeter", min_value=0.1, max_value=1000.0, value=10.0)
mean_perimeter = st.number_input("mean_perimeter", min_value=0.1, max_value=1000.0, value=10.0)
area_se = st.number_input("area_se", min_value=0.1, max_value=1000.0, value=10.0)
worst_area = st.number_input("worst_area", min_value=0.1, max_value=1000.0, value=10.0)
radius_se = st.number_input("radius_se", min_value=0.1, max_value=1000.0, value=10.0)
mean_area = st.number_input("mean_area", min_value=0.1, max_value=1000.0, value=10.0)

if st.button("Predict"):
    input_data = np.array([[mean_radius, mean_perimeter, mean_area,
                             mean_compactness, mean_concavity, mean_concave_points,
                             radius_se, perimeter_se, area_se, worst_radius,
                             worst_perimeter, worst_area, worst_compactness, worst_concavity,
                             worst_concave_points]])

    prediction = model.predict(input_data)

    # Map prediction back to original labels
    predicted_label = 'M' if prediction == 1 else 'B'
   
    st.write(f"Predicted Breast Cancer Status: {predicted_label}")

