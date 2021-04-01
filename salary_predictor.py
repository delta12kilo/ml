import os 
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt 
import streamlit as st
from regression.template import load_model
from sklearn.preprocessing import PolynomialFeatures

st.sidebar.title('Position wise Salary Prediction')
job_titles = ['Business Analyst', 'Junior Consultant', 'Senior Consultant', 'Manager', 'Country Manager','Region Manager', 'Partner','Senior Partner','C-level','CEO']

job_title = st.selectbox('Please select a job title', job_titles)
btn = st.button('Show Predicted Salary')

if job_title and btn:
    model = load_model('model/position_salary.pk')
    pos = job_titles.index(job_title) + 1
    x = np.array([[pos]])
    pf = PolynomialFeatures(degree = 6)
    px = pf.fit_transform(x)
    result = model.predict(px)
    st.success(round(result[0]))