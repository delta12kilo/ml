import numpy as np
import streamlit as st
from regression.template import load_model

st.sidebar.title('Diamond Price Predictor')
st.sidebar.subheader('By Thor')

pe = st.number_input('Enter Pelanoium:- ')
pr = st.number_input('Enter Pressure:- ')

bt = st.button('Show Predicted Price')

if bt:
    model = load_model('model/diamond_price.pk')
    x = np.array([[pe,pr]])
    r = model.predict(x)
    st.success(r)