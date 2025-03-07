import sys
print("Python version:", sys.version)
import os
os.system("pip install --no-cache-dir scikit-learn")
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('Iris Flower Prediction')
st.image('type_of_iris.jpg', width=500)

setosa = 'setosa.jpg'
versicolor = 'versicolor.jpg'
virginica = 'virginica.jpg'
flower_type = [setosa, versicolor, virginica]

df, target_names =pd.read_csv('iris.csv'), ['setosa', 'versicolor', 'virginica']

model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['target'])

st.sidebar.title('Set the Sepal and Petal Length and Width')
sepal_length = st.sidebar.slider('Sepal Length', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width =  st.sidebar.slider('Sepal Width', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider('Petal Length', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width =  st.sidebar.slider('Petal Width', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

# prediction
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
st.write('------------------------------------------------------------------------')
st.title('Prediction')
st.write(f'Iris Flower Type : {target_names[prediction[0]]}')
st.image(flower_type[prediction[0]], width=300)


st.markdown(
    """
    <style>
        # body::-webkit-scrollbar {
        #     display: none;
        # }
    
        # html, body {
        #     overflow: hidden;
        #     height: 100%;
        # }
        # .main {
        #     position: fixed;
        #     top: 0;
        #     left: 0;
        #     width: 100%;
        #     height: 100vh;
        #     overflow: hidden;
        # }
    
        .footer {
            margin-top : 5rem;
            display: flex;
            justify-content: center;
            position : relative;
            bottom: 0rem;
            text-align: center;
            font-size: 14px;
        }
        .st-emotion-cache-t1wise{
            
            padding-bottom:0;   
        }
    </style>
    <div class="footer">
        <p>© Copyright 2025 - Onkar Shinde</p>
    </div>
    """,
    unsafe_allow_html=True
)
