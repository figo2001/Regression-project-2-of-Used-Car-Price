import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor



# Loading the Models
sc=pickle.load(open('Scaler.pkl','rb'))
model=pickle.load(open('XG.pkl','rb'))



# Prediction Function

def predict_results(input_data):
    input_data=np.array(input_data)
    input_data_reshaped=input_data.reshape(1,-1) 
    scaled_data=sc.transform(input_data_reshaped)
    prediction= model.predict(scaled_data)

    return prediction


# Main Function

def main():
    
    # Home page Title
    st.title("Used Car Price Prediction ")

    # Competition info
    st.success('Playground Series - Season 4, Episode 9')

    # Getting input from the user
    
    #brand', 'model', 'model_year', 'milage', 'fuel_type', 'engine',
     #  'engine', 'ext_col', 'int_col', 'accident'],
      

    brand=st.number_input('brand')
    model=st.number_input('model')
    model_year=st.number_input('model_year')
    milage=st.number_input('milage')
    fuel_type=st.number_input('fuel_type')
    engine=st.number_input('engine')
    transmission=st.number_input('transmission')
    ext_col=st.number_input('ext_col')
    int_col=st.number_input('int_col')
    accident=st.number_input('accident')

    results=''

    if st.button("Prediction"):
        results=predict_results([brand, model, model_year, milage, fuel_type,engine,
                                 transmission, ext_col, int_col, accident])
        
        st.success(results)
    


if __name__=='__main__':
    main()