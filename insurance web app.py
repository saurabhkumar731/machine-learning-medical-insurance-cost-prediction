# -*- coding: utf-8 -*-
"""
Created on Mon May  5 03:06:11 2025

@author: Saurabh
"""

import pickle
import numpy as np
import streamlit as st

# loaded the saved model
loaded_model=pickle.load(open('C:/Users/Saurabh/Desktop/desktop/Deploying medical price/insurance_price.pkl','rb'))

def main():
    
    st.title('Insurance price prediction')
    
    age=st.text_input('Enter the Age of the person')
    sex=st.text_input('Enter the sex of the person')
    bmi=st.text_input('Enter bmi of person')
    children=st.text_input('Enter the no. of children of the preson')
    smoker=st.text_input('Enter person is smoker or not')
    region=st.text_input('Enter the region of person')
    
    
    if st.button('Predict'):
      try:
        # Convert inputs into correct data types
        age = int(age)
        bmi = float(bmi)
        children = int(children)

        # Encode categorical variables (assuming predefined encoding scheme)
        sex = 1 if sex.lower() == 'male' else 0
        smoker = 1 if smoker.lower() == 'yes' else 0
        region_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
        region = region_mapping.get(region.lower(), -1)  # Default to -1 for unexpected input

        # Create NumPy array with formatted inputs
        input_data = np.array([[age, sex, bmi, children, smoker, region]])

        # Make prediction
        prediction = loaded_model.predict(input_data)

        # Display the result
        st.write(f'The predicted insurance cost in USD: {prediction[0]:.2f}')
    
      except ValueError:
        st.write('Please enter valid numeric values for all features.')    
            
if __name__ == '__main__':
    main()            