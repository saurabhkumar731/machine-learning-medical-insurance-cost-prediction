# -*- coding: utf-8 -*-
"""
Created on Mon May  5 02:59:12 2025

@author: Saurabh
"""

import pickle
import numpy as np

# loaded the saved model
loaded_model=pickle.load(open('C:/Users/Saurabh/Desktop/desktop/Deploying medical price/insurance_price.pkl','rb'))

input_data= (19,1,27.9,0,0,1)

# changing input_data to nupy array
input_data_as_array =np.asarray(input_data)

# reshape the array
input_data_reshaped=input_data_as_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

print('The insurance cost in USD',prediction[0])