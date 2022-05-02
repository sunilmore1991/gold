import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import base64

st.title('Time Series Forecasting of Gold Price')

st.write("IMPORT DATA")
st.write("Import the time series CSV file. It should have two columns labelled as 'date' and 'price'.The 'date' column should be of DateTime format by Pandas. The 'price' column must be numeric representing the measurement to be forecasted.") 

data = st.file_uploader('Upload here',type='csv')

if data is not None:
     appdata = pd.read_csv(data)  #read the data fro
     appdata['date'] = pd.to_datetime(appdata.date,errors='coerce') 
     st.write(data) #display the data  
     max_date = appdata['date'].max() #compute latest date in the data 

st.write("SELECT FORECAST PERIOD")    #text displayed

periods_input = st.number_input('How many years forecast do you want?',
min_value = 1, max_value = 2)
#The minimum number of days a user can select is one, while the maximum is  #365 (yearly forecast)


if data is not None:
     model =  sm.tsa.statespace.SARIMAX(gold1,order=(0, 1, 2), seasonal_order=(0,1,2,31))    
     model.fit(appdata)    

st.write("VISUALIZE FORECASTED DATA")  
st.write("The following plot shows future predicted values. 'mean' is the  predicted value; upper and lower limits are 80% confidence intervals by  default")
if data is not None:
     periods=12 *periods_input
     future = model.make_future_dataframe(periods, freq='M')
     fcst = model.predict(future) 
     forecast = fcst[['mean']]
     forecast_filtered =  forecast[forecast['date'] > max_date]
     st.write(forecast_filtered)  #Display some forecasted records
     st.write("The next visual shows the actual (black dots) and predicted (blue line) values over time.")    
     figure1 = model.plot(fcst) #plot the actual and predicted values
     st.write(figure1)  #display the plot

