import pandas as pd
import streamlit as st
from xgboost import XGBRegressor
import datetime
import joblib


page_bg_img= """
<style>
[data-testid="stAppViewContainer"]
{
    background-image: url("https://wallpaperswide.com/download/black_background_fabric_ii-wallpaper-960x640.jpg");
    background-repeat: no-repeat;
    background-size: cover;
    
}
[data-testid="stSidebar"]{ 
top:50px;
background-image: url("https://wallpaperswide.com/download/black_background_fabric_ii-wallpaper-960x640.jpg");
background-repeat: no-repeat;
height: 100%;
background-size: cover;
opacity:1;
}
[data-testid="stHeader"]{
background-color: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html = True)



st.header(" :gray[ 🚗 Car price prediction]",divider="blue")
present_price =  st.number_input(label="Present Price of Car :",placeholder="Write a price....",value=None)
kms_driven = st.number_input(label="Kms Driven of car  :",placeholder="type a number...",value=None)
fuel_type = st.selectbox('Select fuel of Car :',('Petrol','Diesel','CNG'))
seller_type = st.selectbox('Select seller of Car :',('Individual','Dealer'))
transmission = st.selectbox('Trasmission of Car :',('Manual','Automatic'))
owner = st.slider('owners of Car :',0,3)
years =  st.number_input(label='Write year when you purchase car :',placeholder='write here...',value=2015)
date_time = datetime.datetime.now()
y = date_time.year-int(years)

def fuel():
    if fuel_type == "Petrol":
        return 0
    elif fuel_type == "Diesel":
        return 1
    else:
        return 2

def seller():
    if seller_type == "Individual":
        return 1
    else:
        return 0

def transmission():
    if transmission == "Manual":
        return 0
    else :
        return 1
data_new = pd.DataFrame({
    'Present_Price':present_price,
    'Kms_Driven':kms_driven,
    'Fuel_Type':fuel(),
    'Seller_Type':seller(),
    'Transmission':transmission(),
    'Owner':owner,
    'Age': y
},index = [0])


data1 = pd.DataFrame(data_new)
if st.button('Predict'):
    
    data = pd.read_csv("car data.xls")
    date_time = datetime.datetime.now()
    data['Age'] = date_time.year - data['Year']
    data.drop('Year',axis=1,inplace=True)
    data = data[~(data["Selling_Price"]>=33.0) & (data["Selling_Price"]<=35.0)]
    data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol':0,'Diesel':1,'CNG':2})
    data['Seller_Type'] = data['Seller_Type'].map({'Dealer':0,'Individual':1})
    data['Transmission'] = data['Transmission'].map({'Manual':0,'Automatic':1})
    X = data.drop(['Car_Name','Selling_Price'],axis = 1) #independent variable
    y = data['Selling_Price']      
    xg = XGBRegressor()
    xg_final = xg.fit(X,y)
    joblib.dump(xg_final,'car_price_prediction')
    model = joblib.load('car_price_prediction')
    
    st.write(data1)
    pred = model.predict(data_new)
    # Customize the appearance using HTML and CSS
    pred_value = f'<h1 style="color:#59D5E0;">Predicted value of Car is : {pred} </h1>'

    # Render the styled header using markdown
    st.markdown(pred_value, unsafe_allow_html=True)
