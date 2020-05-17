#You can directly run the code. 
#I have hardcoded the values in the function call on line 78
from datetime import datetime
from numpy import loadtxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.models import load_model
#from flask import Flask
import tensorflow as tf
global graph
#graph = tf.get_default_graph()
model = load_model('model.h5')
#app = Flask(__name__)
def myOHE(df, column, ohe,mappings):    
    # Encode the column
    column_encoded = np.array(df[column]).reshape(-1, 1)
    column_encoded = ohe.fit_transform(column_encoded)
    for i,j in zip(df[column],column_encoded):
        if i not in mappings:
            mappings[i]=j
    # Add the attributes to the dataframe
    for i in range(len(column_encoded[0]) - 1):
        df['{}_{}'.format(column, str(i))] = column_encoded[:, i]
    
    # Drop the 'column' in the dataframe
    df.drop(column, axis=1, inplace=True)
    # Return the dataframe
    return df
def Age(age):
    date1=str(datetime.today()-datetime.strptime(age,'%d-%m-%Y'))
    date2=date1.split(",")[0]
    date3=int(date2.split(" ")[0])
    return date3/365
def prediction(model1,brand,fuel,vehicle,damage,gear,power,km,age):
    df = pd.read_csv("cleaned_dataset.csv")
    y = np.array(df['dollar_price'])
    MEAN = np.mean(y)
    STD = np.std(y)
    df = df.drop(['dollar_price','vehicle_bin', 'brand_bin', 'model_tag', 'postal_code', 'date_crawled', 'name', 'registration_year', 'gearbox', 'registration_month', 'unrepaired_damage', 'Unnamed: 0', 'last_seen_online', 'Diff_Last_Ad', 'ad_created'], axis=1)
    temp = df.loc[:, ['model', 'brand', 'vehicle_type', 'fuel_type', 'damage_bin', 'gear_bin']]
    df.drop(['model', 'brand', 'vehicle_type', 'fuel_type', 'damage_bin', 'gear_bin'], axis=1, inplace=True)
    mean_power = np.mean(df['power_ps'])
    mean_age = np.mean(df['Age'])
    mean_km = np.mean(df['kilometer'])
    std_power = np.std(df['power_ps'])
    std_age = np.std(df['Age'])
    std_km = np.std(df['kilometer'])
    z_pow = (power - mean_power)/std_power
    z_km = (km - mean_km)/std_km
    z_age = (Age(age) - mean_age)/std_age
    ohe = OneHotEncoder(sparse=False)
    mappings = {}
    temp = myOHE(temp, 'brand', ohe,mappings)
    temp = myOHE(temp, 'model', ohe,mappings)
    temp = myOHE(temp, 'fuel_type', ohe, mappings)
    temp = myOHE(temp, 'vehicle_type', ohe, mappings)
    X = []
    X.append(z_pow)
    X.append(z_km)
    X.append(z_age)
    X.append(damage)
    X.append(gear)
    #print(mappings['honda'][:-1])
    X1=np.append(mappings[brand][:-1],mappings[model1][:-1])
    X2 = np.append(mappings[fuel][:-1],mappings[vehicle][:-1])
    X3 = np.append(X1,X2).tolist()
    X4 = X+X3
    value = model.predict(np.array([X4]),verbose = 0)
    val = (value[0][0]*STD)+MEAN
    return val
value = str(prediction('golf','bmw','diesel','station wagon',0,0,193,150000,'24-12-1999'))
print('value', value)
#@a
