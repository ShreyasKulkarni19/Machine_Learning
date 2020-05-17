import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from numpy import array, reshape, mean, std
from scipy.stats import zscore
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
df = pd.read_csv("cleaned_dataset.csv")
y = array(df['dollar_price'])

df = df.drop(['dollar_price','vehicle_bin', 'brand_bin', 'model_tag', 'postal_code', 'date_crawled', 'name', 'registration_year', 'gearbox', 'registration_month', 'unrepaired_damage', 'Unnamed: 0', 'last_seen_online', 'Diff_Last_Ad', 'ad_created'], axis=1)
temp = df.loc[:, ['model', 'brand', 'vehicle_type', 'fuel_type', 'damage_bin', 'gear_bin']]
df.drop(['model', 'brand', 'vehicle_type', 'fuel_type', 'damage_bin', 'gear_bin'], axis=1, inplace=True)
def myOHE(df, column, ohe_object):    
    # Encode the column
    column_encoded = array(temp[column]).reshape(-1, 1)
    column_encoded = ohe.fit_transform(column_encoded)
    
    # Add the attributes to the dataframe
    for i in range(len(column_encoded[0]) - 1):
        df['{}_{}'.format(column, str(i))] = column_encoded[:, i]
    
    # Drop the 'column' in the dataframe
    df.drop(column, axis=1, inplace=True)
    # Return the dataframe
    return df
    
ohe = OneHotEncoder(sparse=False)
temp = myOHE(temp, 'brand', ohe)
temp = myOHE(temp, 'model', ohe)
temp = myOHE(temp, 'fuel_type', ohe)
temp = myOHE(temp, 'vehicle_type', ohe)
#Creating a numpy array of the independent variables in the dataset
df = pd.DataFrame(zscore(df))
df = pd.concat([df, temp], axis=1)
X = array(df)

# Scale down the values of the cost
MEAN = mean(y)
STD = std(y)
y = zscore(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.142857, random_state=0)

model = Sequential()
model.add(Dense(128, input_dim=(X_train[0]).shape[0], activation='relu', kernel_initializer='normal'))
model.add(Dense(256, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.05))
model.add(Dense(256, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.05))
model.add(Dense(256, activation='relu', kernel_initializer='normal'))
model.add(Dense(1, activation='linear', kernel_initializer='normal'))
print(model.summary())

es = EarlyStopping(monitor='val_mean_absolute_error', min_delta=0.001, patience=20)
adam = Adam(clipnorm=1.)
model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['mean_absolute_error'])
model.fit(X_train, y_train, batch_size=128, epochs=150, verbose=1, validation_data=(X_val, y_val), callbacks=[es])
predictions = model.predict(X_test)

'''for i in range(10):
    print(predictions[i], y_test[i])

val = len(predictions)
count = 0
for i in range(val):
    val1 = 1.15 * predictions[i][0]
    val2 = 0.85 * predictions[i][0]
    if y_test[i] < 0 and y_test[i] >= val1 and y_test[i] <= val2:
        count += 1
    if y_test[i] <= val1 and y_test[i] >= val2:
        count += 1'''

result = r2_score(y_test, predictions)
print(result)
model.save("model.h5")
