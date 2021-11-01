from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
import csv
import joblib

# loading model and scaler
model = load_model('model/model.h5')
x_scaler = joblib.load('./scaler/x_scaler.sav')
y_scaler = joblib.load('./scaler/y_scaler.sav')

# constants
SEQ_LEN = 124 # change in future
FUTURE_PERIOD_PREDICT = 30
TO_PREDICT = 'Total Cloud Cover [%]'



def get_predictions(prediction_df):
    
    x_pred_scaled = prediction_df

    # Input-signals for the model.
    x_pred_scaled = np.expand_dims(x_pred_scaled, axis=0)

    # # Use the model to predict the output-signals.
    y_pred = model.predict(x_pred_scaled)
    
    # rescale the predictions
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    
    return (y_pred_rescaled)

# getting data for predictions

list = os.listdir('./pred')
num_folders = len(list)

predictions = []


for i in range(num_folders):
    for k in range(4):
        j = i + 1
        path = './pred/' + str(j) + '/'
        if k == 0:
            prediction_df = pd.read_csv(path + 'weather_data.csv')
            prediction_df = prediction_df[TO_PREDICT]
            x = prediction_df
            #scaling the predictions
            x_pred = x.values.reshape(-1, 1)
            x_pred_scaled = x_scaler.fit_transform(x_pred)
            prediction_df = x_pred_scaled
        # print(i)
        initial_prediction = get_predictions(prediction_df)
        predictions.append(initial_prediction[len(initial_prediction)-1][0])
        prediction_df = initial_prediction

# splitting the predictions into 300 parts with 4 elements in each
l = 1

predictions_split = []

for i in range(len(predictions)):
    if i % 4 == 0:
        predictions_split_temp = predictions[i:i+4]
        predictions_split_temp.insert(0, l)
        predictions_split.append(predictions_split_temp)
        l = l + 1

# saving the predictions
with open('./test.csv', 'w',  newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(predictions_split)):
            print(predictions_split)
            writer.writerow(predictions_split[i])