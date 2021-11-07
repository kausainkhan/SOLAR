from keras.models import load_model
import numpy as np
import pandas as pd
import os
import csv
import joblib

# constants
SEQ_LEN = 30
FUTURE_PERIOD_PREDICT = 30
TO_PREDICT = 'Total Cloud Cover [%]'
INPUT_SIGNALS = ['Global CMP22 (vent/cor) [W/m^2]','Direct sNIP [W/m^2]','Azimuth Angle [degrees]','Tower Dry Bulb Temp [deg C]','Tower Wet Bulb Temp [deg C]','Tower Dew Point Temp [deg C]','Tower RH [%]','Total Cloud Cover [%]','Peak Wind Speed @ 6ft [m/s]','Avg Wind Direction @ 6ft [deg from N]','Station Pressure [mBar]','Precipitation (Accumulated) [mm]','Snow Depth [cm]','Moisture','Albedo (CMP11)']
BATCH_SIZE = 64
LEARNING_RATE = 0.001


# loading model and scaler
model = load_model(f'model/NEW-model-batch_size-300-learning_rate-0.0001.h5')
x_scaler = joblib.load('./scaler/x_scaler.sav')
y_scaler = joblib.load('./scaler/y_scaler.sav')




def get_predictions(prediction_df):
    
    x_pred_scaled = prediction_df

    start_idx = 0
    length = x_pred_scaled.shape[0]

    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.
    x_pred_scaled = x_pred_scaled[start_idx:end_idx]
    
    # Input-signals for the model.
    x_pred_scaled = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x_pred_scaled)
    
    # "Unscaling" the data
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
            prediction_df = prediction_df[INPUT_SIGNALS]
            x = prediction_df
            # #scaling the predictions
            # x_pred = x.values.reshape(1, -1)
            x_pred_scaled = x_scaler.fit_transform(x)
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
with open(F'./NEW-predictions-batch_size-{BATCH_SIZE}-learning_rate-{LEARNING_RATE}.csv', 'w',  newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(predictions_split)):
            print(predictions_split)
            writer.writerow(predictions_split[i])