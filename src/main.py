import numpy as np
from sklearn.metrics import r2_score
import csv
import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

from datetime import datetime as date

today = date.today()



for j in range(1, 2):
    
    prediction = []
    if j == 1:
        time = input('Enter Time Interval')
        time_org = time
    
    time = int(time_org) + 360

    print('\nPrediction from time:', time)

    for i in range(1, 16):

        x = []
        y = []
        

        with open(f'./temp.csv','r') as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            for row in lines:
                if not row[10] == '-1' or row[10] == '-7999':
                    x.append(row[0])
                    y.append(row[i + 2])

            if not row[10] == '-1' or row[10] == '-7999':
                x.pop(0)
                x = (np.array(x)).astype(float)
                
                print(y)
                y.pop(0)
                y = (np.array(y)).astype(float)


                mymodel = np.poly1d(np.polyfit(x, y, 50))

                prediction.append(mymodel(int(time)))
    
    if not row[10] == '-1' or row[10] == '-7999':

        df = pandas.read_csv(f'./temp.csv')

        X = df[['Global CMP22 (vent/cor) [W/m^2]','Azimuth Angle [degrees]','Tower Dry Bulb Temp [deg C]','Tower Wet Bulb Temp [deg C]','Tower Dew Point Temp [deg C]','Tower RH [%]','Peak Wind Speed @ 6ft [m/s]','Avg Wind Direction @ 6ft [deg from N]','Precipitation (Accumulated) [mm]','Moisture','Albedo (CMP11)']]
        Y = df['Total Cloud Cover [%]']

        regr = linear_model.LinearRegression()
        regr.fit(X.values, Y)
        
        print(prediction)
        # to do
        # import data from TEST CSV then use that data to predict the results
        for k in range(1, 5):
            predictedTCC = regr.predict([[prediction[0], prediction[2], prediction[3], prediction[4], prediction[5], prediction[6], prediction[7], prediction[9], prediction[10], prediction[12], prediction[14]]])
            print(f'Prediction for {time} minutes in {j} ',predictedTCC)
            time = time + 30
    
    

print('Calculation time:', date.today() - today)