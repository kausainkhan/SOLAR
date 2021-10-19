import numpy as np
from sklearn.metrics import r2_score
import csv
import pandas
from sklearn import linear_model

x = []
y = []

with open('weather_data.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        x.append(row[0])
        # index value = Column Number
        y.append(row[7])

x.pop(0)
x = (np.array(x)).astype(np.float)
y.pop(0)
y = (np.array(y)).astype(np.float)

mymodel = np.poly1d(np.polyfit(x, y, 3))

input = input('Time')

humidity_prediction = mymodel(int(input))
print(humidity_prediction)


df = pandas.read_csv('weather_data.csv')

X = df[['Time [Mins]', 'Tower RH [%]']]
y = df['Total Cloud Cover [%]']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedTCC = regr.predict([[input, humidity_prediction]])

print(predictedTCC)