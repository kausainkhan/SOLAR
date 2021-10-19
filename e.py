import pandas
from sklearn import linear_model
import sklearn

df = pandas.read_csv('weather_data.csv')

X = df[['Time [Mins]']]
y = df[['Global CMP22 (vent/cor) [W/m^2]','Direct sNIP [W/m^2]','Azimuth Angle [degrees]','Tower Dry Bulb Temp [deg C]','Tower Wet Bulb Temp [deg C]','Tower Dew Point Temp [deg C]','Tower RH [%]','Total Cloud Cover [%]','Peak Wind Speed @ 6ft [m/s]','Avg Wind Direction @ 6ft [deg from N]','Station Pressure [mBar]','Precipitation (Accumulated) [mm]','Snow Depth [cm]','Moisture','Albedo (CMP11)']]

X=X.values.reshape(-1, 1)
regr = linear_model.LinearRegression()
regr.fit(X, y)

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[300]])

print(predictedCO2)